import sys
import os
import torch
import numpy as np
import asyncio
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GDesigner.graph.graph import Graph
from GDesigner.llm.llm import LLM
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens, Time
from datasets_my.gsm8k_dataset import gsm_get_predict

from transformers import GPT2Config, GPT2Model

# Mock LLM
class MockLLM(LLM):
    def __init__(self, model_name="mock"):
        self.model_name = model_name

    def gen(self, messages, **kwargs):
        # Return a response that contains the answer "2" for "1+1" or similar simple math
        return "Reasoning... The answer is 2"

    async def agen(self, messages, **kwargs):
        return "Reasoning... The answer is 2"

# Mock Embedding function
def mock_get_embedding(text):
    # Return a random embedding of size 384 (default for all-MiniLM-L6-v2)
    # Use a deterministic seed based on text length for consistency if needed, 
    # but for flow test random is fine.
    np.random.seed(len(text))
    return np.random.rand(384).astype(np.float32)

# Mock GPT2Model.from_pretrained to avoid downloading
def mock_gpt2_from_pretrained(model_name):
    config = GPT2Config(n_layer=2, n_head=4, n_embd=768) # Tiny config for speed
    return GPT2Model(config)

def run_gsm8k_flow():
    print("\n--- Starting GSM8K Flow ---")
    
    # 1. Setup Data (Minimal Dataset)
    train_dataset = [{"task": "What is 1 + 1?", "answer": "2", "step": "1+1=2"}]
    test_dataset = [{"task": "What is 2 * 1?", "answer": "2", "step": "2*1=2"}]
    
    # 2. Initialize Graph
    agent_names = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    # Expand agent names based on counts (1 each for simplicity)
    expanded_agent_names = agent_names 
    
    graph = Graph(
        domain="gsm8k",
        llm_name="mock-llm",
        agent_names=expanded_agent_names,
        decision_method="FinalRefer",
        use_transformer=True,
        max_routing=3,
        available_roles=agent_names
    )
    
    # Setup Optimizer
    optimizer = torch.optim.AdamW(graph.routing_transformer.get_gpt_parameters(), lr=1e-4)
    graph.add_to_optimizer(optimizer)
    graph.routing_transformer.train()

    # 3. Training Loop Simulation (1 Epoch, 1 Step)
    print("--- Testing Training Loop ---")
    graph.set_eval() # Sampling phase uses eval mode
    
    # Phase 1: Sampling (Collect Traces)
    all_gradient_inputs = []
    with torch.no_grad():
        for record in train_dataset:
            input_dict = {"task": record["task"]}
            # Run prediction to get trace
            trace_result = graph.run_next_agent_prediction(
                input_dict,
                max_routing=3,
                temperature=1.0,
                available_roles=agent_names,
                training=True
            )
            
            if trace_result is None:
                print("Warning: Trace result is None")
                continue
            
            # Verify trace structure
            if "answers" not in trace_result or "routing_results" not in trace_result:
                print("Warning: Missing required keys in trace result")
                continue
                
            # Mock reward calculation
            predict_answer = trace_result["answers"][0] if trace_result["answers"] else ""
            # MockLLM returns "Reasoning... The answer is 2", gsm_get_predict should extract "2"
            pred_val = gsm_get_predict(predict_answer)
            is_correct = (pred_val == record["answer"])
            reward = 1.0 if is_correct else 0.0
            
            print(f"Training sample: {record['task']} -> Predicted: {pred_val}, Expected: {record['answer']}, Correct: {is_correct}")
            
            routing_results = trace_result["routing_results"]
            # Construct gradient input (simplified)
            if routing_results and 'agent_selections' in routing_results and routing_results['agent_selections']:
                gradient_input = {
                    'task': record['task'],
                    'advantage': reward - 0.5, # Mock advantage
                    'agent_selections': routing_results['agent_selections'],
                    'hint_selections': routing_results['hint_selections'],
                    'agent_logits': routing_results.get('agent_logits', []),
                    'hint_logits': routing_results.get('hint_logits', []),
                    'trace_length': len(routing_results['agent_selections']),
                    'agent_outputs_embeddings': routing_results.get('agent_outputs_embeddings', []),
                    'batch_record_idx': 0,
                    'trace_idx_in_record': 0
                }
                all_gradient_inputs.append(gradient_input)

    # Phase 2: Training (Update Weights)
    print(f"Collected {len(all_gradient_inputs)} traces for training.")
    if all_gradient_inputs:
        graph.set_train()
        optimizer.zero_grad()
        for grad_input in all_gradient_inputs:
            graph.run_next_agent_prediction_grad(grad_input)
        optimizer.step()
        print("Optimizer step executed.")

    # 4. Save Model
    test_dir = Path("tes/temp_results")
    test_dir.mkdir(parents=True, exist_ok=True)
    model_path = test_dir / "test_model.pth"
    graph.save_model(model_path)
    print(f"Model saved to {model_path}")

    # 5. Load Model
    loaded_graph = Graph.load_model(model_path)
    if loaded_graph is None:
        print("Error: Failed to load graph from saved model")
        return
    print("Model loaded successfully")

    # 6. Inference Loop
    print("--- Testing Inference Loop ---")
    loaded_graph.set_eval()
    with torch.no_grad():
        for record in test_dataset:
            input_dict = {"task": record["task"]}
            inference_result = loaded_graph.run_next_agent_prediction(
                input_dict,
                max_routing=3,
                temperature=0.0, # Greedy
                available_roles=agent_names
            )
            if inference_result is not None:
                print(f"Inference Answer: {inference_result['answers']}")
            else:
                print("Inference result is None")

    print("--- GSM8K Flow Completed Successfully ---")

if __name__ == '__main__':
    run_gsm8k_flow()