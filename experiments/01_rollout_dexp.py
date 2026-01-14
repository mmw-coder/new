import sys
import os
import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.tools.reader.readers import JSONLReader
from datasets_my.gsm8k_dataset import gsm_data_process, gsm_get_predict

def parse_args():
    parser = argparse.ArgumentParser(description="Dexp Rollout for AnyMAC")
    parser.add_argument("--dataset_json", type=str, default="datasets_my/gsm8k/gsm8k.jsonl")
    parser.add_argument("--output_file", type=str, default="dexp.jsonl")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_queries", type=int, default=80, help="Number of distinct questions to use from dataset")
    parser.add_argument("--target_success", type=int, default=1000, help="Target number of successful trajectories to collect")
    parser.add_argument("--max_attempts_per_question", type=int, default=5, help="Max attempts to find a correct path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load Dataset - Use absolute path resolution
    dataset_path = Path(args.dataset_json)
    if not dataset_path.is_absolute():
        dataset_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.dataset_json)))
    
    print(f"Loading dataset from {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    dataset = JSONLReader.parse_file(str(dataset_path))
    dataset = gsm_data_process(dataset)
    
    # Initialize Graph
    agent_names = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    # We use counts of 1 for simplicity in this rollout, or matching run_gsm8k defaults
    expanded_agent_names = agent_names * 1 
    
    graph = Graph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=expanded_agent_names,
        decision_method="FinalRefer",
        use_transformer=True,
        max_routing=10, # Will be overridden per call
        available_roles=agent_names,
        optimized_spatial=False,
        optimized_temporal=False
    )
    
    # Set to Eval mode (we are collecting data, but using randomized exploration)
    graph.set_eval()
    
    # Output file
    output_path = Path(args.output_file)
    print(f"Writing results to {output_path}")
    
    # Ensure parent dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collected_count = 0
    
    # Iterate over dataset
    # We want to collect target_success trajectories total from num_queries queries.
    # We will loop through the first num_queries queries repeatedly until we reach the target.
    
    queries_to_use = dataset[:args.num_queries] 
    
    pbar = tqdm(total=args.target_success, desc="Collecting Dexp Trajectories")
    
    while collected_count < args.target_success:
        # Iterate through our subset of queries
        for i, record in enumerate(queries_to_use):
            if collected_count >= args.target_success:
                break
                
            task = record["task"]
            true_answer = record["answer"]
            
            # Try once per query in this pass (or multiple if we want speed, but distributing across passes is fine)
            # 1. Randomize Parameters
            # T_max in [5, 7]
            t_max = 5
            # eta in [0.0, 0.2]
            eta = random.uniform(0.0, 0.2)
            
            # 2. Run Inference
            input_dict = {"task": task}
            
            try:
                result = graph.run_next_agent_prediction(
                    input_dict,
                    max_routing=t_max,
                    temperature=1.0, 
                    available_roles=agent_names,
                    ncs_threshold=eta,
                    training=True 
                )
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                continue
            
            if result is None:
                continue


            # 3. Check Correctness
            predict_answer_list = result.get("answers", [""])
            predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
            predict_val = gsm_get_predict(predict_answer_str)
            
            is_correct = (str(predict_val) == str(true_answer))
            
            # 4. Save if Correct (Positive Trajectory)
            if is_correct:
                routing_results = result["routing_results"]
                
                # Process hint_selections (binary masks) into indices
                selected_context_turn_ids = []
                for step_mask in routing_results.get("hint_selections", []):
                    indices = [idx for idx, val in enumerate(step_mask) if val == 1]
                    selected_context_turn_ids.append(indices)

                # Construct output entry
                entry = {
                    "task": task,
                    "answer": true_answer,
                    "predicted_answer": predict_answer_str,
                    "t_max": t_max,
                    "eta": eta,
                    "agent_selections": routing_results.get("agent_selections", []),
                    "selected_context_turn_ids": selected_context_turn_ids,
                    "ncs_gates": routing_results.get("hint_selections", []), # Binary masks
                    "router_logits": routing_results.get("agent_logits", []),
                    "ncs_logits": routing_results.get("ncs_logits", []),
                    "hint_logits": routing_results.get("hint_logits", []),
                    "history_trace": routing_results.get("history_trace", [])
                }
                
                # Append to JSONL
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

                collected_count += 1
                pbar.update(1)
                
    pbar.close()
    print(f"Finished. Collected {collected_count} valid trajectories.")

if __name__ == "__main__":
    main()