import sys
import os
import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from datasets_my.gsm8k_dataset import gsm_data_process, gsm_get_predict

def parse_args():
    parser = argparse.ArgumentParser(description="Build Dsimple Dataset")
    parser.add_argument("--dataset_json", type=str, default="datasets_my/gsm8k/gsm8k.jsonl")
    parser.add_argument("--output_file", type=str, default="experiments/dsimple.jsonl")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of questions to process")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def run_template(graph, task, true_answer, template_name):
    """
    Run a specific template and return the entry if successful.
    """
    # Define templates
    # Simple-1: Solver -> Finisher (DecisionMaker)
    # T_max=2
    # top_m(Solver)=0 (No context)
    # top_m(Finisher)=1 (Solver output)
    
    # Simple-2: Solver -> Inspector -> Finisher
    # T_max=3
    # top_m(Solver)=0
    # top_m(Verifier)=1 (Solver)
    # top_m(Finisher)=2 (Solver + Verifier)
    
    # Simple-3: Analyst -> Solver -> Finisher
    # T_max=3
    # top_m(Analyst)=0
    # top_m(Solver)=1 (Analyst)
    # top_m(Finisher)=1 (Solver only)
    
    forced_seq = []
    forced_masks = [] # List of lists. forced_masks[i] is for step i+1.
    
    if template_name == "simple-1":
        forced_seq = ["Math Solver"]
        # Step 0 (Solver): No context (auto)
        # Step 1 (Finisher): Context [1] (Solver output). history len 1.
        forced_masks = [[1]] 
        
    elif template_name == "simple-2":
        forced_seq = ["Math Solver", "Inspector"]
        # Step 0 (Solver): No context
        # Step 1 (Inspector): Context [1] (Solver). history len 1.
        # Step 2 (Finisher): Context [1, 1] (Solver, Inspector). history len 2.
        forced_masks = [[1], [1, 1]]
        
    elif template_name == "simple-3":
        forced_seq = ["Mathematical Analyst", "Math Solver"]
        # Step 0 (Analyst): No context
        # Step 1 (Solver): Context [1] (Analyst). history len 1.
        # Step 2 (Finisher): Context [0, 1] (Ignore Analyst, Keep Solver). history len 2.
        # Note: History order is [Analyst, Solver]. So [0, 1] keeps Solver.
        forced_masks = [[1], [0, 1]]
        
    else:
        return None

    # Run Graph
    input_dict = {"task": task}
    try:
        # T_max needs to accommodate the sequence + decision maker
        # len(forced_seq) steps + 1 step for DecisionMaker
        # So max_routing should be len(forced_seq)
        
        result = graph.run_next_agent_prediction(
            input_dict,
            max_routing=len(forced_seq),
            temperature=0.0, # Greedy for templates
            available_roles=["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"],
            training=True, # To avoid timeout logic if any
            forced_agent_sequence=forced_seq,
            forced_context_masks=forced_masks
        )
    except Exception as e:
        print(f"Error running template {template_name}: {e}")
        return None

    if result is None:
        return None
        
    # Check correctness
    predict_answer_list = result.get("answers", [""])
    predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
    predict_val = gsm_get_predict(predict_answer_str)
    
    is_correct = (str(predict_val) == str(true_answer))
    
    if is_correct:
        routing_results = result["routing_results"]
        
        # Process indices
        selected_context_turn_ids = []
        for step_mask in routing_results.get("hint_selections", []):
            indices = [idx for idx, val in enumerate(step_mask) if val == 1]
            selected_context_turn_ids.append(indices)

        entry = {
            "task": task,
            "answer": true_answer,
            "predicted_answer": predict_answer_str,
            "template": template_name,
            "agent_selections": routing_results.get("agent_selections", []),
            "selected_context_turn_ids": selected_context_turn_ids,
            "ncs_gates": routing_results.get("hint_selections", []),
            "history_trace": routing_results.get("history_trace", [])
            # "router_logits": ... (not useful for forced templates)
        }
        return entry
    return None

def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load Dataset
    dataset_path = Path(args.dataset_json)
    if not dataset_path.is_absolute():
        dataset_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.dataset_json)))
        
    print(f"Loading dataset from {dataset_path}")
    dataset = JSONLReader.parse_file(str(dataset_path))
    dataset = gsm_data_process(dataset)
    
    # Initialize Graph
    agent_names = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    graph = Graph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method="FinalRefer",
        use_transformer=True,
        max_routing=10,
        available_roles=agent_names
    )
    graph.set_eval()
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Building Dsimple at {output_path}")
    
    collected_count = 0
    templates = ["simple-1", "simple-2", "simple-3"]
    
    # We loop through dataset. For each question, try all templates.
    # Keep ALL successful templates (one question can generate multiple entries).
    
    for i, record in enumerate(tqdm(dataset[:args.num_samples], desc="Processing Questions")):
        task = record["task"]
        true_answer = record["answer"]
        
        for tmpl in templates:
            entry = run_template(graph, task, true_answer, tmpl)
            if entry:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                collected_count += 1
                
    print(f"Finished. Collected {collected_count} Dsimple trajectories.")

if __name__ == "__main__":
    main()
