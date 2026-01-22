import sys
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import copy

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GDesigner.graph.graph import Graph
from datasets_my.gsm8k_dataset import gsm_get_predict

def parse_args():
    parser = argparse.ArgumentParser(description="Prune Context from Pruned Steps")
    parser.add_argument("--input_file", type=str, default="experiments/traj_pruned_steps.jsonl")
    parser.add_argument("--output_file", type=str, default="experiments/traj_pruned_context.jsonl")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def replay(graph, task, true_answer, agent_sequence, context_masks=None):
    # Same replay function as before
    input_dict = {"task": task}
    
    forced_seq_names = []
    if agent_sequence and isinstance(agent_sequence[0], int):
        for idx in agent_sequence:
            role = graph.idx_to_role[idx]
            forced_seq_names.append(role)
    else:
        forced_seq_names = agent_sequence
        
    decision_role = graph.decision_method
    if forced_seq_names and forced_seq_names[-1] == decision_role:
        forced_seq_names = forced_seq_names[:-1]
        
    try:
        result = graph.run_next_agent_prediction(
            input_dict,
            max_routing=len(forced_seq_names),
            temperature=0.0,
            available_roles=graph.available_roles,
            training=True,
            forced_agent_sequence=forced_seq_names,
            forced_context_masks=context_masks
        )
    except Exception as e:
        return False, None, None

    if result is None:
        return False, None, None
        
    predict_answer_list = result.get("answers", [""])
    predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
    predict_val = gsm_get_predict(predict_answer_str)
    
    is_correct = (str(predict_val) == str(true_answer))
    return is_correct, result, predict_answer_str

def main():
    args = parse_args()
    
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
    
    entries = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
                
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Pruning context for {len(entries)} trajectories...")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in tqdm(entries):
            task = entry["task"]
            true_answer = entry["answer"]
            agent_seq = entry["agent_selections"] # Indices
            
            # Initial masks from the pruned steps run
            # Note: "ncs_gates" field in entry corresponds to the masks used.
            # But we should start with "All Context" or "Current Context"?
            # User says: "1. Set current context set C_t (originally selected turn ids)".
            # So we start with the masks in `ncs_gates` (or `selected_context_turn_ids`).
            
            current_masks = entry.get("ncs_gates", [])
            # Convert to list of lists if needed
            # Ensure it matches the number of steps that need context
            # Agent seq length N. Steps needing context: indices 1..N-1 (Step 0 no context).
            # The `ncs_gates` list usually has N entries?
            # Let's check `run_next_agent_prediction` output structure.
            # `hint_selections` appends a mask for every step where history > 0.
            # Step 0: history=0. No mask appended.
            # Step 1: history=1. Mask appended.
            # So `ncs_gates` has length N-1 (if N is number of agents excluding decision maker?).
            # Wait, `agent_selections` includes the decision maker step?
            # In `run_next_agent_prediction`:
            # Loop `max_routing` times.
            # If `forced_agent_sequence` provided, we run `len(seq)` times.
            # If seq has 3 agents (A, B, C).
            # i=0: Agent A. history=0.
            # i=1: Agent B. history=1. hint_mask stored.
            # i=2: Agent C. history=2. hint_mask stored.
            # Total hints stored: 2.
            # `agent_selections` stores 3 indices.
            
            # So `ncs_gates` should have length `len(agent_seq) - 1`.
            
            # If `ncs_gates` is missing or wrong length, we might need to regenerate it or assume full/empty?
            # Let's assume it's correct from `03_prune_steps`.
            
            # We iterate over each step t (index in ncs_gates)
            # t=0 corresponds to Step 1 (context from Step 0). Mask length 1.
            # t=1 corresponds to Step 2. Mask length 2.
            
            # We need to construct `forced_context_masks` for the replay.
            # Initially `forced_context_masks = current_masks`.
            
            forced_context_masks = copy.deepcopy(current_masks)
            
            # Greedy Pruning per step
            # Iterate steps
            for t in range(len(forced_context_masks)):
                mask = forced_context_masks[t]
                # mask is list of 0/1. Indices correspond to history items 0..t
                
                # Identify selected indices (where mask[i] == 1)
                selected_indices = [i for i, val in enumerate(mask) if val == 1]
                
                # Try removing each selected index
                # Order: User says "Start from earliest or lowest gate".
                # Let's try removing from earliest (index 0).
                
                indices_to_try = list(selected_indices)
                
                for idx_to_remove in indices_to_try:
                    # Create trial mask
                    trial_mask = list(mask)
                    trial_mask[idx_to_remove] = 0
                    
                    # Update forced_context_masks temporarily
                    original_mask_t = forced_context_masks[t]
                    forced_context_masks[t] = trial_mask
                    
                    # Replay
                    success, _, _ = replay(graph, task, true_answer, agent_seq, forced_context_masks)
                    
                    if success:
                        # Keep removal
                        mask = trial_mask
                        # forced_context_masks[t] is already updated
                    else:
                        # Revert
                        forced_context_masks[t] = original_mask_t
                        mask = original_mask_t
                        
            # Final Replay to get updated entry
            success, final_res, final_ans = replay(graph, task, true_answer, agent_seq, forced_context_masks)
            
            if success:
                routing_results = final_res["routing_results"]
                
                selected_context_turn_ids = []
                for step_mask in routing_results.get("hint_selections", []):
                    indices = [idx for idx, val in enumerate(step_mask) if val == 1]
                    selected_context_turn_ids.append(indices)
                
                # We need "target_context_mask" and "target_context_turn_ids" as requested
                # Output format: traj_pruned_context.jsonl
                
                new_entry = copy.deepcopy(entry)
                new_entry["target_context_mask"] = routing_results.get("hint_selections", [])
                new_entry["target_context_turn_ids"] = selected_context_turn_ids
                new_entry["predicted_answer"] = final_ans
                new_entry["history_trace"] = routing_results.get("history_trace", [])
                
                f_out.write(json.dumps(new_entry) + "\n")
            else:
                print(f"Warning: Context pruning failed verification for task: {task[:20]}...")

if __name__ == "__main__":
    main()
