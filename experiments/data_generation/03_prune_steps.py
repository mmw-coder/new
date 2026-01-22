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
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prune Steps from Dexp")
    parser.add_argument("--input_file", type=str, default="experiments/dexp.jsonl")
    parser.add_argument("--output_file", type=str, default="experiments/traj_pruned_steps.jsonl")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def replay(graph, task, true_answer, agent_sequence, context_masks=None):
    """
    Replay a trajectory with fixed agent sequence.
    Forces 'Full Context' if context_masks is None.
    """
    input_dict = {"task": task}

    # We need to map agent indices back to role names
    # Check if agent_sequence contains names or indices
    forced_seq_names = []
    if agent_sequence and isinstance(agent_sequence[0], int):
        # Map indices to names
        for idx in agent_sequence:
            role = graph.idx_to_role[idx]
            forced_seq_names.append(role)
    else:
        forced_seq_names = agent_sequence

    # Check if the last agent in sequence is DecisionMaker (index or name)
    decision_role = graph.decision_method
    if forced_seq_names and forced_seq_names[-1] == decision_role:
        forced_seq_names = forced_seq_names[:-1]

    # Force Full Context if not provided
    # Full Context means: for step i (1-indexed), use all available history (length i).
    # forced_context_masks should be list of lists.
    # forced_context_masks[k] is mask for routing step k+1 (which has history length k+1? No, k).
    # run_next_agent_prediction loop:
    # routing_count starts at 0.
    # Step 0 (Agent 1): History []. No mask used.
    # Step 1 (Agent 2): History [Agent1_Out]. Length 1. Mask needed.
    # ...
    # Step N (Agent N+1/Decision): History [A1...AN]. Length N. Mask needed.

    # The length of forced_seq_names is M.
    # This means we run M steps of agents, then 1 step of DecisionMaker.
    # Total routing steps = M + 1 (DecisionMaker).
    # Context masks needed for steps 1 to M. (Step 0 has no history).
    # So we need M masks.
    # Mask k corresponds to history length k+1.
    # wait, step 0 has history 0.
    # step 1 has history 1. Mask should be [1].
    # step k has history k. Mask should be [1] * k.

    if context_masks is None:
        # Generate full context masks for steps 1 to M (and potentially decision maker step M+1)
        # We need masks for every step that HAS history.
        # i.e. steps 1, 2, ..., M.
        # Total M masks.
        # Mask i (0-indexed in list) corresponds to history length i+1.

        # We need to cover up to len(forced_seq_names) + 1 (decision maker) steps?
        # run_next_agent_prediction uses masks for any step with history > 0.
        # Total steps = len(forced_seq_names) agents + 1 decision maker.
        # History grows: 0, 1, 2, ..., M.
        # Masks needed for history sizes: 1, 2, ..., M.
        # So we need M masks.

        num_steps_with_history = len(forced_seq_names)
        context_masks = [[1] * (i + 1) for i in range(num_steps_with_history)]

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
        # print(f"Replay error: {e}")
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

    # Load Dexp
    dexp_entries = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dexp_entries.append(json.loads(line))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pruning steps for {len(dexp_entries)} trajectories...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in tqdm(dexp_entries):
            task = entry["task"]
            true_answer = entry["answer"]
            original_seq = entry["agent_selections"]  # Indicesasd

            # Remove DecisionMaker from end if present (usually last index is decision maker)
            # Check graph.role_to_idx[decision_role]
            dm_idx = graph.role_to_idx.get(graph.decision_method)
            if dm_idx is not None and original_seq and original_seq[-1] == dm_idx:
                current_seq = original_seq[:-1]
            else:
                current_seq = original_seq

            # Greedy Pruning
            # Start from middle? User: "Avoid deleting first/last immediately".
            # Let's iterate index k from 1 to len-2. (Indices 0..N-1)
            # If len is small (e.g. 1 or 2), maybe try index 0?
            # Let's just try all indices, but maybe prioritize middle?
            # Or just iterate 0 to N-1.

            # User recommendation: "Start from middle".
            # Let's create a list of indices to try removing.
            # e.g. [1, 2, ..., N-2, 0, N-1]

            n = len(current_seq)
            if n == 0:
                # Empty sequence (only decision maker), nothing to prune
                json.dump(entry, f_out)
                f_out.write("\n")
                continue

            indices_to_try = list(range(1, n - 1)) + [0, n - 1]
            # Filter valid indices
            indices_to_try = [i for i in indices_to_try if i < n]
            # Remove duplicates if n is small
            indices_to_try = sorted(list(set(indices_to_try)), key=lambda x: (1 if 0 < x < n - 1 else 2))

            pruned_any = False

            # We need to loop. Since we delete items, indices shift.
            # Easier to loop: Try to remove index i. If success, current_seq becomes shorter.
            # Repeat until no deletion works.

            stable = False
            while not stable:
                stable = True
                # Generate candidate removals
                # We try removing one step from current_seq

                # Heuristic: iterate through current_seq indices
                # To follow user order: Middle first.
                n_curr = len(current_seq)
                if n_curr == 0:
                    break

                check_indices = list(range(n_curr))
                # Sort to prioritize middle: distance from center
                center = n_curr / 2
                check_indices.sort(key=lambda x: abs(x - center))

                for idx in check_indices:
                    # Construct trial sequence
                    trial_seq = current_seq[:idx] + current_seq[idx + 1:]

                    # Replay
                    success, res, _ = replay(graph, task, true_answer, trial_seq)

                    if success:
                        current_seq = trial_seq
                        stable = False  # Modified, restart loop
                        pruned_any = True
                        break  # Break inner loop to restart with new sequence

            # Save pruned entry
            # We need to run one final replay to get the full routing results (logits, context etc) for the pruned seq
            # (If we just updated current_seq, we might not have the result object)
            success, final_res, final_ans = replay(graph, task, true_answer, current_seq)

            if success:
                routing_results = final_res["routing_results"]

                # Re-extract info
                selected_context_turn_ids = []
                for step_mask in routing_results.get("hint_selections", []):
                    indices = [idx for idx, val in enumerate(step_mask) if val == 1]
                    selected_context_turn_ids.append(indices)

                new_entry = {
                    "task": task,
                    "answer": true_answer,
                    "predicted_answer": final_ans,
                    "agent_selections": routing_results.get("agent_selections", []),
                    "selected_context_turn_ids": selected_context_turn_ids,
                    "ncs_gates": routing_results.get("hint_selections", []),
                    "router_logits": routing_results.get("agent_logits", []),
                    "ncs_logits": routing_results.get("ncs_logits", []),
                    "hint_logits": routing_results.get("hint_logits", []),
                    "history_trace": routing_results.get("history_trace", []),
                    "original_seq_len": len(original_seq),
                    "pruned_seq_len": len(current_seq)
                }
                f_out.write(json.dumps(new_entry) + "\n")
            else:
                # If final replay fails (shouldn't happen if logic is correct), save original?
                # Or just skip.
                print(f"Warning: Pruned sequence failed verification for task: {task[:20]}...")


if __name__ == "__main__":
    main()
