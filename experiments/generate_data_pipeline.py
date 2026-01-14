import subprocess
import argparse
import sys
import os

def run_step(script_name, description, args_list):
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"Running: {script_name} {' '.join(args_list)}")
    print(f"{'='*50}\n")
    
    cmd = [sys.executable, script_name] + args_list
    result = subprocess.run(cmd, capture_output=False) # Let stdout flow to console
    
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="AnyMAC Data Generation Pipeline")
    parser.add_argument("--gsm8k_file", type=str, default="datasets_my/gsm8k/gsm8k.jsonl")
    parser.add_argument("--output_dir", type=str, default="experiments/data")
    parser.add_argument("--num_queries", type=int, default=80, help="Number of distinct questions to use")
    parser.add_argument("--target_dexp", type=int, default=20, help="Target number of Dexp trajectories (default small for demo)")
    parser.add_argument("--target_dsimple", type=int, default=20, help="Target number of Dsimple trajectories (approx)")
    parser.add_argument("--replay_ratio", type=float, default=0.2)
    args = parser.parse_args()
    
    # Ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths
    dexp_file = os.path.join(args.output_dir, "dexp.jsonl")
    dsimple_file = os.path.join(args.output_dir, "dsimple.jsonl")
    pruned_steps_file = os.path.join(args.output_dir, "traj_pruned_steps.jsonl")
    pruned_context_file = os.path.join(args.output_dir, "traj_pruned_context.jsonl")
    deff_file = os.path.join(args.output_dir, "deff.jsonl")
    
    # 1. Rollout Dexp
    run_step("experiments/01_rollout_dexp.py", "Generating Dexp (Exploration Set)", [
        "--dataset_json", args.gsm8k_file,
        "--output_file", dexp_file,
        "--num_queries", str(args.num_queries),
        "--target_success", str(args.target_dexp)
    ])
    
    # 2. Build Dsimple
    run_step("experiments/02_build_dsimple.py", "Generating Dsimple (Templates)", [
        "--dataset_json", args.gsm8k_file,
        "--output_file", dsimple_file,
        "--num_samples", str(args.target_dsimple) # This script uses num_samples as num questions to scan
    ])
    
    # 3. Prune Steps
    run_step("experiments/03_prune_steps.py", "Pruning Steps (Dexp -> Dpruned_steps)", [
        "--input_file", dexp_file,
        "--output_file", pruned_steps_file
    ])
    
    # 4. Prune Context
    run_step("experiments/04_prune_context.py", "Pruning Context (Dpruned_steps -> Dpruned_context)", [
        "--input_file", pruned_steps_file,
        "--output_file", pruned_context_file
    ])
    
    # 5. Build Deff
    run_step("experiments/05_build_deff.py", "Assembling Deff (Efficiency Set)", [
        "--dexp_file", dexp_file,
        "--dsimple_file", dsimple_file,
        "--dpruned_file", pruned_context_file,
        "--output_file", deff_file,
        "--replay_ratio", str(args.replay_ratio)
    ])
    
    print(f"\nPipeline Complete! Final dataset saved to: {deff_file}")

if __name__ == "__main__":
    main()
