import sys
import os
import argparse
import json
import random
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="Build Deff Dataset")
    parser.add_argument("--dexp_file", type=str, default="experiments/dexp.jsonl")
    parser.add_argument("--dsimple_file", type=str, default="experiments/dsimple.jsonl")
    parser.add_argument("--dpruned_file", type=str, default="experiments/traj_pruned_context.jsonl")
    parser.add_argument("--output_file", type=str, default="experiments/deff.jsonl")
    parser.add_argument("--replay_ratio", type=float, default=0.2, help="Ratio of Dexp to keep for Dreplay (0.2 = 20%)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # 1. Load Datasets
    print(f"Loading Dexp from {args.dexp_file}")
    dexp = load_jsonl(args.dexp_file)
    
    print(f"Loading Dsimple from {args.dsimple_file}")
    dsimple = load_jsonl(args.dsimple_file)
    
    print(f"Loading Dpruned from {args.dpruned_file}")
    dpruned = load_jsonl(args.dpruned_file)
    
    # 2. Construct Dreplay
    # Randomly sample p% from Dexp
    num_replay = int(len(dexp) * args.replay_ratio)
    dreplay = random.sample(dexp, num_replay) if dexp else []
    print(f"Sampled {len(dreplay)} entries for Dreplay ({args.replay_ratio*100}%)")
    
    # 3. Assemble Deff
    # Deff = Dsimple + Dpruned + Dreplay
    # Note: Dpruned is derived from Dexp, so there might be overlap with Dreplay (content-wise), 
    # but Dpruned has modified structure (shorter steps, sparser context).
    # Dreplay has original structure.
    # This is intended ("Avoid forgetting").
    
    deff = dsimple + dpruned + dreplay
    
    # Shuffle
    random.shuffle(deff)
    
    # 4. Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Deff ({len(deff)} entries) to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in deff:
            f.write(json.dumps(entry) + "\n")
            
    print("Deff construction complete.")

if __name__ == "__main__":
    main()
