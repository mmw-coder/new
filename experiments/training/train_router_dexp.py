import sys
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GDesigner.graph.graph import Graph

def parse_args():
    parser = argparse.ArgumentParser(description="Train Router Stage 1: Cold Start (Deff)")
    parser.add_argument("--data_file", type=str, default="experiments/experiments/data/deff.jsonl", help="Path to training data (jsonl)")
    parser.add_argument("--save_path", type=str, default="checkpoints/router_dexp.pt")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--alpha", type=float, default=0.5, help="BiRouter fusion weight")
    parser.add_argument("--seed", type=int, default=42)
    
    # Stage 1 Defaults
    parser.add_argument("--lambda_sparse", type=float, default=0.01, help="Sparsity penalty (Low for Stage 1)")
    parser.add_argument("--w_ncs", type=float, default=0.1, help="NCS weight (Low for Stage 1)")
    parser.add_argument("--load_from", type=str, default=None, help="Path to pretrained model checkpoint (optional)")
    
    return parser.parse_args()

def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Stage 1 Training: Dexp -> Basic Routing. w_ncs={args.w_ncs}, lambda={args.lambda_sparse}")

    # Load Data
    dataset = load_data(args.data_file)
    print(f"Loaded {len(dataset)} samples.")
    
    # Initialize Graph
    agent_names = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    graph = Graph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method="FinalRefer",
        use_transformer=True,
        max_routing=5,
        available_roles=agent_names
    )
    
    # Load pretrained weights if specified
    if args.load_from:
        graph.load_weights(args.load_from)
    
    # Optimizer
    # Re-create optimizer with correct params group logic from graph
    optimizer = torch.optim.Adam([{'params': graph.routing_transformer.parameters()}], lr=args.lr)
    graph.optimizer = optimizer
    # Add other components
    optimizer.add_param_group({"params": graph.proj_to_transformer_dim_history.parameters()})
    optimizer.add_param_group({"params": graph.proj_to_transformer_dim_task.parameters()})
    optimizer.add_param_group({"params": graph.proj_to_transformer_dim_role.parameters()})
    optimizer.add_param_group({"params": graph.linear_imp.parameters()})
    optimizer.add_param_group({"params": graph.linear_gap.parameters()})
    optimizer.add_param_group({"params": graph.ncs_mlp.parameters()})
    
    graph.set_train()
    
    # Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(dataset)
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, sample in enumerate(pbar):
            # Extract fields
            input_dict = {"task": sample["task"]}
            target_agent_ids = sample["agent_selections"]
            
            # Target masks: use 'ncs_gates' for Dexp
            if "target_context_mask" in sample:
                target_masks = sample["target_context_mask"]
            else:
                target_masks = sample.get("ncs_gates", [])
                
            history_trace = sample.get("history_trace", [])
            
            # If data is inconsistent, skip
            if not target_agent_ids:
                continue
                
            # Run SFT Step
            try:
                loss, metrics = graph.train_step_sft(
                    input_dict=input_dict,
                    target_agent_ids=target_agent_ids,
                    target_masks=target_masks,
                    history_trace=history_trace,
                    alpha=args.alpha,
                    lambda_sparse=args.lambda_sparse,
                    w_ncs=args.w_ncs
                )
                
                # Normalize loss by batch size for accumulation
                loss = loss / args.batch_size
                loss.backward()
                
                epoch_loss += loss.item() * args.batch_size
                
                # Gradient Accumulation
                if (i + 1) % args.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                pbar.set_postfix({
                    "loss": f"{loss.item() * args.batch_size:.4f}",
                    "nap": f"{metrics['loss_nap']:.4f}",
                    "ncs": f"{metrics['loss_ncs']:.4f}"
                })
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataset):.4f}")
        
        # Save checkpoint
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        graph.save_model(args.save_path)
        print(f"Saved checkpoint to {args.save_path}")

if __name__ == "__main__":
    main()
