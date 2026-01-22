import sys
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from datasets_my.gsm8k_dataset import gsm_data_process, gsm_get_predict

def parse_args():
    parser = argparse.ArgumentParser(description="Test Router on GSM8K")
    parser.add_argument("--model_path", type=str, default="checkpoints/router_dexp.pt", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data_file", type=str, default=r"E:\anymac\new\datasets_my\gsm8k\gsm8k.jsonl", help="Path to GSM8K test data")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--test_samples", type=int, default=10, help="Number of samples to test (-1 for all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_routing", type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Testing Router Model: {args.model_path}")
    print(f"Dataset: {args.data_file}")
    
    # 1. Load Data
    # Use the same data loading logic as run_gsm8k.py to ensure consistency
    try:
        raw_dataset = JSONLReader.parse_file(args.data_file)
        dataset = gsm_data_process(raw_dataset)
        # Typically GSM8K test set is the last 1319 samples in some splits, 
        # but here we might be using the full file as provided. 
        # Let's assume the user wants to test on the provided file.
        # If the file contains both train and test, we might need to split.
        # However, run_gsm8k.py splits manually: train[:80], test[80:].
        # Let's check the size first.
        print(f"Total samples in file: {len(dataset)}")
        
        # If test_samples is specified, use it. Otherwise use all.
        if args.test_samples > 0:
            dataset = dataset[80:80+args.test_samples] if args.test_samples > 0 else dataset[80:]
            print(f"Selected first {len(dataset)} samples for testing.")
        else:
            print(f"Using all {len(dataset)} samples for testing.")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Initialize Graph
    agent_names = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    graph = Graph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method="FinalRefer",
        use_transformer=True,
        max_routing=args.max_routing,
        available_roles=agent_names
    )
    
    # 3. Load Weights
    if os.path.exists(args.model_path):
        graph.load_weights(args.model_path)
        print(f"Successfully loaded weights from {args.model_path}")
    else:
        print(f"Error: Model path {args.model_path} does not exist.")
        return
        
    graph.set_eval()
    
    # 4. Evaluation Loop
    total_solved = 0
    total_executed = 0
    
    results = []
    
    # Cosine scaling for inference (as seen in run_gsm8k.py)
    graph.cos_scaling = 1e3 
    
    print("Starting Evaluation...")
    pbar = tqdm(dataset)
    
    for i, record in enumerate(pbar):
        task = record["task"]
        true_answer = record["answer"]
        input_dict = {"task": task}
        
        try:
            # Run Inference
            inference_result = graph.run_next_agent_prediction(
                input_dict,
                max_routing=args.max_routing,
                temperature=0.0, # Greedy for evaluation
                available_roles=agent_names
            )
            
            if inference_result is None:
                continue
                
            # Extract Answer
            predict_answer_list = inference_result.get("answers", [""])
            predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
            predict_answer_val = gsm_get_predict(predict_answer_str)
            routing_length = inference_result.get("routing_count", args.max_routing)
            
            # Check Correctness
            # GSM8K evaluation usually involves numeric comparison
            try:
                is_solved = float(predict_answer_val) == float(true_answer)
            except:
                is_solved = False
                
            if is_solved:
                total_solved += 1
            total_executed += 1
            
            # Record result
            results.append({
                "task": task,
                "true_answer": true_answer,
                "predicted_answer": predict_answer_val,
                "raw_prediction": predict_answer_str,
                "is_solved": is_solved,
                "routing_length": routing_length,
                "trace": inference_result.get("routing_results", {}).get("agent_selections", [])
            })
            
            # Update Progress Bar
            current_acc = total_solved / total_executed
            pbar.set_postfix({"Acc": f"{current_acc:.4f}", "Solved": f"{total_solved}/{total_executed}"})
            
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
            
    # 5. Final Report
    final_accuracy = total_solved / total_executed if total_executed > 0 else 0
    print("\n" + "="*50)
    print(f"Final Evaluation Results")
    print(f"Model: {args.model_path}")
    print(f"Total Samples: {total_executed}")
    print(f"Solved: {total_solved}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print("="*50)
    
    # Save detailed results
    output_file = Path(args.model_path).parent / f"eval_results_{Path(args.model_path).stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
