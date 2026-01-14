import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import torch.nn.functional as F
import copy

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.transformer.transformer import RoutingTransformer
from GDesigner.transformer.utils import gumbel_softmax
from copy import deepcopy

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                use_transformer:bool = False,
                max_routing:int = 10,
                available_roles:List[str] = None,
                ):
        
        self.id:str = shortuuid.ShortUUID().random(length=max_routing)
        self.domain:str = domain
        self.llm_name:str = 'Qwen/Qwen2.5-7B-Instruct'
        self.agent_names:List[str] = agent_names
        self.max_routing:int = max_routing
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.role_embeddings = None        
        self.optimizer = None
        self.role_to_idx = None
        self.idx_to_role = None
        
        self.decision_method = decision_method

        self.prompt_set = PromptSetRegistry.get(domain)
        
        if use_transformer:
            transformer_hidden_dim = 768
            # Transformer components for agent routing
            self.transformer_hidden_dim = transformer_hidden_dim
            # Projection layer from concatenated features (384*4) to transformer hidden dim (192)
            self.proj_to_transformer_dim_history = torch.nn.Linear(384*2, transformer_hidden_dim)
            self.proj_to_transformer_dim_task = torch.nn.Linear(384, transformer_hidden_dim)
            self.proj_to_transformer_dim_role = torch.nn.Linear(384, transformer_hidden_dim)
            # Projection layers for dual-head scoring and NHP
            # self.proj_to_transformer_dim_nap = torch.nn.Linear(384, transformer_hidden_dim) # Removed NAP
            
            # Dual-Head Scoring Layers
            self.linear_imp = torch.nn.Linear(transformer_hidden_dim, transformer_hidden_dim)
            self.linear_gap = torch.nn.Linear(transformer_hidden_dim, transformer_hidden_dim)
            
            # NCS (Neural Context Selector) Components
            # Input dim = Task (D) + State (D) + History (D) = 3 * D
            self.ncs_mlp = torch.nn.Sequential(
                torch.nn.Linear(3 * transformer_hidden_dim, transformer_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(transformer_hidden_dim, 1) # Output single score (logit)
            )

            # Routing transformer 
            self.routing_transformer = RoutingTransformer(hidden_dim=transformer_hidden_dim, max_seq_len=max_routing + 1) # +1 for the decision node


        # Initialize role embeddings
        self.role_embeddings = {}

        self.cos_scaling = 1
        
        # Add regular agents
        embedding_dim = 384  # Standard embedding dimension
        for role in available_roles:
            # Get role description and create embedding
            role_desc = self.prompt_set.get_description(role)
            role_embedding = torch.tensor(get_sentence_embedding(role_desc))  # (384,)
            # Store as tensor (not Parameter) to make it fixed/not trainable
            self.role_embeddings[role] = role_embedding

        # Add decision node
        decision_role = self.decision_method
        decision_desc = self.prompt_set.get_decision_role()
        decision_embedding = torch.tensor(get_sentence_embedding(decision_desc))
        # Store as tensor (not Parameter) to make it fixed/not trainable
        self.role_embeddings[decision_role] = decision_embedding

        self.role_to_idx = deepcopy({role: idx for idx, role in enumerate(available_roles)})
        self.role_to_idx[decision_role] = len(available_roles)  # DecisionMaker gets the next index
        self.idx_to_role = deepcopy({idx: role for role, idx in self.role_to_idx.items()})

        self.available_roles = available_roles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def run_next_agent_prediction(self, input: Dict[str, str], max_routing=10, temperature=1.0,
                               available_roles:List[str] = None, training=False, agent_group_type="MathSolver", max_context = 5, ncs_threshold=0.5,
                               forced_agent_sequence:List[str]=None, forced_context_masks:List[List[int]]=None):
        """
        Synchronous version of arun_next_agent_prediction that runs the async function in an event loop.
        """
        loop = asyncio.new_event_loop()
        timeout = 1800
        if training:
            timeout = 1800
        else:
            timeout = 1800

        try:
            # Run the async function in the loop with a 1-minute timeout
            return loop.run_until_complete(
                asyncio.wait_for(
                    self.arun_next_agent_prediction(
                        input=input,
                        max_routing=max_routing,
                        temperature=temperature,
                        available_roles=available_roles,
                        agent_group_type=agent_group_type,
                        max_context=max_context,
                        ncs_threshold=ncs_threshold,
                        forced_agent_sequence=forced_agent_sequence,
                        forced_context_masks=forced_context_masks
                    ),
                    timeout=timeout  
                )
            )
        except asyncio.TimeoutError:
            print("Timeout occurred")
            return None
        finally:
            loop.close()

    async def arun_next_agent_prediction(self, input: Dict[str, str], max_routing=10, temperature=1.0,
                                         available_roles:List[str] = None, agent_group_type="MathSolver", max_context = 5, ncs_threshold=0.5,
                                         forced_agent_sequence:List[str]=None, forced_context_masks:List[List[int]]=None):
        """
        Predicts the next agent to execute using a routing transformer, and finish a trace of the routing process.
        """
        # Extract task query
        self.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        query = input['task']

        available_roles = self.available_roles
        decision_role = self.decision_method

        # Get task embedding (384 dimensions)
        raw_task_embedding = torch.tensor(get_sentence_embedding(query)).detach()
        raw_task_embedding = raw_task_embedding.to(self.device)
        task_embedding = self.proj_to_transformer_dim_task(raw_task_embedding)

        role_embeddings = self.role_embeddings
        # Create role indices mapping
        role_to_idx = self.role_to_idx  
        idx_to_role = self.idx_to_role
        
        # Initialize projected role embeddings (start empty)
        proj_roles_embeddings = []
        
        # Project role embeddings to transformer dimension
        for role in available_roles:
            role_embedding = role_embeddings[role].to(self.device)
            # Project role embedding to transformer dimension
            proj_role_embedding = self.proj_to_transformer_dim_role(role_embedding)
            proj_roles_embeddings.append(proj_role_embedding)
            
        decision_embedding = self.role_embeddings[self.decision_method].to(self.device)
        # Project decision embedding to transformer dimension
        proj_decision_embedding = self.proj_to_transformer_dim_role(decision_embedding)
        # Add decision role embedding
        proj_roles_embeddings.append(proj_decision_embedding)
        
        # Convert to tensor
        proj_roles_embeddings = torch.stack(proj_roles_embeddings) if proj_roles_embeddings else torch.empty((0, self.transformer_hidden_dim)) # Handle case with no roles
        
        # NAP_embedding = self.proj_to_transformer_dim_nap(raw_task_embedding) # Removed NAP
        
        # Routing process
        routing_results = {
            "agent_selections": [],
            "hint_selections": [],
            "agent_logits": [],
            "hint_logits": [],
            "ncs_logits": [],
            "agent_outputs_embeddings": [],
            "history_trace": []
        }
        routing_count = 0
        final_answers = []
        
        # History states for tracking outputs and embeddings
        history_states = []
        
        # Use inference mode to disable gradient computation
        while routing_count <= max_routing:
            # Create input for transformer
            if len(history_states) == 0:
                # No history yet, just use roles and special tokens
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),
                    proj_roles_embeddings,
                    # NAP_embedding.unsqueeze(0),
                ], dim=0)
            else:
                # Include history embeddings
                history_embeddings = torch.stack([state["embedding"] for state in history_states])
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),
                    proj_roles_embeddings,
                    history_embeddings,
                    # NAP_embedding.unsqueeze(0),
                ], dim=0)
            
            # Get encoded tokens from transformer
            num_roles = len(proj_roles_embeddings) # Number of role tokens
            encoded_tokens = self.routing_transformer(all_embeddings, num_prefix_tokens=num_roles + 1)
            
            # Feature Extraction for BiRouter
            encoded_task = encoded_tokens[0]
            encoded_roles = encoded_tokens[1 : 1 + num_roles]
            
            if len(history_states) == 0:
                encoded_state = encoded_task # Use task vector as state vector for cold start
            else:
                # History starts at index 1 + num_roles
                # Last history token is at index 1 + num_roles + len(history_states) - 1
                last_history_idx = num_roles + len(history_states)
                encoded_state = encoded_tokens[last_history_idx]

            # Dual-Head Scoring
            # ImpScore: Static matching (Task <-> Roles)
            query_imp = self.linear_imp(encoded_task)
            scores_imp = torch.matmul(encoded_roles, query_imp)

            # GapScore: Dynamic transition (State <-> Roles)
            query_gap = self.linear_gap(encoded_state)
            scores_gap = torch.matmul(encoded_roles, query_gap)
            
            # Standardization
            mean_imp, std_imp = scores_imp.mean(), scores_imp.std()
            scores_imp = (scores_imp - mean_imp) / (std_imp + 1e-8)
            
            mean_gap, std_gap = scores_gap.mean(), scores_gap.std()
            scores_gap = (scores_gap - mean_gap) / (std_gap + 1e-8)
            
            # Dynamic Fusion (alpha = 0.5)
            alpha = 0.5
            agent_similarity_scores = alpha * scores_imp + (1 - alpha) * scores_gap
            
            # Agent prediction
            # nap_idx = len(all_embeddings) - 2  # Second to last token
            # encoded_nap = encoded_tokens[nap_idx]
            
            # Force DecisionMaker in the last step or if forced_agent_sequence ends
            # Handle forced_agent_sequence
            forced_role = None
            if forced_agent_sequence is not None:
                if routing_count < len(forced_agent_sequence):
                    forced_role = forced_agent_sequence[routing_count]
                else:
                    # Sequence finished, force decision maker
                    forced_role = decision_role

            if forced_role is not None:
                chosen_agent_idx = role_to_idx.get(forced_role, role_to_idx[decision_role])
                chosen_role = forced_role
                # Mock scores for consistency
                agent_similarity_scores = torch.zeros(len(available_roles)).to(self.device)
                if chosen_role != decision_role:
                    # Find index in available_roles (0 to num_roles-1)
                    # chosen_agent_idx matches index in available_roles
                    pass
            elif routing_count == max_routing:
                chosen_agent_idx = role_to_idx[decision_role]
                chosen_role = decision_role
                # use inner product + standardization (mean=0, std=1)
                # agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
                # mean = agent_similarity_scores.mean()
                # std = agent_similarity_scores.std()
                # agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)
                agent_similarity_scores = self.cos_scaling  * agent_similarity_scores

            else:
                # Calculate similarity scores for agent selection (indices 0 to num_roles-1)
                # agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
                # mean = agent_similarity_scores.mean()
                # std = agent_similarity_scores.std()
                # agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)   
                
                # Apply Gumbel softmax for sampling with -3 to 3 input to balance the exploration and exploitation
                agent_similarity_scores = self.cos_scaling  * agent_similarity_scores

                # use softmax if self.cos_scaling > 100
                if self.cos_scaling > 100:
                    agent_probabilities = F.softmax(agent_similarity_scores, dim=0)
                else:
                    agent_probabilities = gumbel_softmax(agent_similarity_scores, tau=temperature, hard=True)
                
                # Sample next agent
                chosen_agent_idx = torch.argmax(agent_probabilities).item()
                chosen_role = idx_to_role[chosen_agent_idx] # Direct mapping using 0-based index
            
            # Hint prediction (only if we have history)
            hint_mask = []
            # NHP Logic Removed - Using NCS instead below
            
            # --- NCS (Neural Context Selector) Logic ---
            # Use NCS probabilities to refine hint_mask
            if len(history_states) > 0:
                history_start_idx = 1 + num_roles
                history_end_idx = history_start_idx + len(history_states)
                encoded_histories = encoded_tokens[history_start_idx : history_end_idx] # [N, D]
                
                # 2. Expand Task and State vectors to match history count
                num_history = len(history_states)
                task_expanded = encoded_task.unsqueeze(0).expand(num_history, -1) # [N, D]
                state_expanded = encoded_state.unsqueeze(0).expand(num_history, -1) # [N, D]
                
                # 3. Concatenate: [Task, State, History_i] -> [N, 3*D]
                ncs_input = torch.cat([task_expanded, state_expanded, encoded_histories], dim=1)
                
                # 4. MLP Scoring
                ncs_logits = self.ncs_mlp(ncs_input).squeeze(-1) # [N, 1] -> [N]
                ncs_probs = torch.sigmoid(ncs_logits) # [N]
                
                # Store NCS logits
                routing_results["ncs_logits"].append(ncs_logits.tolist())

                # Apply NCS selection (Sample or Threshold)
                if self.cos_scaling > 100:
                    ncs_mask = (ncs_threshold < ncs_probs).int()
                else:
                    # Use threshold for exploration as well, as per Dexp requirements (eta control)
                    ncs_mask = (ncs_threshold < ncs_probs).int()
                
                # Combine original hint_mask (from NHP) and NCS mask? 
                # Or just replace hint_mask with NCS mask?
                # Based on user request "Change original history selection to what I described", 
                # we should use NCS output to determine hints.
                
                # Let's replace hint_mask with ncs_mask for context selection
                hint_mask = ncs_mask
                
                # Override with forced_context_masks if provided
                if forced_context_masks is not None:
                    # forced_context_masks[k] corresponds to routing_count = k + 1 (since step 0 has no history)
                    # Current routing_count is 1 or more (since len(history_states) > 0)
                    # history_states has length 1 at routing_count 1.
                    # So we want forced_context_masks[routing_count - 1] ?
                    # Let's assume forced_context_masks aligns with steps that HAVE history.
                    # Or simpler: forced_context_masks is a list of masks for each step that executes an agent.
                    # But step 0 has no mask.
                    # Let's assume forced_context_masks index i corresponds to step i+1.
                    force_idx = routing_count - 1
                    if 0 <= force_idx < len(forced_context_masks):
                         # Ensure mask length matches history length
                         forced_mask = forced_context_masks[force_idx]
                         # If history length > forced mask length, pad with 0? Or truncate?
                         # Usually context grows. 
                         # Let's assume the passed mask is correct for the current history size.
                         # If forced_mask is just list of 0/1, convert to tensor.
                         if len(forced_mask) == len(hint_mask):
                             hint_mask = torch.tensor(forced_mask, dtype=torch.int32).to(self.device)
                         else:
                             # Handle mismatch (e.g. if we pruned steps, history indices might be different?)
                             # But here we are running the graph.
                             # If we are replaying a pruned trace, history_states should match what was recorded?
                             # Let's just try to match length.
                             # If forced_mask is longer, truncate. If shorter, pad 0.
                             current_len = len(hint_mask)
                             target_len = len(forced_mask)
                             if target_len >= current_len:
                                 hint_mask = torch.tensor(forced_mask[:current_len], dtype=torch.int32).to(self.device)
                             else:
                                 # Pad with 0
                                 padded = forced_mask + [0] * (current_len - target_len)
                                 hint_mask = torch.tensor(padded, dtype=torch.int32).to(self.device)

                # Recency Fallback: Force retain the last history item if all are filtered out
                if hint_mask.sum() == 0 and len(hint_mask) > 0:
                    hint_mask[-1] = 1
                
                # Update stored selection
                # Note: hint_selections in routing_results currently stores NHP mask. 
                # We should update it to store the actual mask used (NCS mask).
                routing_results["hint_selections"].append(hint_mask.tolist())
                # routing_results["hint_selections"][-1] = hint_mask.tolist()

                # set max context length by maskout before last max_context steps
                if len(hint_mask) > max_context:
                    hint_mask[:-max_context] = 0
            # -------------------------------------------

            # Construct hints based on the mask
            hints = ""
            if len(hint_mask) > 0 and sum(hint_mask) > 0:
                selected_hints = [history_states[i]["output"] for i in range(len(hint_mask)) if hint_mask[i] == 1]
                if selected_hints:
                    for i, hint in enumerate(selected_hints):
                        hints += f"\nAgent {i+1}: {hint} \n"
            
            # Execute the chosen agent with hints
            # Create a node for the chosen role if not decision maker
            output_embedding = None
            if chosen_role != self.decision_method:
                # Create input with hints
                agent_input = input.copy()
                if hints:
                    agent_input["hints"] = hints
                else:
                    agent_input["hints"] = ""
                
                # Create and execute the agent node
                agent_node = AgentRegistry.get(agent_group_type, **{"domain": self.domain, "llm_name": self.llm_name, "role": chosen_role})
                await agent_node.async_execute_with_hints(agent_input)
                agent_output = agent_node.outputs[-1] if agent_node.outputs else "No output produced"
                
                
                # Create embedding for this output
                output_embedding = torch.tensor(get_sentence_embedding(agent_output)).to(self.device).detach()
                
                # Create history state
                state_embedding = self.proj_to_transformer_dim_history(torch.cat([
                    role_embeddings[chosen_role],
                    output_embedding
                ], dim=0))
                
                # Add to history
                history_states.append({
                    "role": chosen_role,
                    "output": agent_output,
                    "embedding": state_embedding
                })
                
                # Store in routing results
                routing_results["agent_outputs_embeddings"].append(output_embedding.cpu())
                routing_results["agent_selections"].append(chosen_agent_idx)
                routing_results["agent_logits"].append(agent_similarity_scores.tolist())
                routing_results["history_trace"].append({"role": chosen_role, "content": agent_output})
            else:
                # If decision maker is chosen, use it as final answer and break
                decision_input = input.copy()
                if hints:
                    decision_input["hints"] = hints
                
                decision_node = AgentRegistry.get(decision_role, **{"domain": self.domain, "llm_name": self.llm_name})
                await decision_node.async_execute_with_hints(decision_input)
                final_answers = decision_node.outputs
                if not final_answers:
                    final_answers = ["No final answer produced by decision maker"]
                
                # Store in routing results
                if final_answers:
                    routing_results["agent_outputs_embeddings"].append(torch.tensor(get_sentence_embedding(final_answers[-1])).detach())
                    
                # Store agent selection results
                routing_results["agent_selections"].append(chosen_agent_idx)
                routing_results["agent_logits"].append(agent_similarity_scores.tolist())
                    
                break
            
            # Update routing count
            routing_count += 1
        
        return {
            "answers": final_answers,
            "routing_results": routing_results,
            "routing_count": routing_count
        }

    def train_step_sft(self, input_dict, target_agent_ids, target_masks, history_trace, 
                       alpha=0.5, lambda_sparse=0.01, w_ncs=1.0):
        """
        Execute one SFT training step for a single trajectory using Teacher Forcing.
        
        Args:
            input_dict: Dict with 'task'
            target_agent_ids: List[int] of gold agent indices (length T)
            target_masks: List[List[int]] of gold NCS masks. 
                          target_masks[k] is mask for step k+1 (using history 0..k).
                          Length should be T-1 (or T if we continue).
            history_trace: List[Dict] of gold history outputs [{"role": str, "content": str}, ...]
                           Length should be T-1 (inputs for steps 1..T-1).
            alpha: BiRouter fusion weight
            lambda_sparse: Sparsity weight
            w_ncs: NCS loss weight
            
        Returns:
            loss: scalar tensor
            metrics: dict of metric values
        """
        self.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        query = input_dict['task']
        
        # Prepare Task Embedding
        raw_task_embedding = torch.tensor(get_sentence_embedding(query)).detach().to(self.device)
        task_embedding = self.proj_to_transformer_dim_task(raw_task_embedding)
        
        # Prepare Role Embeddings (Fixed)
        proj_roles_embeddings = []
        for role in self.available_roles:
            role_emb = self.role_embeddings[role].to(self.device)
            proj_roles_embeddings.append(self.proj_to_transformer_dim_role(role_emb))
        
        decision_role = self.decision_method
        decision_emb = self.role_embeddings[decision_role].to(self.device)
        proj_roles_embeddings.append(self.proj_to_transformer_dim_role(decision_emb))
        proj_roles_embeddings = torch.stack(proj_roles_embeddings)
        num_roles = len(proj_roles_embeddings)
        
        # Prepare History States (Teacher Forcing)
        # We process step by step to accumulate loss
        history_states = []
        
        total_loss = 0.0
        loss_nap_accum = 0.0
        loss_ncs_accum = 0.0
        loss_sparse_accum = 0.0
        
        steps = len(target_agent_ids)
        
        for step in range(steps):
            # 1. Construct Input
            if len(history_states) == 0:
                all_embeddings = torch.cat([task_embedding.unsqueeze(0), proj_roles_embeddings], dim=0)
            else:
                hist_embs = torch.stack([s["embedding"] for s in history_states])
                all_embeddings = torch.cat([task_embedding.unsqueeze(0), proj_roles_embeddings, hist_embs], dim=0)
                
            # 2. Transformer Forward
            encoded_tokens = self.routing_transformer(all_embeddings, num_prefix_tokens=num_roles + 1)
            
            # 3. Extract Features
            encoded_task = encoded_tokens[0]
            encoded_roles = encoded_tokens[1 : 1 + num_roles]
            
            if len(history_states) == 0:
                encoded_state = encoded_task
            else:
                last_hist_idx = num_roles + len(history_states)
                encoded_state = encoded_tokens[last_hist_idx]
                
            # 4. BiRouter Scoring (NAP)
            # ImpScore
            query_imp = self.linear_imp(encoded_task)
            scores_imp = torch.matmul(encoded_roles, query_imp)
            
            # GapScore
            query_gap = self.linear_gap(encoded_state)
            scores_gap = torch.matmul(encoded_roles, query_gap)
            
            # Standardize
            scores_imp = (scores_imp - scores_imp.mean()) / (scores_imp.std() + 1e-8)
            scores_gap = (scores_gap - scores_gap.mean()) / (scores_gap.std() + 1e-8)
            
            # Fuse
            logits = alpha * scores_imp + (1 - alpha) * scores_gap
            logits = self.cos_scaling * logits # Scaling
            
            # NAP Loss (Cross Entropy)
            target_idx = torch.tensor(target_agent_ids[step]).to(self.device)
            loss_nap = F.cross_entropy(logits.unsqueeze(0), target_idx.unsqueeze(0))
            loss_nap_accum += loss_nap
            
            # 5. NCS Scoring & Loss
            loss_ncs = torch.tensor(0.0).to(self.device)
            loss_sparse = torch.tensor(0.0).to(self.device)
            
            if len(history_states) > 0:
                hist_start = 1 + num_roles
                hist_end = hist_start + len(history_states)
                encoded_hists = encoded_tokens[hist_start:hist_end]
                
                num_h = len(history_states)
                task_exp = encoded_task.unsqueeze(0).expand(num_h, -1)
                state_exp = encoded_state.unsqueeze(0).expand(num_h, -1)
                
                ncs_input = torch.cat([task_exp, state_exp, encoded_hists], dim=1)
                ncs_logits = self.ncs_mlp(ncs_input).squeeze(-1) # [N]
                
                # NCS Target Mask
                # target_masks indices correspond to steps with history.
                # step 0 has no history. step 1 has history 0.
                # target_masks[0] is for step 1.
                # So for current 'step', we need target_masks[step-1].
                mask_idx = step - 1
                if mask_idx < len(target_masks):
                    target_mask = torch.tensor(target_masks[mask_idx], dtype=torch.float).to(self.device)
                    
                    # Ensure dimensions match (in case of truncation/mismatch)
                    L = min(len(ncs_logits), len(target_mask))
                    if L > 0:
                        loss_ncs = F.binary_cross_entropy_with_logits(ncs_logits[:L], target_mask[:L])
                        
                        # Sparsity
                        ncs_probs = torch.sigmoid(ncs_logits[:L])
                        loss_sparse = torch.mean(torch.abs(ncs_probs))
            
            loss_ncs_accum += loss_ncs
            loss_sparse_accum += loss_sparse
            
            # Combine
            step_total = loss_nap + w_ncs * loss_ncs + lambda_sparse * loss_sparse
            total_loss += step_total
            
            # 6. Update History (Teacher Forcing)
            # If not the last step, we need history for next step
            if step < len(history_trace):
                # Embed the ground truth history
                gt_item = history_trace[step] # history for step+1 comes from output of step
                role_name = gt_item["role"]
                content = gt_item["content"]
                
                # Create embedding
                # Note: This operation is slow if done strictly sequentially on CPU.
                # Ideally pre-compute embeddings. But here we do on-fly.
                out_emb = torch.tensor(get_sentence_embedding(content)).detach().to(self.device)
                
                # Concat with role
                role_vec = self.role_embeddings[role_name].to(self.device)
                cat_emb = torch.cat([role_vec, out_emb], dim=0)
                
                # Project
                state_emb = self.proj_to_transformer_dim_history(cat_emb)
                
                history_states.append({
                    "role": role_name,
                    "embedding": state_emb,
                    "concat_embedding": cat_emb # Kept for consistency though not used here
                })
        
        # Average loss over steps? Or sum?
        # User says: "Sum or Mean over t=1..T"
        # Let's use Mean to be scale-invariant
        total_loss = total_loss / steps
        
        return total_loss, {
            "loss_nap": loss_nap_accum.item() / steps,
            "loss_ncs": loss_ncs_accum.item() / steps,
            "loss_sparse": loss_sparse_accum.item() / steps
        }

    def run_next_agent_prediction_grad(self, gradient_input, sparse_context=0.0):
        """
        Synchronous version of arun_next_agent_prediction_grad that runs the async function in an event loop.
        """
        # Create a new event loop
        loop = asyncio.new_event_loop()
        try:
            # Run the async function in the loop and return its result
            return loop.run_until_complete(self.arun_next_agent_prediction_grad(gradient_input, sparse_context))
        finally:
            # Clean up the loop
            loop.close()

    async def arun_next_agent_prediction_grad(self, gradient_input, sparse_context=0.0):
        """
        Calculate policy gradient for a single trace using the provided gradient input.
        Performs backpropagation and direct parameter updates at each step to prevent memory explosion.
        """
        # Extract task query and other inputs
        self.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        query = gradient_input['task']
        advantage = gradient_input['advantage']
        agent_selections = gradient_input['agent_selections']
        hint_selections = gradient_input['hint_selections']
        trace_length = gradient_input['trace_length']
        
        # Return immediately if advantage is zero (no gradient to backprop)
        if abs(advantage) < 1e-8:
            return 0.0
        
        # Prepare embeddings for the task
        task_embedding_raw = torch.tensor(get_sentence_embedding(query))
        task_embedding_raw = task_embedding_raw.detach().to(self.device)
        # Initialize history states for tracking outputs and embeddings
        history_states = []
        
        # Process each step in the trace
        for step_idx in range(trace_length - 1):  # -1 because the last step is decision maker
            task_embedding = self.proj_to_transformer_dim_task(task_embedding_raw)
            # import ipdb; ipdb.set_trace()
            proj_roles_embeddings = []
            for role in self.role_to_idx.keys():
                role_embedding = self.role_embeddings[role].to(self.device)
                # Project role embedding to transformer dimension
                proj_role_embedding = self.proj_to_transformer_dim_role(role_embedding)
                proj_roles_embeddings.append(proj_role_embedding)
            
            
            # Convert to tensor
            proj_roles_embeddings = torch.stack(proj_roles_embeddings)
            
            # Construct NAP and NHP embeddings
            assert torch.is_grad_enabled(), "Grad disabled unexpectedly!"

            # NAP_embedding = self.proj_to_transformer_dim_nap(task_embedding_raw) # Removed NAP
            
            # Create input for transformer
            if len(history_states) == 0:
                # No history yet, just use roles and special tokens
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),    
                    proj_roles_embeddings,
                    # NAP_embedding.unsqueeze(0),
                ], dim=0)
            else:
                # Include history embeddings
                concat_embeddings = torch.stack([state["concat_embedding"] for state in history_states])
                # project concat_embeddings to transformer dimension
                history_embeddings = self.proj_to_transformer_dim_history(concat_embeddings)
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),
                    proj_roles_embeddings,
                    history_embeddings,
                    # NAP_embedding.unsqueeze(0),
                ], dim=0)
            
            # Get encoded tokens from transformer
            num_roles = len(proj_roles_embeddings)
            encoded_tokens = self.routing_transformer(all_embeddings, num_prefix_tokens=num_roles + 1)
            
            # Agent prediction
            # nap_idx = len(all_embeddings) - 2  # Second to last token
            # encoded_nap = encoded_tokens[nap_idx]
            
            # Feature Extraction for BiRouter
            encoded_task = encoded_tokens[0]
            encoded_roles = encoded_tokens[1 : 1 + num_roles]
            
            if len(history_states) == 0:
                encoded_state = encoded_task # Use task vector as state vector for cold start
            else:
                # History starts at index 1 + num_roles
                # Last history token is at index 1 + num_roles + len(history_states) - 1
                last_history_idx = num_roles + len(history_states)
                encoded_state = encoded_tokens[last_history_idx]

            # Dual-Head Scoring
            # ImpScore: Static matching (Task <-> Roles)
            query_imp = self.linear_imp(encoded_task)
            scores_imp = torch.matmul(encoded_roles, query_imp)

            # GapScore: Dynamic transition (State <-> Roles)
            query_gap = self.linear_gap(encoded_state)
            scores_gap = torch.matmul(encoded_roles, query_gap)
            
            # Standardization
            mean_imp, std_imp = scores_imp.mean(), scores_imp.std()
            scores_imp = (scores_imp - mean_imp) / (std_imp + 1e-8)
            
            mean_gap, std_gap = scores_gap.mean(), scores_gap.std()
            scores_gap = (scores_gap - mean_gap) / (std_gap + 1e-8)
            
            # Dynamic Fusion (alpha = 0.5)
            alpha = 0.5
            agent_similarity_scores = alpha * scores_imp + (1 - alpha) * scores_gap

            # Calculate similarity scores for agent selection
            # use inner product + standardization (mean=0, std=1)
            # agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
            # mean = agent_similarity_scores.mean()
            # std = agent_similarity_scores.std()
            # agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)
            
            # --- NCS (Neural Context Selector) Logic ---
            # 1. Extract Candidate History Vectors
            # History starts at index 1 + num_roles
            # We need all history tokens: encoded_tokens[1+num_roles : 1+num_roles+len(history_states)]
            # if len(history_states) > 0:
            #     history_start_idx = 1 + num_roles
            #     history_end_idx = history_start_idx + len(history_states)
            #     encoded_histories = encoded_tokens[history_start_idx : history_end_idx] # [N, D]
                
            #     # 2. Expand Task and State vectors to match history count
            #     num_history = len(history_states)
            #     task_expanded = encoded_task.unsqueeze(0).expand(num_history, -1) # [N, D]
            #     state_expanded = encoded_state.unsqueeze(0).expand(num_history, -1) # [N, D]
                
            #     # 3. Concatenate: [Task, State, History_i] -> [N, 3*D]
            #     ncs_input = torch.cat([task_expanded, state_expanded, encoded_histories], dim=1)
                
            #     # 4. MLP Scoring
            #     ncs_logits = self.ncs_mlp(ncs_input).squeeze(-1) # [N, 1] -> [N]
            #     ncs_probs = torch.sigmoid(ncs_logits) # [N]
                
            #     # Store NCS logits
            #     routing_results["ncs_logits"].append(ncs_logits.tolist())

            #     # TODO: In future steps, use ncs_probs to filter/select context
            #     # For now, we just compute it as requested.
            # -------------------------------------------

            # Get logprobs for the agent selection
            agent_logprobs = F.log_softmax(self.cos_scaling * agent_similarity_scores, dim=0)
            
            # Get the selected agent from trace
            selected_agent_idx = agent_selections[step_idx]
            
            # Calculate agent selection loss (negative because we want to maximize reward)
            step_loss = -agent_logprobs[selected_agent_idx] * advantage

            hint_logprob = None # Initialize hint_logprob
            if len(history_states) > 0:
                # Hint prediction (if we have history)
                # NHP Removed
                pass
                
                # NHP-based loss calculation removed

            # Backward pass for this step only
            try:
                with torch.autograd.set_detect_anomaly(True):
                    step_loss.backward()
            except RuntimeError as e:
                print(f"ERROR during backward: {e}")
                import ipdb; ipdb.set_trace() # Re-enter debugger if backward fails

            # Detach everything for the next step to free memory
            # Simulate the output that would have been generated
            chosen_role = self.idx_to_role[selected_agent_idx]
            
            if chosen_role != self.decision_method:
                # For non-decision roles, we need to add a state to history
                # Get pre-computed output embedding directly from gradient input
                output_embedding = gradient_input['agent_outputs_embeddings'][step_idx].detach().to(self.device)
                
                # Add to history without detaching to preserve gradients
                history_states.append({
                    "role": chosen_role,
                    "concat_embedding": torch.cat([
                        self.role_embeddings[chosen_role],
                        output_embedding
                    ], dim=0),
                    
                })
        
        return True
    
    def set_train(self):
        self.routing_transformer.train()
        self.proj_to_transformer_dim_history.train()
        self.proj_to_transformer_dim_task.train()
        self.proj_to_transformer_dim_role.train()
        self.linear_imp.train()
        self.linear_gap.train()
        self.ncs_mlp.train()

    def set_eval(self):
        self.routing_transformer.eval()
        self.proj_to_transformer_dim_history.eval()
        self.proj_to_transformer_dim_task.eval()
        self.proj_to_transformer_dim_role.eval()
        self.linear_imp.eval()
        self.linear_gap.eval()
        self.ncs_mlp.eval()

    def to_device(self, device):
        self.routing_transformer = self.routing_transformer.to(device)
        self.proj_to_transformer_dim_history = self.proj_to_transformer_dim_history.to(device)
        self.proj_to_transformer_dim_task = self.proj_to_transformer_dim_task.to(device)
        self.proj_to_transformer_dim_role = self.proj_to_transformer_dim_role.to(device)
        self.linear_imp = self.linear_imp.to(device)
        self.linear_gap = self.linear_gap.to(device)
        self.ncs_mlp = self.ncs_mlp.to(device)
        for role in self.role_embeddings:
            self.role_embeddings[role] = self.role_embeddings[role].to(device)
        self.device = device

    def add_to_optimizer(self, optimizer):
        self.optimizer = optimizer
        # self.optimizer.add_param_group({"params": self.routing_transformer.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_history.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_task.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_role.parameters()})
        self.optimizer.add_param_group({"params": self.linear_imp.parameters()})
        self.optimizer.add_param_group({"params": self.linear_gap.parameters()})
        self.optimizer.add_param_group({"params": self.ncs_mlp.parameters()})
        # Role embeddings are not trainable, so we don't add them to the optimizer

    def save_model(self, path):
        """
        Saves the graph model and all its components to the specified path.
        """
        model_state = {
            'id': self.id,
            'domain': self.domain,
            'llm_name': self.llm_name,
            'agent_names': self.agent_names,
            'decision_method': self.decision_method,
            'max_routing': self.max_routing,
            'optimized_spatial': self.optimized_spatial,
            'optimized_temporal': self.optimized_temporal,
            'cos_scaling': self.cos_scaling,
            'available_roles': self.available_roles,
            'role_to_idx': self.role_to_idx,
            'idx_to_role': self.idx_to_role,
            'transformer_hidden_dim': self.transformer_hidden_dim
        }
        
        # Save transformer components if present
        if hasattr(self, 'routing_transformer'):
            model_state['routing_transformer'] = self.routing_transformer.state_dict()
            model_state['proj_to_transformer_dim_history'] = self.proj_to_transformer_dim_history.state_dict()
            model_state['proj_to_transformer_dim_task'] = self.proj_to_transformer_dim_task.state_dict()
            model_state['proj_to_transformer_dim_role'] = self.proj_to_transformer_dim_role.state_dict()
            model_state['linear_imp'] = self.linear_imp.state_dict()
            model_state['linear_gap'] = self.linear_gap.state_dict()
            model_state['ncs_mlp'] = self.ncs_mlp.state_dict()
        
        # Save role embeddings
        role_embeddings_dict = {}
        for role, embedding in self.role_embeddings.items():
            role_embeddings_dict[role] = embedding.detach().cpu()
        model_state['role_embeddings'] = role_embeddings_dict
        
        # Save the state dictionary
        torch.save(model_state, path)

    def clone_for_inference(self):
        """
        Creates a safe copy of the graph for inference without deepcopy issues with tensors.
        Returns a new Graph instance with copied and detached tensors.
        """
        # Create a new graph instance with the same parameters
        new_graph = type(self)(
            domain=self.domain,
            llm_name=self.llm_name,
            agent_names=self.agent_names.copy() if hasattr(self.agent_names, 'copy') else self.agent_names,
            decision_method=self.decision_method,
            optimized_spatial=self.optimized_spatial,
            optimized_temporal=self.optimized_temporal,
            use_transformer=hasattr(self, 'routing_transformer'),
            max_routing=self.max_routing,
            available_roles=self.available_roles.copy() if hasattr(self.available_roles, 'copy') else self.available_roles
        )
        
        # Copy and detach tensors manually
        if hasattr(self, 'routing_transformer'):
            new_graph.routing_transformer.load_state_dict(self.routing_transformer.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_history'):
            new_graph.proj_to_transformer_dim_history.load_state_dict(self.proj_to_transformer_dim_history.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_task'):
            new_graph.proj_to_transformer_dim_task.load_state_dict(self.proj_to_transformer_dim_task.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_role'):
            new_graph.proj_to_transformer_dim_role.load_state_dict(self.proj_to_transformer_dim_role.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_nap'):
            # new_graph.proj_to_transformer_dim_nap.load_state_dict(self.proj_to_transformer_dim_nap.state_dict())
            pass
            
        if hasattr(self, 'linear_imp'):
            new_graph.linear_imp.load_state_dict(self.linear_imp.state_dict())
            
        if hasattr(self, 'linear_gap'):
            new_graph.linear_gap.load_state_dict(self.linear_gap.state_dict())
            
        if hasattr(self, 'ncs_mlp'):
            new_graph.ncs_mlp.load_state_dict(self.ncs_mlp.state_dict())

        # Copy role embeddings
        if hasattr(self, 'role_embeddings'):
            new_graph.role_embeddings = {
                role: embedding.clone().detach() 
                for role, embedding in self.role_embeddings.items()
            }

        # clone the cos_scaling
        new_graph.cos_scaling = self.cos_scaling
        new_graph.routing_transformer.eval()
        return new_graph

    @staticmethod
    def load_model(path):
        """
        Loads a graph model from the specified path.
        """
        # Load the state dictionary
        model_state = torch.load(path, map_location=torch.device('cpu'))
        
        # Create a new instance of the Graph class
        graph = Graph(
            domain=model_state['domain'],
            llm_name=model_state['llm_name'],
            agent_names=model_state['agent_names'],
            decision_method=model_state['decision_method'],
            optimized_spatial=model_state['optimized_spatial'],
            optimized_temporal=model_state['optimized_temporal'],
            use_transformer='routing_transformer' in model_state,
            max_routing=model_state['max_routing'],
            available_roles=model_state['available_roles']
        )
        
        # Restore ID and other attributes
        graph.id = model_state['id']
        graph.cos_scaling = model_state['cos_scaling']
        graph.role_to_idx = model_state['role_to_idx']
        graph.idx_to_role = model_state['idx_to_role']
        
        # Load transformer components if present
        if 'routing_transformer' in model_state:
            graph.transformer_hidden_dim = model_state['transformer_hidden_dim']
            graph.routing_transformer.load_state_dict(model_state['routing_transformer'])
            graph.proj_to_transformer_dim_history.load_state_dict(model_state['proj_to_transformer_dim_history'])
            graph.proj_to_transformer_dim_task.load_state_dict(model_state['proj_to_transformer_dim_task'])
            graph.proj_to_transformer_dim_role.load_state_dict(model_state['proj_to_transformer_dim_role'])
            
            if 'linear_imp' in model_state:
                graph.linear_imp.load_state_dict(model_state['linear_imp'])
            if 'linear_gap' in model_state:
                graph.linear_gap.load_state_dict(model_state['linear_gap'])
            if 'ncs_mlp' in model_state:
                graph.ncs_mlp.load_state_dict(model_state['ncs_mlp'])
        
        # Load role embeddings
        role_embeddings_dict = model_state['role_embeddings']
        graph.role_embeddings = {}
        for role, embedding in role_embeddings_dict.items():
            graph.role_embeddings[role] = embedding
        
        return graph
