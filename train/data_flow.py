import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

import json

class FlowMatchingParquetDataset(Dataset):
    """
    Dataset loader for LeRobot/HuggingFace style Parquet datasets.
    Supports on-the-fly action chunking and normalization.
    """
    def __init__(
        self,
        dataset_path,
        processor,
        norm_stats,
        tasks_json_path=None,
        action_horizon=8,
        image_column="image",
        instruction_column="instruction", # Can be task_index if using lookup
        action_column="actions",
        episode_column="episode_index",
        split="train",
        target_action_dim=19
    ):
        """
        Args:
            target_action_dim (int): Target dimension to pad actions to (default 19).
        """
        self.processor = processor
        self.norm_stats = norm_stats
        self.action_horizon = action_horizon
        self.image_col = image_column
        self.text_col = instruction_column
        self.action_col = action_column
        self.episode_col = episode_column
        self.target_action_dim = target_action_dim
        
        # Load Task Map if provided
        self.task_map = None
        if tasks_json_path:
            logger.info(f"Loading task map from {tasks_json_path}...")
            self.task_map = {}
            with open(tasks_json_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    self.task_map[entry['task_index']] = entry['task']
        
        logger.info(f"Loading dataset from {dataset_path}...")
        # 'load_dataset' uses memory mapping (Apache Arrow), so it doesn't load 
        # the full dataset into RAM, allowing training on huge files.
        self.dataset = load_dataset("parquet", data_files=dataset_path, split=split)
        
        if self.text_col not in self.dataset.column_names:
             # If exact column missing, check if we can fall back to task_index
             if "task_index" in self.dataset.column_names and self.task_map:
                 logger.info(f"Column '{self.text_col}' not found, but 'task_index' + tasks.jsonl present. Switching mode.")
                 self.text_col = "task_index"
             else:
                 logger.warning(f"Text column '{self.text_col}' not found. Will default to placeholder.")

        self.length = len(self.dataset)

    def __len__(self):
        # We perform a sliding window over the dataset to create chunks.
        # We allow the window to go up to the very last frame, handling padding in __getitem__.
        return self.length

    def __getitem__(self, idx):
        # 1. Get the main frame (Current Time T)
        # This is the "Observation" the robot sees right now.
        item = self.dataset[idx]
        
        # 2. Load Image
        # Parquet often stores images as raw bytes to save space. We decode them here.
        image = item[self.image_col]
        if isinstance(image, dict) and "bytes" in image:
            import io
            image = Image.open(io.BytesIO(image["bytes"]))
        
        # 3. Load Text
        # The instruction (e.g., "Pick up the apple")
        if self.task_map and self.text_col == "task_index":
            # Lookup instruction from task_index
            t_idx = item[self.text_col]
            text = self.task_map.get(t_idx, "unknown task")
        elif self.text_col in item:
            text = str(item[self.text_col])
        else:
            text = "do the task" 
        
        # 4. Load Action Chunk (T to T+Horizon) with Episode Boundary Handling
        # We calculate the safe end index. We cannot go beyond dataset length.
        end_idx = min(idx + self.action_horizon, self.length)
        
        # Fetch rows efficiently (returns a dict of lists)
        chunk_batch = self.dataset[idx : end_idx]
        
        # Extract actions and episode indices
        raw_actions_list = chunk_batch[self.action_col]
        episode_indices = chunk_batch[self.episode_col]
        
        current_episode = episode_indices[0]
        valid_length = 0
        
        # Find how many actions belong to the CURRENT episode
        for ep_idx in episode_indices:
            if ep_idx == current_episode:
                valid_length += 1
            else:
                break
                
        # Truncate to valid length (remove next episode's actions)
        valid_actions = np.array(raw_actions_list[:valid_length])
        
        # If we hit end of episode (or dataset end) and chunk is too short, PAD it.
        # We repeat the LAST valid action.
        if valid_length < self.action_horizon:
            pad_needed = self.action_horizon - valid_length
            last_action = valid_actions[-1]
            # Create padding (repeat last row)
            padding = np.tile(last_action, (pad_needed, 1))
            # Concatenate
            raw_actions = np.concatenate([valid_actions, padding], axis=0)
        else:
            raw_actions = valid_actions
            
        # 5. Normalize Actions
        # Flow Matching requires inputs to be Standard Gaussian (mean 0, std 1).
        from train.utils_flow import normalize_action
        normalized_actions = normalize_action(raw_actions, self.norm_stats)
        
        # Pad Action Dimension if necessary (e.g. 7 -> 19)
        current_dim = normalized_actions.shape[-1]
        if current_dim < self.target_action_dim:
             padding_dim = self.target_action_dim - current_dim
             # Pad with zeros along the last dimension
             zeros = np.zeros((normalized_actions.shape[0], padding_dim), dtype=normalized_actions.dtype)
             normalized_actions = np.concatenate([normalized_actions, zeros], axis=-1)
        
        # 6. Process Inputs using the VLA Processor
        # CRITICAL: We pass `text` and `images` to create the VLM inputs (input_ids, pixel_values).
        # We DO NOT pass `suffix_actions` because we don't want discrete action tokens appended.
        # We want the text input to be clean: "<image> Instruction <EOS>"
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length", # Ensures consistent batch shapes
            truncation=True
        )
        
        # The processor adds a batch dimension [1, seq_len] by default. 
        # We remove it because the DataLoader will add its own batch dimension later.
        inputs = {k: v.squeeze(0) if hasattr(v, "squeeze") else v for k, v in inputs.items()}
        
        # 7. Add Continuous Actions for Loss Calculation
        # We attach the normalized continuous actions to the input dictionary.
        # The model's forward pass will detect key "actions" and trigger the Flow Matching loss.
        inputs["actions"] = torch.tensor(normalized_actions, dtype=torch.float32)
        
        return inputs

def flow_data_collator(features):
    """
    Custom collator to stack action chunks and handle variable length sequences.
    Batches individual samples from __getitem__ into a single batch tensor.
    """
    first = features[0]
    batch = {}
    
    # Iterate through keys (input_ids, pixel_values, actions, etc.)
    for k in first.keys():
        if k in ["pixel_values", "actions", "intrinsic"]:
            # Simple stacking for fixed-size tensors (Images, Action Chunks)
            batch[k] = torch.stack([f[k] for f in features])
        elif k in ["input_ids", "attention_mask", "labels"]:
            # Stack text tensors. Since we used padding="max_length" in processor,
            # these are all same size.
            batch[k] = torch.stack([f[k] for f in features])
        elif k in ["image_token_id", "image_token_index", "num_image_tokens"]:
            # Metadata: typically scalar integers. Pass as tensor or single value?
            # Model expects scalar or 1D tensor. Let's pass as Tensor [B] or just the value if constant.
            # Safe bet: Stack them if they are in all items, to handle potential batching requirements
            # But model handles `int` or `Tensor`. 
            # Let's take the first one as they should be constant for the processor.
            batch[k] = first[k]
            
    # Auto-regressive labels generation
    # Even if we care mainly about Action Loss, the Trainer expects 'labels' for logging.
    # We clone input_ids as labels (standard Causal LM practice).
    # The model will internally ignore these for Flow Matching if configured, or compute aux loss.
    if "labels" not in batch and "input_ids" in batch:
        batch["labels"] = batch["input_ids"].clone()
        
    return batch
