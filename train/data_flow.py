import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class FlowMatchingParquetDataset(Dataset):
    """
    Dataset loader for LeRobot/HuggingFace style Parquet datasets.
    This class handles the complexity of:
    1. Loading large-scale robot data efficiently via Parquet/Arrow.
    2. Creating "Action Chunks" on the fly (Sliding Window).
    3. Normalizing continuous actions for Flow Matching training.
    """
    def __init__(
        self,
        dataset_path,
        processor,
        norm_stats,
        action_horizon=8,
        image_column="image",
        instruction_column="instruction",
        action_column="actions",
        episode_column="episode_index",
        split="train"
    ):
        """
        Args:
            dataset_path (str): Path to the parquet file or directory.
            processor (SpatialVLAProcessor): VLA processor for tokenizing text and processing images.
            norm_stats (dict): Normalization statistics (mean, std) for actions.
            action_horizon (int): The number of future actions to predict (Chunk Size). Default 8.
            image_column (str): Column name in parquet for images.
            instruction_column (str): Column name in parquet for text instructions.
            action_column (str): Column name in parquet for raw actions.
            episode_column (str): Column name in parquet for episode index (to handle boundaries).
            split (str): Dataset split to load (e.g., 'train').
        """
        self.processor = processor
        self.norm_stats = norm_stats
        self.action_horizon = action_horizon
        self.image_col = image_column
        self.text_col = instruction_column
        self.action_col = action_column
        self.episode_col = episode_column
        
        logger.info(f"Loading dataset from {dataset_path}...")
        # 'load_dataset' uses memory mapping (Apache Arrow), so it doesn't load 
        # the full dataset into RAM, allowing training on huge files.
        self.dataset = load_dataset("parquet", data_files=dataset_path, split=split)
        
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
        text = item[self.text_col]
        
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
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
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
            
    # Auto-regressive labels generation
    # Even if we care mainly about Action Loss, the Trainer expects 'labels' for logging.
    # We clone input_ids as labels (standard Causal LM practice).
    # The model will internally ignore these for Flow Matching if configured, or compute aux loss.
    if "labels" not in batch and "input_ids" in batch:
        batch["labels"] = batch["input_ids"].clone()
        
    return batch
