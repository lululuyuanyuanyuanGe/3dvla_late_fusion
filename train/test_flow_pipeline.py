import torch
import numpy as np
import os
import shutil
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel
)
from model.configuration_spatialvla_dev import SpatialVLAConfig
from model.modeling_spatialvla_dev import SpatialVLAForConditionalGeneration
from model.modeling_llava3d_v2 import LLaVA3DForCausalLMV2
from model.processing_spatialvla_dev import SpatialVLAProcessor
from train.data_flow import FlowMatchingParquetDataset, flow_data_collator
from train.utils_flow import freeze_backbone, load_norm_stats

def create_dummy_image_bytes():
    img = Image.new('RGB', (32, 32), color = 'red')
    with BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()

# --- HARDCODE YOUR REAL PARQUET PATH HERE ---
REAL_PARQUET_PATH = "/2025233147/zzq/cache/huggingface/lerobot/physical-intelligence/libero/data/chunk-000/episode_000000.parquet"
REAL_STATS_PATH = "/2025233147/zzq/cache/huggingface/lerobot/physical-intelligence/libero/data/norm_stats.json"
MODEL_ZOO_BASE = "/2025233147/mapAnythingLlava3dPi0.5/model_parameters"
# --------------------------------------------

def test_pipeline():
    parquet_path = None
    
    if os.path.exists(REAL_PARQUET_PATH):
        print(f">>> 1. Found Real Parquet Data at: {REAL_PARQUET_PATH}")
        parquet_path = REAL_PARQUET_PATH
    else:
        print(f">>> 1. Real path '{REAL_PARQUET_PATH}' not found. Creating Valid Dummy Data...")
        os.makedirs("temp_test_data", exist_ok=True)
        
        dummy_image_bytes = create_dummy_image_bytes()
        
        # Create 20 frames of dummy data
        data = {
            "image": [
                {"bytes": dummy_image_bytes} for _ in range(20)
            ], # Valid image bytes
            "instruction": [f"fake instruction {i}" for i in range(20)],
            "action": np.random.randn(20, 14).tolist(), # 14-dim actions
            "episode_index": [0]*10 + [1]*10 # 2 episodes, 10 frames each
        }
        df = pd.DataFrame(data)
        parquet_path = "temp_test_data/train.parquet"
        df.to_parquet(parquet_path)
    
    # Load Norm Stats
    if os.path.exists(REAL_STATS_PATH):
        print(f">>> 1b. Loading Real Norm Stats from: {REAL_STATS_PATH}")
        norm_stats = load_norm_stats(REAL_STATS_PATH)
    else:
        print("Using Dummy Norm Stats...")
        norm_stats = {"mean": [0.0]*14, "std": [1.0]*14}
    
    print(">>> 2. initializing Processor...")
    # Initialize Real Processor components
    # We use the paths defined in MODEL_ZOO_BASE
    language_model_path = f"{MODEL_ZOO_BASE}/llava3d_7B"
    vision_model_path = f"{MODEL_ZOO_BASE}/siglip-so400m-patch14-224"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(language_model_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(language_model_path, use_fast=False)
        
    image_processor = AutoImageProcessor.from_pretrained(vision_model_path)
    # Inject image_seq_length for patch-world expansion (Standard LLaVA-3D logic)
    seq_len = (image_processor.size["height"] // getattr(image_processor, "patch_size", 14)) ** 2 if hasattr(image_processor, "size") else 256
    setattr(image_processor, "image_seq_length", int(seq_len))
    
    # Load Real Vision Config
    vision_config = AutoConfig.from_pretrained(vision_model_path)
    if hasattr(vision_config, "vision_config"):
        print("Extracting vision_config from SiglipConfig...")
        vision_config = vision_config.vision_config
    
    processor = SpatialVLAProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        statistics={"default": {"action": {"q01": [0,0,0], "q99": [1,1,1], "mask": [1,1,1]}}},
        bin_policy={
            "translation": {
                "theta_bins": [0.0, 0.785398, 1.570796, 2.356194, 3.141593],
                "phi_bins": [-3.141593, -1.570796, 0.0, 1.570796, 3.141593],
                "r_bins": [0.0, 0.433013, 0.866025, 1.299038, 1.732051],
            },
            "rotation": {
                "roll_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "pitch_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "yaw_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
        },
        intrinsic_config={"default": {"width": 224, "height": 224, "intrinsic": [[200,0,112],[0,200,112],[0,0,1]]}},
        action_config={
            "num_bins": {
                "translation": {"theta_bins": 4, "phi_bins": 4, "r_bins": 4},
                "rotation": {"roll_bins": 4, "pitch_bins": 4, "yaw_bins": 4},
                "gripper": 2,
            },
            "use_spherical": False,
        },
    )
    
    print(">>> 3. Testing Dataset Loading...")
    dataset = FlowMatchingParquetDataset(
        dataset_path=parquet_path,
        processor=processor,
        norm_stats=norm_stats,
        action_horizon=8,
        image_column="image",
        instruction_column="instruction", # Will fallback to placeholder if missing
        action_column="actions",
        episode_column="episode_index"
    )
    print(f"Dataset length: {len(dataset)}")
    
    # Fetch one item
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    
    # Dynamic dimension check
    action_shape = item['actions'].shape
    print(f"Action Chunk Shape: {action_shape} (Expected: [8, ?])")
    assert action_shape[0] == 8
    detected_dim = action_shape[1]
    
    print(">>> 4. Loading REAL Model (this may take time)...")
    
    # Manually construct Config (Pattern from test/test_huggingface_dev.py)
    # This avoids transformers library validation issues with local paths
    config = SpatialVLAConfig(
        language_model_name_or_path=language_model_path,
        vision_model_name_or_path=vision_model_path,
        map_anything_model_name_or_path=f"{MODEL_ZOO_BASE}/mapanything",
        vision_config=vision_config, # Use loaded config
        text_config={"model_type": "llama"}, # Minimal required
        action_expert_config={"model_type": "gemma", "hidden_size": 2048, "num_hidden_layers": 4}, # Minimal/Default
        action_dim=detected_dim,
        action_horizon=8
    )

    # Initialize Model from Config
    # To answer "did we load weights?": We must manually load components now.
    print(f"Loading Vision Tower from {vision_model_path}...")
    vision_tower = AutoModel.from_pretrained(vision_model_path, config=vision_config)
    
    print(f"Loading Language Model from {language_model_path}...")
    language_model = LLaVA3DForCausalLMV2.from_pretrained(language_model_path)
    
    print("Combining into SpatialVLA...")
    # Use float32 (default) to avoid Core Dump, use Checkpointing to avoid OOM.
    model = SpatialVLAForConditionalGeneration(
        config,
        vision_model=vision_tower,
        language_model=language_model
    )
    model.gradient_checkpointing_enable()
    model.train() # Ensure training mode
    
    print(">>> 5. Freezing Backbone...")
    freeze_backbone(model)
    
    print(">>> 6. Running Forward Pass (Flow Matching Loss)...")
    # Collate a small batch
    batch_list = [dataset[0], dataset[1]]
    batch = flow_data_collator(batch_list)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Safely move batch to device
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(**batch)
    
    print(f"Output Keys: {outputs.keys()}")
    if hasattr(outputs, "loss"):
        print(f"Loss: {outputs.loss.item()}")
        
    print(">>> 7. Running Inference (Sampling Actions)...")
    # Prepare inputs for inference (remove training-specific keys)
    inference_inputs = {k: v for k, v in batch.items() if k not in ["actions", "labels"]}
    
    with torch.no_grad():
        # predict_action uses the Flow Matching ODE solver (Euler)
        generated_actions = model.predict_action(inference_inputs)
        
    print(f"Generated Actions Shape: {generated_actions.shape}")
    print(f"Sample Action (Normalized - First 3 steps):\n{generated_actions[0, :3, :]}")
    
    # Unnormalize to check physical values
    actions_raw = generated_actions.detach().cpu().numpy()
    actions_unnorm = processor.unnormalize_actions(actions_raw, unnorm_key="default")
    print(f"Sample Action (Real World - First 3 steps):\n{actions_unnorm[0, :3, :]}")

    print("\n>>> REAL PIPELINE TEST PASSED! <<<")

if __name__ == "__main__":
    test_pipeline()
