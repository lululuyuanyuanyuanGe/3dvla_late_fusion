import torch
import numpy as np
import os
import shutil
import pandas as pd
from transformers import AutoConfig
from model.configuration_spatialvla_dev import SpatialVLAConfig
from model.modeling_spatialvla_dev import SpatialVLAForConditionalGeneration
from train.data_flow import FlowMatchingParquetDataset, flow_data_collator
from train.utils_flow import freeze_backbone

def test_pipeline():
    print(">>> 1. Creating Dummy Parquet Data...")
    os.makedirs("temp_test_data", exist_ok=True)
    
    # Create 20 frames of dummy data
    data = {
        "image": [
            {"bytes": b"fake_image_bytes" * 10} for _ in range(20)
        ], # Mocking image bytes
        "instruction": [f"fake instruction {i}" for i in range(20)],
        "action": np.random.randn(20, 14).tolist(), # 14-dim actions
        "episode_index": [0]*10 + [1]*10 # 2 episodes, 10 frames each
    }
    df = pd.DataFrame(data)
    parquet_path = "temp_test_data/train.parquet"
    df.to_parquet(parquet_path)
    
    # Create Dummy Norm Stats
    norm_stats = {"action": {"mean": [0.0]*14, "std": [1.0]*14}}
    
    print(">>> 2. initializing Mock Processor...")
    class MockProcessor:
        def __call__(self, text, images, **kwargs):
            # Return dummy tensors matching what VLA processor does
            return {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "pixel_values": torch.randn(1, 3, 224, 224)
            }
        @property
        def tokenizer(self):
            return None # Not needed for this test
            
    processor = MockProcessor()
    
    print(">>> 3. Testing Dataset Loading...")
    dataset = FlowMatchingParquetDataset(
        dataset_path=parquet_path,
        processor=processor,
        norm_stats=norm_stats["action"],
        action_horizon=8,
        image_column="image",
        instruction_column="instruction",
        action_column="action",
        episode_column="episode_index"
    )
    print(f"Dataset length: {len(dataset)}")
    
    # Fetch one item
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    print(f"Action Chunk Shape: {item['actions'].shape} (Expected: [8, 14])")
    assert item['actions'].shape == (8, 14)
    
    # Test Boundary Condition (End of Ep 0)
    # Ep 0 ends at index 9. If we fetch index 8, we have frames 8, 9 (valid) and need 6 pads.
    print(">>> 3b. Testing Episode Boundary Padding...")
    boundary_item = dataset[8]
    actions = boundary_item['actions']
    # Check if last 6 actions are identical to the action at index 1 (relative to chunk start)
    # actions[0] = frame 8, actions[1] = frame 9
    # actions[2..7] should be padding (copy of frame 9)
    # Since we use float equality, we check closeness
    diff = (actions[2:] - actions[1]).abs().sum()
    print(f"Padding Difference (should be 0): {diff}")
    assert diff < 1e-5, "Padding logic failed! Boundary actions are not repeated."
    print("Boundary padding verified!")
    
    print(">>> 4. Testing Model Initialization & Freezing...")
    # Minimal Config for Testing
    config = SpatialVLAConfig(
        vision_config={"model_type": "siglip_vision_model", "hidden_size": 32, "image_size": 224, "patch_size": 14},
        text_config={"model_type": "llama", "hidden_size": 32, "vocab_size": 100, "num_hidden_layers": 2},
        action_expert_config={"hidden_size": 32, "model_type": "mlp", "out_dim": 14*8}, # simplified
        action_dim=14,
        action_horizon=8
    )
    # Mocking model creation (since we don't want to load 7B params here)
    # For this test, we just check if our util function runs without error on a torch module
    model = torch.nn.Module()
    model.vision_tower = torch.nn.Linear(10, 10)
    model.language_model = torch.nn.Linear(10, 10)
    model.action_expert = torch.nn.Linear(10, 10) # Mock expert
    model.geometric_projector = torch.nn.Linear(10, 10)
    
    freeze_backbone(model)
    
    print(">>> 5. Checking Gradients...")
    assert model.vision_tower.weight.requires_grad == False
    assert model.language_model.weight.requires_grad == False
    assert model.action_expert.weight.requires_grad == True
    assert model.geometric_projector.weight.requires_grad == True
    print("Freezing logic verified!")

    # Cleanup
    shutil.rmtree("temp_test_data")
    print("\n>>> TEST PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_pipeline()
