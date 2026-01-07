# Custom 3D VLA Training Guide

## 1. Project Summary & Status
We are building a **Custom 3D Vision-Language-Action (VLA)** model composed of three modules:
1.  **3D Perception (Frozen):** MapAnything (2D $\to$ 3D reconstruction).
2.  **Spatial VLM (Frozen):** LLaVA-3D (Spatial reasoning & embedding).
3.  **Action Expert (Trainable):** LLaVA-based model using **Flow Matching** for action generation.

**Current Progress:**
- [x] **Data Pipeline:** `LeRobot` dataset (Libero) downloaded and inspected.
- [x] **Normalization:** Statistics computed (`norm_stats.json`) and verified.
- [x] **Training Logic:** Analyzed `train_pytorch.py` and understood Flow Matching parameter updates.
- [x] **Architecture Design:** Defined strategy to freeze perception backbones and train only the Adapter MLPs and Action Expert.

---

## 2. Implementation Plan

To train this custom architecture, we need to implement specific components in the `openpi` structure.

### Step 1: Create the Model Wrapper (`src/openpi/models/modeling_spatialvla.py`)
We need a single `nn.Module` that wraps the three components to manage data flow and gradient calculation.

```python
class SpatialVLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Initialize Frozen Backbones
        self.map_anything = MapAnythingPretrained()
        self.llava_3d = Llava3DPretrained()
        
        # 2. Initialize Trainable Components
        self.action_expert = Pi0ActionExpert(config) # The Flow Matching Transformer
        
        # 3. Initialize Connectors (Trainable)
        # Projects 3D embeddings to Action Expert dimension
        self.connector_mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Freeze Perception immediately
        for param in self.map_anything.parameters():
            param.requires_grad = False
        for param in self.llava_3d.parameters():
            param.requires_grad = False

    def forward(self, images, text, robot_state, actions=None):
        # --- Perception Phase (No Grad) ---
        with torch.no_grad():
            voxel_repr = self.map_anything(images)
            spatial_features = self.llava_3d.encode(voxel_repr, text)
            
        # --- Alignment Phase (Trainable) ---
        context_tokens = self.connector_mlp(spatial_features)
        
        # --- Action Generation Phase (Trainable) ---
        if actions is not None:
            # Training Mode: Compute Flow Matching Loss
            return self.action_expert.compute_loss(context_tokens, robot_state, actions)
        else:
            # Inference Mode: Sample Actions
            return self.action_expert.sample_actions(context_tokens, robot_state)
```

### Step 2: Configure Parameter Groups
In the training script (or model config), we must ensure the optimizer *only* sees the trainable parameters.

```python
# In scripts/train_pytorch.py or optimizer setup
param_groups = [
    # High learning rate for the new MLP
    {"params": model.connector_mlp.parameters(), "lr": 1e-3},
    # Standard learning rate for the Action Expert
    {"params": model.action_expert.parameters(), "lr": 1e-4},
]
# Note: map_anything and llava_3d parameters are excluded automatically 
# if requires_grad=False, but explicit filtering is safer.
optimizer = torch.optim.AdamW(param_groups)
```

### Step 3: Data Loading Strategy
Since MapAnything might be heavy, we have two options for the `DataLoader`:

1.  **Online (Slow):** Load images $\to$ MapAnything $\to$ Training.
    *   *Pros:* Simple, data augmentation on images works natively.
    *   *Cons:* Training speed limited by MapAnything inference time.
2.  **Offline (Fast - Recommended):** Pre-process the dataset.
    *   Run MapAnything + LLaVA-3D on all frames *once*.
    *   Save the `spatial_features` (tensors) to disk (e.g., `.pt` or `.h5` files).
    *   Create a new Dataset class that loads these embeddings directly.

## 3. Revised Training Implementation for Flow Matching

We are implementing a strict Flow Matching pipeline using Parquet datasets.

### A. Core Architecture (`train/train_flow.py`)
- **Main Entry Point**: `train_flow.py` (replaces legacy `spatialvla_finetune.py` for this task).
- **Freezing Logic**:
    - **Backbone (Frozen)**: `vision_tower`, `language_model`, `map_anything`.
    - **Trainable**: `action_expert`, `connector_mlp` (projector).
- **Training Objective**: Flow Matching Loss (Vector Field Regresson).
- **Precision**: `bfloat16`.

### B. Data Pipeline (`train/data_flow.py`)
- **Format**: Parquet (LeRobot style).
- **Loader**: `FlowMatchingParquetDataset`.
    - **Chunking**: Implements sliding window `[T, T+Horizon]`.
    - **Normalization**: Applies Gaussian normalization using `norm_stats.json`.
    - **Constraint**: MUST NOT append discrete action tokens to text input.
    - **Outputs**:
        - `pixel_values`: `[B, C, H, W]`
        - `input_ids`: `[B, Seq_Len]` (Text only)
        - `actions`: `[B, Horizon, Action_Dim]` (Normalized Continuous)

### C. Utilities (`train/utils_flow.py`)
- **Normalization Helpers**: `load_norm_stats`, `normalize_action`, `unnormalize_action`.
- **Freeze Helpers**: `freeze_backbone` (Ensures only Expert+MLP grads are active).

## 4. How to Run Training

### 1. Pre-requisites
Ensure you have the normalization stats ready (which we just computed):
```bash
# Verify file exists
ls assets/pi0_libero/physical-intelligence/libero/norm_stats.json
```

### 2. Launch Command
Use the documented PyTorch script (single GPU for development, Multi-GPU for full training).

```bash
# Development / Debugging
uv run scripts/train_pytorch.py pi0_libero --exp-name dev_3d_vla_v1

# Full Training (Multi-GPU)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi0_libero --exp-name train_3d_vla_v1
```

## 5. Next Actions
1.  **Create the File:** `train/utils_flow.py`.
2.  **Create the File:** `train/data_flow.py`.
3.  **Create the File:** `train/train_flow.py`.
4.  **Debug Run:** Verify the freezing logic and data shapes with a small test run.

```