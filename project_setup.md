Here is the complete summary documentation of the installation, troubleshooting, and code modifications required to run the **SpatialVLA / LLaVA-3D** environment on your cluster.

---

# SpatialVLA & MapAnything Setup Documentation

**Project Root:** `/public/home/luyuange/Models/SpatialVLA_llava3d`
**Environment:** `mapA_llava3d_pi05`
**Constraint:** Compute nodes have **no internet access**. All downloads must happen on Login Nodes.

---

## 1. Environment & Dependency Fixes
We encountered several version conflicts between modern libraries and the older codebase.

**Commands run:**
```bash
# Fix 1: Downgrade HuggingFace Hub (Transformers required < 1.0)
pip install "huggingface-hub<1.0.0"

# Fix 2: Downgrade Protobuf (SentencePiece conflict)
pip install protobuf==3.20.3

# Fix 3: Downgrade NumPy (TensorFlow/Transformers conflict with NumPy 2.x)
pip install "numpy<2"

# Fix 4: Install missing MapAnything dependencies
pip install omegaconf timm einops
```

---

## 2. Model Zoo Setup (Offline Mode)
Since the compute node cannot download models, we manually downloaded weights and source code on the Login Node.

### A. Hugging Face Weights
Location: `/public/home/luyuange/Models/SpatialVLA_llava3d/model_zoo/`

We downloaded these using Python scripts with `hf-mirror.com`:
1.  **LLaVA-3D-7B:** `ChaimZhu/LLaVA-3D-7B`
2.  **SigLIP:** `google/siglip-so400m-patch14-224`
3.  **MapAnything:** `facebook/map-anything` (or `crowd-play/MapAnything`)
4.  **CLIP:** `openai/clip-vit-large-patch14-336` (Required by LLaVA config)

### B. Source Code Repositories
We cloned these into the project root to satisfy `ModuleNotFoundError`:
1.  **MapAnything:** `git clone https://github.com/crowd-play/MapAnything.git`
2.  **UniCeption:** `git clone https://github.com/castacks/UniCeption.git`

### C. DINOv2 Setup (Torch Hub Cache)
`torch.hub` tries to query GitHub, which caused **403 Forbidden** errors. We bypassed this by manually caching the repo.

**Action:**
1.  Created `~/.cache/torch/hub/facebookresearch_dinov2_main`.
2.  Cloned DINOv2 into that folder.
3.  **Code Patch:** Modified `UniCeption/uniception/models/encoders/dinov2.py` to use `source='local'` and point to the cache path.

---

## 3. Code Modifications
We had to patch several files to make the code compatible with local paths and fix logic errors.

### A. LLaVA Builder (`builder.py`)
**Issue:** The code hardcoded checks for model names starting with "openai". It rejected our local path `/public/home/...`.
**Fix:** Modified `build_vision_tower` logic.
**File:** `LLaVA_3D/llava/model/multimodal_encoder/builder.py`
```python
# Before
if vision_tower.startswith("openai") ...
# After
if "clip" in vision_tower or vision_tower.startswith("openai") ...
```

### B. LLaVA Config (`config.json`)
**Issue:** The config pointed to the online CLIP model.
**Fix:** Pointed `mm_vision_tower` to the local folder.
**File:** `model_zoo/llava3d_7B/config.json`
```json
"mm_vision_tower": "/public/home/luyuange/Models/SpatialVLA_llava3d/model_zoo/clip-vit-large-patch14-336"
```

### C. SpatialVLA Model (`modeling_spatialvla_dev.py`)
**File:** `model/modeling_spatialvla_dev.py`

1.  **Fix Dimension Mismatch (Line 173):**
    *   **Issue:** Confusion between DINOv2-Large (1024) and MapAnything output (768).
    *   **Fix:** Hardcoded the linear layer to accept 768.
    ```python
    self.geometric_projector = nn.Linear(768, vision_hidden_size)
    ```

2.  **Fix Error Check (Line 334):**
    *   **Issue:** The code raised `NotImplementedError` comparing incorrect variables.
    *   **Fix:** Commented out the check.
    ```python
    # raise NotImplementedError(...)
    ```

3.  **Fix Undefined Variable (Line 340-341):**
    *   **Issue:** `geom_broadcast` was used but defined in commented-out lines.
    *   **Fix:** Uncommented the lines.
    ```python
    geom_global = projected_geometric_features.mean(dim=1, keepdim=True)
    geom_broadcast = geom_global.expand(feats.shape[0], feats.shape[1], geom_global.shape[-1])
    ```

---

## 4. Final Execution Command

To run the test, use the following command. Note the `PYTHONPATH` includes the current directory (`.`) and the manually cloned libraries.

```bash
cd /public/home/luyuange/Models/SpatialVLA_llava3d

PYTHONPATH=.:./MapAnything:./UniCeption python test/test_huggingface_dev.py \
  --language_model_path /public/home/luyuange/Models/SpatialVLA_llava3d/model_zoo/llava3d_7B \
  --vision_model_path /public/home/luyuange/Models/SpatialVLA_llava3d/model_zoo/siglip-so400m-patch14-224 \
  --map_anything_model_path /public/home/luyuange/Models/SpatialVLA_llava3d/model_zoo/mapanything \
  --image_path /public/home/luyuange/Models/SpatialVLA_llava3d/test/example.png \
  --device 3
```