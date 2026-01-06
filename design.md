# SpatialVLA: Action Expert Integration Design

## 1. Architectural Strategy: Late Fusion vs. Deep Fusion

We are integrating a continuous control **Action Expert** (based on Pi0's Flow Matching) into the **SpatialVLA** pipeline (powered by **LLaVA-3D**).

### The Challenge: Incompatibility of Pi0's "Deep Fusion"
The original Pi0 architecture uses a **"Deep Fusion" (Lockstep)** approach where VLM (PaliGemma) and Expert (Gemma) run layer-by-layer in parallel.
- **Blocker:** This requires identical depth/structure. Our VLM (Llama-based, ~32 layers) and Expert (Gemma-based, ~12-18 layers) are incompatible.

### The Solution: "Late Fusion" (Decoupled Action Head)
We will implement a **Decoupled Flow Matching Action Expert**.
- **Approach:** VLM processes perception fully $\to$ outputs `Hidden_States` $\to$ Expert consumes states to generate actions.
- **Benefit:** Efficient, modular, allows freezing VLM, uses pre-trained LLaVA-3D without "Open Heart Surgery".

---

## 2. System Components

### A. The VLM Backbone (LLaVA-3D)
- **Role:** Perception & Reasoning.
- **Output:** High-dimensional `Hidden_States` (Context) encoding 3D geometry and semantics.

### B. The Action Expert (New Module)
- **Role:** Continuous Action Generation via Flow Matching.
- **Backbone:** **GemmaModel** (pre-trained, e.g., `google/gemma-2b`) instantiated via Hugging Face `transformers`.
- **Input:**
    1.  **Context:** `Hidden_States` from LLaVA-3D (Projected to Gemma dimension).
    2.  **Noisy Actions:** Action trajectory at time $t$.
    3.  **Time Step ($t$):** Sinusoidal embedding.
- **Output:** Predicted Velocity ($v_t$) for denoising.

---

## 3. Communication Mechanism (Conditioning)

**Prefix Conditioning (Pseudo Cross-Attention)**
We use the standard Gemma Decoder but manipulate the input sequence.

1.  **Sequence Construction:** `[Context_Tokens | Time_Token | Action_Tokens]`
2.  **Projection:** LLaVA Context is projected to match Gemma's hidden size.
3.  **Attention:**
    - `Context` attends to itself (Bidirectional).
    - `Time` attends to `Context`.
    - `Action` attends to `Context` + `Time`.
    - *Note:* Since Flow Matching generates the whole trajectory at once (not auto-regressive per joint), `Action` tokens can likely attend to each other fully (Bidirectional) or Causally depending on preference. Pi0 often treats the action chunk as a block.

---

## 4. Implementation Plan

### Step 1: Define `FlowMatchingActionExpert`
Create `model/modeling_flow_expert.py`:
- **Backbone:** `GemmaModel`.
- **Components:**
    - `context_projector`: Linear(VLM_Dim $\to$ Expert_Dim).
    - `time_embedding`: Sinusoidal + MLP.
    - `action_in_proj` / `action_out_proj`.
- **Logic:** `forward()` computes flow matching MSE loss; `sample_actions()` runs Euler integration.

### Step 2: Integrate into `SpatialVLAForConditionalGeneration`
Update `model/modeling_spatialvla.py`:
- **Init:** Initialize `FlowMatchingActionExpert`.
- **Forward:** Pass LLaVA outputs to Expert.
- **Inference:** Implement `predict_action` to call Expert's sampling loop.

### Step 3: Clean Up
- Remove incompatible `PaliGemmaWithExpertModel` and `Pi0Pytorch` files.



1. llava作为action expert的基模型
2. late fusion, 用llava3d的hidden state和gemma做cross attention