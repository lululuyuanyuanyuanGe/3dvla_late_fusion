# Plan 1: Late Fusion (Gemma Expert) - Implementation Summary

We have successfully implemented the "Late Fusion" architecture where a lightweight Gemma-based Action Expert is conditioned on the output of the LLaVA-3D VLM.

## 1. Created Artifacts

### A. New Module: `model/modeling_flow_expert.py`
This file contains the standalone `FlowMatchingActionExpert` class.
- **Backbone:** Uses `transformers.GemmaModel` as the core Transformer decoder.
- **Key Components:**
    - `context_projector`: Linearly maps the VLM's hidden state dimension (e.g., 4096 from Llama) to the Expert's hidden dimension (e.g., 2048 for Gemma).
    - `time_embedding`: Implements the Sinusoidal Positional Embedding + MLP, identical to the Pi0/Diffusion design.
    - `action_in_proj` / `action_out_proj`: Handles the projection of raw action vectors (dim 14) to/from the model dimension.
- **Key Logic:**
    - `forward(context, actions, time)`: Constructs the input sequence `[Context | Time | Actions]`, creates the appropriate attention masks (bidirectional for expert, attending to context), and returns the predicted velocity $v_t$.
    - `compute_loss(context, actions)`: Implements the **Flow Matching Training Loop**.
        1. Samples time $t \sim U[0,1]$.
        2. Samples noise $x_1$.
        3. Interpolates to get noisy action $x_t$.
        4. Predicts velocity and returns MSE Loss against target $(noise - action)$.
    - `sample_actions(context, steps)`: Implements the **Euler Integration Inference Loop**.
        1. Starts from pure noise ($t=1$).
        2. Iteratively predicts velocity and steps backward to $t=0$ to recover the clean action.

### B. Modified Model: `model/modeling_spatialvla.py`
We integrated the expert into the main `SpatialVLAForConditionalGeneration` class.
- **Initialization:**
    - Checks `config.action_expert_config`.
    - Initializes `self.action_expert` if config is present.
- **Forward Pass (`forward`)**:
    - Added a new branch: `if actions is not None and self.action_expert is not None`.
    - In this branch, it runs the VLM backbone *only* to extract the `last_hidden_state`.
    - Passes this state to `self.action_expert.compute_loss()`.
    - Returns the flow matching loss directly.
- **Inference (`predict_action`)**:
    - Checks for `self.action_expert`.
    - If present, runs the VLM to get context, then calls `self.action_expert.sample_actions()` to generate the trajectory.
    - Preserves backward compatibility for legacy auto-regressive text generation if no expert is present.

### C. Modified Config: `model/configuration_spatialvla.py`
- Added `action_expert_config` to the `__init__` arguments.
- Added logic to automatically instantiate it as a `GemmaConfig` (via `CONFIG_MAPPING`) if provided as a dictionary.

## 2. Integration Status
- **Code:** Complete and logically sound.
- **Testing:** Preliminary unit tests were drafted (but not yet committed/run) to verify tensor shapes and connection flow.
- **Next Steps for Plan 1:**
    - Run the unit tests to confirm the dimension projections work (Llama 4096 -> Gemma 2048).
    - Verify the loss computation runs without error.

## 3. How to Use
To train this model, you simply provide the `actions` argument to the forward pass:
```python
outputs = model(
    input_ids=...,
    pixel_values=...,
    actions=action_tensor  # (Batch, Horizon, Dim)
)
loss = outputs.loss # This is now the Flow Matching MSE loss
```
