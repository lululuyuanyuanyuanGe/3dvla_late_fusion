# SpatialVLA Research Plans

We are exploring three distinct architectural strategies to integrate continuous control (Flow Matching) into the SpatialVLA (LLaVA-3D) pipeline.

## Plan 1: Late Fusion (Gemma Expert)
**"The Efficient Decoupled Head"**

*   **VLM Backbone:** LLaVA-3D (Llama architecture).
*   **Action Expert:** Gemma (2B or smaller).
*   **Integration Mechanism:** **Late Fusion (Prefix Conditioning)**.
    *   The VLM processes the perception fully and outputs `Hidden_States`.
    *   The Expert takes `[Context (VLM) | Time | Actions]` as input.
    *   The Expert runs its own independent Transformer stack.
*   **Pros:**
    *   **Architecture Agnostic:** The VLM and Expert can be any model (Llama + Gemma, Llama + Bert, etc.).
    *   **Efficiency:** Allows freezing the massive VLM during action training.
    *   **Simplicity:** No need to modify the internal attention mechanisms of the VLM.
*   **Cons:**
    *   **Bandwidth:** No bidirectional communication during the reasoning process. The Expert only sees the final result of the VLM.

## Plan 2: Late Fusion (LLaVA/Llama Expert)
**"The Homogeneous Decoupled Head"**

*   **VLM Backbone:** LLaVA-3D (Llama architecture).
*   **Action Expert:** LLaVA/Llama (Small variant or same architecture).
*   **Integration Mechanism:** **Late Fusion (Prefix Conditioning)**.
    *   Identical mechanism to Plan 1, but the Expert uses the **Llama architecture** instead of Gemma.
*   **Rationale:**
    *   If the VLM is Llama-based, using a Llama-based Expert might offer better transfer learning or feature compatibility (embedding spaces) if we decide to share weights or tokenizers later.
*   **Pros:**
    *   Same as Plan 1.
    *   Potential for weight sharing or easier tokenizer alignment.
*   **Cons:**
    *   Llama models (even small ones like 1B/3B) might be heavier than Gemma 2B.

## Plan 3: Deep Fusion (LLaVA + LLaVA)
**"The Pi0-Style Lockstep Architecture"**

*   **VLM Backbone:** LLaVA-3D (Llama architecture).
*   **Action Expert:** LLaVA/Llama (Must match VLM layer count & hidden dim).
*   **Integration Mechanism:** **Deep Fusion (Interleaved Attention)**.
    *   **Two Models, One Heartbeat:** Both models run layer-by-layer in parallel.
    *   **Shared Attention:** Inside every Attention Layer, the Keys (K) and Values (V) from the VLM and the Expert are **concatenated**.
    *   The Action tokens can attend to the VLM tokens *at that specific layer*.
*   **Requirements:**
    *   The Expert **MUST** be structurally identical to the VLM (same number of layers, same hidden dimension). This likely means instantiating a second copy of the LLaVA-3D model (or a Llama model of exact same spec) as the expert.
*   **Implementation:** Requires creating a custom `LlamaDeepFusion` class that modifies the `LlamaAttention` logic to accept and fuse two streams.
*   **Pros:**
    *   **Rich Communication:** The Expert can influence and query the VLM throughout the entire reasoning depth.
    *   **State-of-the-Art:** Replicates the exact architectural advantage of Pi0.
*   **Cons:**
    *   **Heavy:** Requires running TWO full-sized Llama models in parallel. (Memory usage $\approx$ 2x).
    *   **Complex:** Requires custom model code ("Open Heart Surgery" on Llama).
