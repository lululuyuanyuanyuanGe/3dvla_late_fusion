# Project Goal: SpatialVLA with 3D Capabilities and Flow Matching Action Expert

## Overview
Our research goal is to build a Vision-Language-Action (VLA) model with intrinsic 3D understanding capabilities.
The architecture consists of three main components:

1.  **3D Reconstruction**: Powered by **MapAnything**. This provides the model with geometric understanding of the environment. (Status: Integrated)
2.  **VLM Backbone**: **LLaVA-3D**. This is a Vision-Language Model trained on 3D data, serving as the core reasoning engine. (Status: Integrated)
3.  **Action Expert**: **Pi0 (v0.5)**. This component uses a **Flow Matching** architecture to generate continuous robot actions. (Status: **Pending Integration**)

## Current Task
We have integrated the Pi0 Action Expert into the SpatialVLA pipeline using the "Late Fusion" architecture.
- **Active Development File:** `model/modeling_spatialvla_dev.py`.
- **Constraint:** Do **not** modify or create `model/modeling_spatialvla.py`. All development must happen in the `_dev.py` files. Renaming to production filenames will only occur upon explicit instruction.
- **Status:** The integration is implemented in `modeling_spatialvla_dev.py` and `modeling_flow_expert.py`. `configuration_spatialvla_dev.py` has been patched to support `action_expert_config`.

## Technical Objectives
- **Verify:** Ensure `model/modeling_spatialvla_dev.py` correctly instantiates and uses `FlowMatchingActionExpert`.
- **Refine:** Update `SpatialVLAProcessor` (in `model/processing_spatialvla_dev.py`) to support continuous action tokens for Flow Matching training.
- **Test:** Validate the integration using `test/test_llava3d_integration.py` or new dedicated tests.
