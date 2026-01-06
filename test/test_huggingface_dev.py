import argparse
from pathlib import Path
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoImageProcessor, AutoTokenizer

# Force protobuf to use pure-Python implementation to avoid sentencepiece pb2 C++ issues
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

def main():
    parser = argparse.ArgumentParser("Huggingface AutoModel Testing (dev)")
    parser.add_argument("--language_model_path", required=True, help="Language model path (LLaVA-3D)")
    parser.add_argument("--vision_model_path", required=True, help="Vision model path (SigLIP or LLaVA vision)")
    parser.add_argument("--map_anything_model_path", required=True, help="MapAnything model path")
    parser.add_argument("--spatialvla_model_path", default=None, help="Integrated SpatialVLA checkpoint path (optional)")
    parser.add_argument("--image_path", default=None, help="Test image path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images for testing")
    args = parser.parse_args()

    # Build processor and model
    if args.spatialvla_model_path and os.path.isdir(args.spatialvla_model_path):
        processor = AutoProcessor.from_pretrained(args.spatialvla_model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.spatialvla_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    else:
        # Compose processor from components to reflect patch-world semantics
        # Robust tokenizer loading with fallbacks to avoid sentencepiece/protobuf issues
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, use_fast=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.language_model_path, use_fast=False)
            except Exception:
                import traceback
                traceback.print_exc()
                raise ValueError("Cannot load tokenizer from language model path.")
                # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        image_processor = AutoImageProcessor.from_pretrained(args.vision_model_path)
        # Inject image_seq_length for patch-world expansion
        seq_len = (image_processor.size["height"] // getattr(image_processor, "patch_size", 14)) ** 2 if hasattr(image_processor, "size") else 256
        setattr(image_processor, "image_seq_length", int(seq_len))
        from model.processing_spatialvla_dev import SpatialVLAProcessor
        from model.configuration_spatialvla_dev import SpatialVLAConfig
        from model.modeling_spatialvla_dev import SpatialVLAForConditionalGeneration

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

        config = SpatialVLAConfig(
            language_model_name_or_path=args.language_model_path,
            vision_model_name_or_path=args.vision_model_path,
            map_anything_model_name_or_path=args.map_anything_model_path,
            image_token_index=256000,
            ignore_index=-100,
            # Add action expert config for testing Flow Matching
            action_expert_config={
                "model_type": "gemma",
                "hidden_size": 256, # Use very small model for fast dev test
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "intermediate_size": 512,
                "vocab_size": 256000
            },
            action_dim=14,
            action_horizon=8
        )
        model = SpatialVLAForConditionalGeneration(config).eval()
    # Move model to device
    dev = str(args.device)
    if dev.isdigit():
        dev = f"cuda:{dev}"
    model = model.to(dev)

    # Prepare image(s)
    if args.image_path and os.path.exists(args.image_path):
        image = Image.open(args.image_path).convert("RGB")
    else:
        # Create a dummy image if test/example.png doesn't exist
        image = Image.new('RGB', (224, 224), color = 'red')
    images = [image] * args.num_images

    prompt = "What action should the robot take to pick the cup?"
    # use a known key present in statistics for this simple test
    inputs = processor(images=images, text=prompt, unnorm_key="default", return_tensors="pt")
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in inputs.items()})

    # Predict actions (dev model exposes predict_action)
    if hasattr(model, "predict_action"):
        generation_outputs = model.predict_action(inputs)
        print("generation_outputs shape:", generation_outputs.shape)
        
        # Check if output is continuous (Flow Matching) or discrete tokens (AR)
        if torch.is_floating_point(generation_outputs):
             # Flow Matching: output is already continuous actions, just un-normalize
             # Move to CPU numpy for processor
             actions_raw = generation_outputs.detach().cpu().numpy()
             actions = processor.unnormalize_actions(actions_raw, unnorm_key="default")
             print("Decoded Continuous Actions:", actions)
        else:
             # AR: output is token IDs
             actions = processor.decode_actions(generation_outputs, unnorm_key="default")
             print("Decoded Discrete Actions:", actions)

    else:
        # Fallback to generate
        out = model.generate(**inputs, max_new_tokens=16)
        print("generate outputs:", out)

    print("DONE!")

if __name__ == "__main__":
    main()