import os
import sys
import logging
import torch
import json
import transformers
from transformers import (
    Trainer, 
    TrainingArguments, 
    HfArgumentParser, 
    set_seed,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoConfig
)
from dataclasses import dataclass, field
from typing import Optional

# Import custom model/processor
from model.modeling_spatialvla_dev import SpatialVLAForConditionalGeneration
from model.modeling_llava3d_v2 import LLaVA3DForCausalLMV2
from model.configuration_spatialvla_dev import SpatialVLAConfig
from model.processing_spatialvla_dev import SpatialVLAProcessor
from train.data_flow import FlowMatchingParquetDataset, flow_data_collator
from train.utils_flow import freeze_backbone, load_norm_stats

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained SpatialVLA/LLaVA model config."})
    action_horizon: int = field(default=8, metadata={"help": "Number of future action steps to predict."})
    action_dim: int = field(default=14, metadata={"help": "Dimension of action space."})

@dataclass
class DataArguments:
    train_data_path: str = field(metadata={"help": "Path to the training parquet file (or directory)."})
    norm_stats_path: str = field(metadata={"help": "Path to norm_stats.json."})
    tasks_json_path: Optional[str] = field(default=None, metadata={"help": "Path to tasks.jsonl mapping task_index to instruction."})
    image_column: str = field(default="image", metadata={"help": "Column name for images."})
    text_column: str = field(default="instruction", metadata={"help": "Column name for text instructions."})
    action_column: str = field(default="actions", metadata={"help": "Column name for actions."})
    episode_column: str = field(default="episode_index", metadata={"help": "Column name for episode index."})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    
    # 1. Load Config Manually to allow Path Patching BEFORE init
    logger.info(f"Loading config from {model_args.model_name_or_path}...")
    config_path = os.path.join(model_args.model_name_or_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Override paths if MODEL_ZOO_BASE env var is set (Fixes path mismatch)
    zoo_base = os.environ.get("MODEL_ZOO_BASE")
    if zoo_base:
        logger.info(f"Overriding model paths using MODEL_ZOO_BASE: {zoo_base}")
        config_dict["language_model_name_or_path"] = os.path.join(zoo_base, "llava3d_7B")
        config_dict["vision_model_name_or_path"] = os.path.join(zoo_base, "siglip-so400m-patch14-224")
        config_dict["map_anything_model_name_or_path"] = os.path.join(zoo_base, "mapanything")
        
    # print(f"[Debug Train] config_dict keys: {config_dict.keys()}")
    # if "text_config" in config_dict:
    #     print(f"[Debug Train] text_config type: {type(config_dict['text_config'])}")

    config = SpatialVLAConfig.from_dict(config_dict)
    config.action_horizon = model_args.action_horizon
    config.action_dim = model_args.action_dim

    # 2. Manually Construct Processor
    logger.info("Constructing Processor from sub-model paths...")
    # Load processor_config.json
    proc_cfg_path = os.path.join(model_args.model_name_or_path, "processor_config.json")
    with open(proc_cfg_path, "r") as f:
        proc_args = json.load(f)
    
    # Load sub-components
    logger.info(f"Loading Tokenizer from {config.language_model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.language_model_name_or_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(config.language_model_name_or_path, use_fast=False)
        
    logger.info(f"Loading ImageProcessor from {config.vision_model_name_or_path}...")
    image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
    
    # Inject standard LLaVA-3D logic
    seq_len = (image_processor.size["height"] // getattr(image_processor, "patch_size", 14)) ** 2 if hasattr(image_processor, "size") else 256
    setattr(image_processor, "image_seq_length", int(seq_len))
    
    # Initialize Processor with loaded config args
    processor = SpatialVLAProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        **proc_args # Pass statistics, bins, etc.
    )
    
    # 3. Load Model Components Manually (to ensure weights are loaded)
    logger.info(f"Loading Vision Tower from {config.vision_model_name_or_path}...")
    # Load Real Vision Config first to ensure correct architecture (patch size)
    vision_config = AutoConfig.from_pretrained(config.vision_model_name_or_path)
    if hasattr(vision_config, "vision_config"):
        vision_config = vision_config.vision_config
    # Update main config
    config.vision_config = vision_config
    
    vision_tower = AutoModel.from_pretrained(config.vision_model_name_or_path, config=vision_config)
    
    logger.info(f"Loading Language Model from {config.language_model_name_or_path}...")
    language_model = LLaVA3DForCausalLMV2.from_pretrained(config.language_model_name_or_path)
    
    logger.info("Initializing SpatialVLA...")
    model = SpatialVLAForConditionalGeneration(
        config,
        vision_model=vision_tower,
        language_model=language_model
    )
    
    # Handle casting manually since we didn't use from_pretrained on the wrapper
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model.to(dtype=dtype)
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # 4. Freeze Backbone
    freeze_backbone(model)
    
    # 5. Load Dataset
    norm_stats = load_norm_stats(data_args.norm_stats_path)
    train_dataset = FlowMatchingParquetDataset(
        dataset_path=data_args.train_data_path,
        processor=processor,
        norm_stats=norm_stats,
        tasks_json_path=data_args.tasks_json_path,
        action_horizon=model_args.action_horizon,
        image_column=data_args.image_column,
        instruction_column=data_args.text_column,
        action_column=data_args.action_column,
        episode_column=data_args.episode_column,
        split="train",
        target_action_dim=model_args.action_dim
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=flow_data_collator,
        tokenizer=processor.tokenizer,
    )

    # 7. Train
    if training_args.do_train:
        logger.info("Starting training...")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # Save Processor and Stats
        processor.save_pretrained(training_args.output_dir)
        import shutil
        shutil.copy(data_args.norm_stats_path, os.path.join(training_args.output_dir, "norm_stats.json"))

if __name__ == "__main__":
    main()
