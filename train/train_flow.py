import os
import sys
import logging
import torch
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from dataclasses import dataclass, field
from typing import Optional

# Import custom model/processor
from model.modeling_spatialvla_dev import SpatialVLAForConditionalGeneration
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
    model_name_or_path: str = field(metadata={"help": "Path to pretrained SpatialVLA/LLaVA model."})
    action_horizon: int = field(default=8, metadata={"help": "Number of future action steps to predict."})
    action_dim: int = field(default=14, metadata={"help": "Dimension of action space."})

@dataclass
class DataArguments:
    train_data_path: str = field(metadata={"help": "Path to the training parquet file (or directory)."})
    norm_stats_path: str = field(metadata={"help": "Path to norm_stats.json."})
    image_column: str = field(default="image", metadata={"help": "Column name for images."})
    text_column: str = field(default="instruction", metadata={"help": "Column name for text instructions."})
    action_column: str = field(default="action", metadata={"help": "Column name for actions."})
    episode_column: str = field(default="episode_index", metadata={"help": "Column name for episode index."})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    
    # 1. Load Processor
    logger.info(f"Loading processor from {model_args.model_name_or_path}...")
    processor = SpatialVLAProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    
    # 2. Load Config & Update
    logger.info(f"Loading config from {model_args.model_name_or_path}...")
    config = SpatialVLAConfig.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    config.action_horizon = model_args.action_horizon
    config.action_dim = model_args.action_dim
    
    # 3. Load Model
    logger.info("Loading SpatialVLA Model...")
    model = SpatialVLAForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        local_files_only=True
    )
    
    # 4. Freeze Backbone
    freeze_backbone(model)
    
    # 5. Load Dataset
    norm_stats = load_norm_stats(data_args.norm_stats_path)
    train_dataset = FlowMatchingParquetDataset(
        dataset_path=data_args.train_data_path,
        processor=processor,
        norm_stats=norm_stats,
        action_horizon=model_args.action_horizon,
        image_column=data_args.image_column,
        instruction_column=data_args.text_column,
        action_column=data_args.action_column,
        episode_column=data_args.episode_column
    )
    logger.info(f"Loaded training dataset with {len(train_dataset)} samples.")

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
