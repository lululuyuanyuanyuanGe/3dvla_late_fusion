import os
import torch
import numpy as np
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
import transformers
from transformers import logging, TrainerCallback, Trainer
from transformers.trainer import LengthGroupedSampler, RandomSampler, has_length, is_datasets_available, seed_worker, _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import logger
from typing import List, Optional
from torch.utils.data import Dataset, Sampler

logger = logging.get_logger(__name__)

from LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX


# data patch
def concat_pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        
        temp_labels = torch.LongTensor([LLAVA3D_IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # handel temp_token_type_ids for gemma
        temp_token_type_ids = torch.LongTensor([0] * max_item_length) # pad with 0 to indicate first scentence
        temp_token_type_ids[:feat['token_type_ids'].shape[0]] = feat['token_type_ids']
        feat['token_type_ids'] = temp_token_type_ids

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            # self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)

def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
    print('Replace train sampler!!')

def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    if train_dataset.use_raw_dataloader:
        return DataLoader(train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    print("Replace train dataloader!!")

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    outputs = model(**inputs)
    # Save past state if it exists
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
        loss *= self.accelerator.num_processes

    with torch.no_grad():
        logits = outputs["logits"]  # (bs, seq, voc)
        labels = inputs["labels"]  # (bs, seq)
        shift_logits = logits[..., :-1, :].argmax(-1).contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = (shift_labels >= model.action_tokenizer.translation_tokenizer.token_start_idx) & (
            shift_labels <= model.action_tokenizer.gripper_tokenizer.token_end_idx
        )
        gt_action_ids, pred_action_ids = shift_labels[mask], shift_logits[mask]
        correct_preds = gt_action_ids == pred_action_ids
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # NOTE: acc of translation, rotation and gripper
        token_start_idx, token_end_idx = (
            model.action_tokenizer.translation_tokenizer.token_start_idx,
            model.action_tokenizer.translation_tokenizer.token_end_idx,
        )
        translation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            model.action_tokenizer.rotation_tokenizer.token_start_idx,
            model.action_tokenizer.rotation_tokenizer.token_end_idx,
        )
        rotation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            model.action_tokenizer.gripper_tokenizer.token_start_idx,
            model.action_tokenizer.gripper_tokenizer.token_end_idx,
        )
        gripper_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        translation_gt_action_ids, translation_pred_action_ids = gt_action_ids[translation_mask], pred_action_ids[translation_mask]
        rotation_gt_action_ids, rotation_pred_action_ids = gt_action_ids[rotation_mask], pred_action_ids[rotation_mask]
        gripper_gt_action_ids, gripper_pred_action_ids = gt_action_ids[gripper_mask], pred_action_ids[gripper_mask]

        translation_correct_preds = translation_gt_action_ids == translation_pred_action_ids
        rotation_correct_preds = rotation_gt_action_ids == rotation_pred_action_ids
        gripper_correct_preds = gripper_gt_action_ids == gripper_pred_action_ids

        translation_action_accuracy = translation_correct_preds.sum().float() / translation_mask.sum().float()
        rotation_action_accuracy = rotation_correct_preds.sum().float() / rotation_mask.sum().float()
        gripper_action_accuracy = gripper_correct_preds.sum().float() / gripper_mask.sum().float()

        # convert to continue actions
        gt_actions = inputs["actions"].reshape(-1, 7).to(device="cpu", dtype=torch.float32)
        pred_actions = model.action_tokenizer.decode_token_ids_to_actions(pred_action_ids.cpu().numpy().reshape(-1, 3))
        l1_loss = nn.functional.l1_loss(torch.tensor(pred_actions), torch.tensor(gt_actions))

        self.log(
            {
                "accuracy": action_accuracy.item(),
                "translation_accuracy": translation_action_accuracy.item(),
                "rotation_accuracy": rotation_action_accuracy.item(),
                "gripper_accuracy": gripper_action_accuracy.item(),
                "l1_loss": l1_loss.item(),
            }
        )

    return (loss, outputs) if return_outputs else loss

def replace_compute_loss():
    transformers.Trainer.compute_loss = compute_loss
    print("Replace compute_loss!!")

class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            output_dir = args.output_dir
            if state.global_step > 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.processor.save_pretrained(output_dir)
        return control

class ProfilerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output")
        )
        self.profiler.__enter__()

    def training_step(self, model, inputs):
        output = super().training_step(model, inputs)
        self.profiler.step()
        return output

    def __del__(self):
        self.profiler.__exit__(None, None, None)