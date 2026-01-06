# 在SpatialVLA/model/目录下创建新文件
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig
from transformers.generation import GenerationMixin
from typing import List, Optional, Tuple, Union

# 导入LLaVA-3D相关模块（使用项目内的绝对路径）
from LLaVA_3D.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from LLaVA_3D.llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX
from LLaVA_3D.llava.constants import IMAGE_TOKEN_INDEX as LLAVA3D_IMAGE_TOKEN_INDEX

# 常量值转换 (SpatialVLA与LLaVA-3D的常量值不同)
# SpatialVLA: IMAGE_TOKEN_INDEX = 256000
# LLaVA-3D: IMAGE_TOKEN_INDEX = -200

# 创建LLaVA-3D语言模型适配器
class LLaVA3DForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        from LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX
        self.ignore_index = LLAVA3D_IGNORE_INDEX
        # 根据配置选择合适的LLaVA-3D语言模型
        model_type = getattr(config, "llava3d_model_type", "llama")
        pretrained_path = getattr(config, "llava3d_pretrained_path", None)
        if model_type == "llama":
            # 优先使用预训练权重路径加载
            if pretrained_path is not None:
                llava_cfg = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
                # 关闭视频塔：评测中仅用单图像，避免对RGBD依赖；保留视觉塔配置以避免None导致的类型错误
                setattr(llava_cfg, "mm_video_tower", None)
                self.model = LlavaLlamaForCausalLM.from_pretrained(pretrained_path, low_cpu_mem_usage=True, config=llava_cfg)
            else:
                self.model = LlavaLlamaForCausalLM(config)
        elif model_type == "mistral":
            if pretrained_path is not None:
                llava_cfg = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
                setattr(llava_cfg, "mm_video_tower", None)
                self.model = LlavaMistralForCausalLM.from_pretrained(pretrained_path, low_cpu_mem_usage=True, config=llava_cfg)
            else:
                self.model = LlavaMistralForCausalLM(config)
        else:
            raise ValueError(f"Unsupported LLaVA-3D model type: {model_type}")
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self.model, "set_output_embeddings"):
            self.model.set_output_embeddings(new_embeddings)
        elif hasattr(self.model, "lm_head"):
            self.model.lm_head = new_embeddings
        else:
            raise AttributeError("Underlying LLaVA-3D model does not support setting output embeddings.")

    def get_decoder(self):
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()
        return self.model

    def set_decoder(self, value):
        self.model = value

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    # Pass-through helpers to align with SpatialVLA expectations
    def set_output_embeddings(self, new_embeddings):
        if hasattr(self.model, "set_output_embeddings"):
            self.model.set_output_embeddings(new_embeddings)
        elif hasattr(self.model, "lm_head"):
            self.model.lm_head = new_embeddings
        else:
            raise AttributeError("Underlying LLaVA-3D model does not support setting output embeddings.")

    def get_decoder(self):
        # Some HF models expose get_decoder; if not, return the core model
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()
        return self.model

    def set_decoder(self, value):
        # Allow replacing the underlying decoder/model if needed
        self.model = value

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()