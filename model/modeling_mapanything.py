#/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "map-anything"))
from mapanything.models.mapanything.model import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput


class MapAnythingWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.map_anything_model = MapAnything.from_pretrained(config.mapanything_model_name_or_path)
        enc_dim = getattr(self.map_anything_model.encoder, "enc_embed_dim", None)
        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.hidden_size = int(enc_dim) if enc_dim is not None else 1024

    def forward(self, pixel_values, intrinsics):
        views = [{"img": pixel_values, "data_norm_type": ["dinov2"], "intrinsics": intrinsics}]
        views[0]["img"] = views[0]["img"].float().contiguous()
        views[0]["intrinsics"] = views[0]["intrinsics"].float().contiguous()

        all_encoder_features = self.map_anything_model._encode_n_views(views)

        info_sharing_input = MultiViewTransformerInput(features=all_encoder_features)

        final_features, _ = self.map_anything_model.info_sharing(info_sharing_input)

        geometric_features = final_features.features[0]

        class _Out:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return _Out(geometric_features)
