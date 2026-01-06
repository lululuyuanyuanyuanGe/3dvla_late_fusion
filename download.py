# save as download_model.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IPEC-COMMUNITY/spatialvla-4b-224-pt",
    local_dir="/cpfs01/qianfy_workspace/openvla_oft_rl/zzq_vla/SpatialVLA_llava3d/model_zoo/spatialvla-4b-224",
    local_dir_use_symlinks=False,
    resume_download=True
)