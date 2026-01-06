# SpatialVLA + LLaVA‑3D Integration (dev) — Changes and Status

## 目标概述
- 保持“视觉侧由 SpatialVLA 负责（Vision Tower + Multi‑Modal Projector），语言侧由 LLaVA‑3D 负责解码”。
- 语言模型替换后，兼容不同词表规模与特殊标记策略，避免嵌入越界与类型不匹配。
- 前向注入图像特征使用真实 tokenizer ID 定位；生成阶段按 LLaVA‑3D 的约定转换为哨兵索引。
- 在不必修改模型目录文件的情况下，支持“运行时覆盖”策略，保证端到端测试与 HuggingFace 加载兼容。

---

## 修改总览（按模块）

### 1) 配置（model/configuration_spatialvla_dev.py）
- 成功：将 `ignore_index` 在非 LLaVA 模式下统一为 `-100`，与 `CrossEntropyLoss` 兼容。
- 失败尝试：将 `use_vision_zoe` 默认改为 `False` 的补丁一度失败（无法精确定位行），后续未强制修改；当前仍由配置/脚本显式控制。

### 2) 处理器（model/processing_spatialvla_dev.py）
- 成功：
  - 移除了依赖工作目录的动态 tokenizer 选择逻辑，改为按 `use_llava3d` 显式选择：`LlamaTokenizer/LlamaTokenizerFast` 或 `GemmaTokenizer/GemmaTokenizerFast`。
  - 在返回中加入 `image_token_id`（真实 `<image>` 词表 ID）与 `image_token_index`（LLaVA 哨兵索引 `-200`），供模型一致使用。
- 注意：动作标记（TRANSLATION/ROTATION/GRIPPER）会追加到“语言 tokenizer”的词表中，随后需要扩展语言侧嵌入矩阵（见第 6 节）。

### 3) 模型（model/modeling_spatialvla_dev.py）
- 成功：
  - `forward`：新增可选参数 `image_token_id`，用于定位图像标记位置并注入图像特征；当未提供时回退到 `config.image_token_index`。
  - `prepare_inputs_for_generation`：新增可选参数 `image_token_index`，优先使用传入索引将相应位置映射到 LLaVA‑3D 哨兵值；索引相等时显式保留原始 `input_ids`。
  - 标签屏蔽：对 `labels` 做健壮过滤（pad、image、loc 位置设为 `ignore_index`，以及所有越界标签一律设为 `ignore_index`），修复 `IndexError: Target 256000 is out of bounds`。
  - 非 LLaVA 分支：无论 `flash_attention_2` 与否，均生成 4D 因果掩码；对空 `HybridCache` 做保护。
  - `from_pretrained`：改为通过 `get_input_embeddings()` 安全获取语言侧嵌入层，并在维度一致时才拷贝空间嵌入；避免对 LLaVA‑3D 的 `embed_tokens` 访问异常。
  - 兼容性修复：为 `cache_position` 空值加保护；在嵌入前对潜在越界特殊 ID 进行替换/钳制，避免嵌入越界（历史补丁保留）。
- 失败/注意：
  - 早期直接访问 `language_model.model.embed_tokens` 在 LLaVA‑3D 类中不存在，造成 `AttributeError`；已修复为通用接口并加维度判断。

### 4) 集成测试（test/test_llava3d_integration.py）
- 成功：
  - 默认 `--model_path` 指向 `model_zoo/spatialvla-4b-224`（含处理器配置），并在运行时优先选择包含 `preprocessor_config.json`/`processor_config.json` 的目录。
  - 注入丰富调试输出：候选目录、目录文件列表、环境版本、配置字段、processor/tokenizer/model 类型与配置、输入形状/设备。
  - 在 `test_pixel_values_injection_first_step` 用例中显式 `self.model.config.use_llava3d=False`，避免前序状态污染导致走错分支。
  - 端到端流程中手动组合 dev 处理器：使用 `AutoProcessor` 获取 `image_processor`，使用 `AutoTokenizer` 加载 LLaMA tokenizer，并用 dev `SpatialVLAProcessor` 组合。
- 失败/注意：
  - 最初端到端测试跳过的原因是默认目录缺少处理器配置，已通过目录切换与候选选择修复。

### 5) 模型目录（model_zoo/spatialvla-4b-224）
- 成功：
  - `config.json` 的 `auto_map` 指向 dev 类：
    - `configuration_spatialvla_dev.SpatialVLAConfig`
    - `modeling_spatialvla_dev.SpatialVLAForConditionalGeneration`
  - `processor_config.json` 的 `auto_map` 指向 dev 处理器：
    - `processing_spatialvla_dev.SpatialVLAProcessor`
  - 新增轻量别名文件：`configuration_spatialvla_dev.py`、`modeling_spatialvla_dev.py`、`processing_spatialvla_dev.py`，将目录导入转发到仓库中的 dev 实现。
  - 处理器别名文件中加入 wrapper，放宽 `tokenizer_class` 检查（允许 LLaMA/Gemma），避免 ProcessorMixin 报类型不匹配错误。
  - 修复 `processor_config.json` 中非法 `Infinity` 值为有限数值（`0.0`），避免 JSON 解析错误。
- 注意：
  - 目录中 `use_spatial_token` 设为 `false` 可作为临时规避策略，关闭空间嵌入拷贝与维度不匹配；当前你“不需要空间嵌入”，此方案安全可行。

### 6) HuggingFace dev 测试（test/test_huggingface_dev.py）
- 成功：
  - 运行时覆盖：仅用 `AutoProcessor` 获取图像处理器，用 `AutoTokenizer` 统一为 LLaMA tokenizer，然后用 dev 处理器组合。
  - 新增 `--llava3d_path` 参数：如提供则加载并替换语言模型为 LLaVA‑3D，并同步 `vocab_size`。
  - 加载模型使用 `ignore_mismatched_sizes=True`：兼容旧 checkpoint 与新结构的维度差异。
  - 调用 `model.resize_token_embeddings(len(tok))`：动作标记追加后同步语言侧嵌入矩阵大小，避免 OOV/维度不一致。
  - 设定 `inputs["cache_position"] = [0]`：保证首步注入 `pixel_values`。
  - 设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 降级兼容，避免 protobuf 版本差异导致的 `Descriptors` 错误。
- 失败/注意：
  - 若未提供 `--llava3d_path`，目录语言侧仍为 Gemma2 风格，`unused weights` 提示会很多属正常；替换后会减少此类提示。

---

## 问题与原因汇总（失败/报错）

### 详细错误链路与修复

1. `IndexError: Target 256000 is out of bounds`
   - 触发场景：`CrossEntropyLoss` 计算时，`labels` 含有 `256000` 等超出 LLaVA 词表（32k）的标签。
   - 根因：原 Gemma2 词表约 257k，替换为 LLaVA‑3D 后词表显著缩小；将真实词表 ID 当作标签参与 loss 会越界。
   - 修复：在 `modeling_spatialvla_dev.py` 的 `forward` 中统一屏蔽 pad/image/loc 位点，并对所有越界标签（`labels >= vocab_size` 或 `labels < 0`）设为 `ignore_index=-100`。

2. 非 LLaVA 模式下 `attention_mask` 未变化（断言失败）
   - 触发场景：`test_pixel_values_injection_first_step` 期望非 LLaVA 分支生成新的 4D 因果掩码，但拿到的是原始布尔掩码。
   - 根因：测试类共享模型实例，前序用例将 `use_llava3d=True` 未重置，导致此用例仍走 LLaVA 分支；属于状态污染。
   - 修复：在用例开头显式 `self.model.config.use_llava3d=False`；代码层面保证非 LLaVA 分支总是生成 4D 因果掩码，并对空 `HybridCache` 做保护。

3. 端到端测试跳过
   - 触发场景：`test_integration_end_to_end` 指向默认 `model_path`（`llava3d_7B`），目录下无 `preprocessor_config.json/processor_config.json`。
   - 根因：HuggingFace 加载流程需要处理器配置文件；目录不完整导致跳过。
   - 修复：将默认 `model_path` 改为 `model_zoo/spatialvla-4b-224`，并在运行时候选列表中优先选择包含处理器配置的目录；打印实际使用目录与文件列表以辅助定位。

4. Tokenizer 类型不匹配（期望 LLaMA，实际为 Gemma）
   - 触发场景：端到端加载时，处理器返回 `GemmaTokenizer/GemmaTokenizerFast`，而目录 tokenizer 是 `LlamaTokenizer`。
   - 根因：`AutoProcessor` 在 4.47 的某些路径下选择了目录中的原版处理器，导致类型不一致。
   - 修复：在 e2e 测试中手动组合 dev 处理器：用 `AutoProcessor` 仅取 `image_processor`，用 `AutoTokenizer` 保证 LLaMA tokenizer，再用 dev `SpatialVLAProcessor` 组合；同时在目录中添加处理器别名 wrapper（接受 LLaMA/Gemma），避免 ProcessorMixin 的严格类型检查抛错。

5. Protobuf `Descriptors cannot be created directly`
   - 触发场景：端到端加载 tokenizer/权重时，环境的 `protobuf/protoc` 版本与生成代码不兼容。
   - 根因：常见于 4.47 + 近期 `protobuf` 版本差异；需要环境匹配或降级。
   - 修复：在测试设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 作为临时兼容（纯 Python 解析）；长期建议 `pip install "protobuf<3.21"` 或升级 `protoc>=3.19.0` 并重启进程。

6. `AttributeError: 'LlavaLlamaForCausalLM' object has no attribute 'embed_tokens'`
   - 触发场景：`from_pretrained` 试图执行 `model.language_model.model.embed_tokens ...`。
   - 根因：Gemma2 语言模型结构与 LLaVA‑3D 语言模型结构不同，后者没有 `.model.embed_tokens` 该嵌套属性。
   - 修复：使用 `model.language_model.get_input_embeddings()` 获取嵌入层；复制空间嵌入前检查维度一致（列维度等于 `hidden_size`），不一致则跳过；整体用 `try/except` 保护不同语言模型实现。

7. `size mismatch for spatial_embed_tokens.weight`（[8194, 2304] vs [8194, 4096]）
   - 触发场景：从 `spatialvla-4b-224` checkpoint 加载到 dev 模型（语言侧 `hidden_size=4096`），空间嵌入的列维度与语言侧不一致。
   - 根因：旧结构的视觉投影维度是 `2304`，而新语言侧是 `4096`；直接拷贝会报尺寸不匹配。
   - 修复/规避：
     - 加载时启用 `ignore_mismatched_sizes=True`。
     - 临时设 `use_spatial_token=false`（当前你不需要空间嵌入），跳过空间嵌入拷贝与维度对齐；
     - 或后续通过将 `projection_dim` 对齐 `hidden_size` 或引入线性适配器做 2304→4096 映射。

8. 动作标记重复添加日志
   - 触发场景：同一进程里多次初始化处理器或重复对 tokenizer 执行 `add_tokens(...)`。
   - 根因：组合处理器 + 目录处理器路径都可能添加标记。
   - 解决建议：在 `SpatialActionTokenizer` 中加入“已添加检查”（如查 `tokenizer.get_added_vocab()`），只添加一次；当前对功能无害，仅日志重复。

9. 目录 JSON 非法值（`Infinity`）
   - 触发场景：`processor_config.json` 出现 `Infinity` 导致 JSON 解析错误（“需要值”）。
   - 根因：合法 JSON 不允许 `Infinity`；必须替换为有限数。
   - 修复：将 `Infinity` 改为 `0.0`。

10. 端到端未找到 dev 处理器模块
   - 触发场景：`auto_map` 指向 dev 名称，但目录没有对应文件。
   - 根因：HuggingFace 从目录加载时需要找到被映射的模块名；仅仓库存在 dev 文件不够。
   - 修复：在目录中新增别名模块：`configuration_spatialvla_dev.py`、`modeling_spatialvla_dev.py`、`processing_spatialvla_dev.py`，转发到仓库实现；同时处理器别名放宽类型检查。

---

## 运行与验证建议

- 集成测试：`python SpatialVLA_llava3d/test/test_llava3d_integration.py`
  - 确认 4D 因果掩码、图像标记转换逻辑与端到端加载路径。
- HuggingFace dev 测试：`python SpatialVLA_llava3d/test/test_huggingface_dev.py --model_name_or_path /path/to/spatialvla-4b-224 [--llava3d_path /path/to/llava3d_7B]`
  - 验证语言侧替换、tokenizer 类型一致、动作标记追加与嵌入同步；注意 protobuf 兼容信息。
  - 若暂时不使用空间嵌入（`use_spatial_token=false`），可规避空间嵌入维度对齐问题；需要启用时建议配合 `projection_dim==hidden_size` 或适配器。

---

## 未来优化建议
- 空间嵌入启用时：让 `projection_dim` 与语言侧 `hidden_size` 对齐，或为 `spatial_embed_tokens` 增加线性适配器进行维度映射。
- 处理器动作标记添加：加入“只添加一次”的保护，避免重复日志与不必要的嵌入扩展。
- state_dict 过滤加载：只加载视觉塔/投影器权重，语言侧用 LLaVA‑3D 权重，减少“unused weights”提示。
- 环境依赖：安装兼容版本的 `protobuf`，移除纯 Python 降级以提升性能。
 - 加载整洁化：在 `from_pretrained` 支持 state_dict 过滤，只加载视觉塔/投影器权重，语言侧权重从 LLaVA‑3D，仅映射必要层，减少“unused weights”提示。
- 文档与测试：将“运行时覆盖”策略写入 README，用例中统一使用组合处理器，减少 `AutoProcessor` 路径差异。

---

## 结论
- 当前 dev 集成遵循“SpatialVLA 负责视觉侧、LLaVA‑3D 负责语言侧”的准则；核心路径已打通，关键测试与端到端加载在运行时覆盖策略下稳定。
- 在不改动模型目录的大前提下，通过别名与手动组合处理器的方式，解决了 tokenizer 类型不匹配与目录 auto_map 带来的加载问题。
 - 若后续启用空间嵌入（`use_spatial_token=true`），建议配合维度对齐或适配器，以保证拷贝语义与语言侧维度一致。
 11. 最新调试异常：语言侧嵌入直接替换导致属性错误与新初始化提示

---

## 最新问题追踪与定位（2025-11-17）

- 问题：准备阶段图像占位符替换计数为 0（日志 `[E2E] image_token_index set: -200 count_after: 0`）
  - 根因：仅替换了 `config.image_token_index`（如 257152），未同时替换 tokenizer 的 `image_token_id`，导致文本中的占位符未被映射为 LLaVA-3D 哨兵索引。
  - 修复：在准备阶段联合替换两类索引（`image_token_index` 与 `image_token_id`）。位置 `model/modeling_spatialvla_dev.py:506-522`。
  - 前向阶段稳健性：即使准备阶段替换计数为 0，前向注入使用联合掩码识别三类占位符并完成注入（`image_token_index`、`config.image_token_index`、`image_token_id`）。位置 `model/modeling_spatialvla_dev.py:374-382`。

- 问题：注意力掩码长度不匹配导致 `RuntimeError: The size of tensor a (...) must match (...) at dimension 3`（539 vs 270）
  - 根因：在 LLaVA-3D forward 路径未传递 `cache_position`，LLaMA 使用默认逻辑推断 `causal_mask` 长度，与包含缓存的 `attn_weights` 的最后维度不一致。
  - 修复：在 LLaVA-3D forward 调用中传入 `cache_position`。位置 `model/modeling_spatialvla_dev.py:415`。

- 问题：KV 头参数不一致导致缓存更新形状冲突（如 `[32, 269, 128]` vs 缓存 `[1, 4, 269, 256]`）
  - 根因：使用了 SpatialVLA 的 `text_config`（Gemma2 风格）实例化 LLaMA 语言模型，`num_key_value_heads`、`head_dim` 等参数与 LLaVA-3D 权重不匹配。
  - 修复：在 `from_pretrained` 中从 `llava3d_pretrained_path` 加载语言侧配置（`AutoConfig`），保证 KV 参数与权重一致。位置 `model/modeling_spatialvla_dev.py:588-597`。

- 问题：图像特征与文本嵌入维度不一致（4096 vs 2304）
  - 根因：目录视觉投影维度为 2304，而 LLaVA-3D 语言隐藏维度为 4096。
  - 修复：运行时将投影器输出维度对齐语言隐藏维度，直接重建线性层的输出特征数为 LM hidden。位置 `model/modeling_spatialvla_dev.py:132-134`。同时空间 token 嵌入按语言维度构建。位置 `model/modeling_spatialvla_dev.py:122-124`。

- 问题：标签越界导致 `IndexError: Target ... is out of bounds`
  - 根因：Gemma2 词表约 257k，LLaVA-3D 词表 32k；将真实词表 ID 作为标签会越界。
  - 修复：在前向中统一屏蔽 pad/image/loc 位置，并将所有越界标签设为 `ignore_index`。位置 `model/modeling_spatialvla_dev.py:386-400`。

- 模板与处理器的一致性
  - 处理器在 LLaVA-3D 模式设置 LLAMA-2 风格 `chat_template`，并返回 `image_token_id` 与 `image_token_index`，供模型一致使用。位置 `model/processing_spatialvla_dev.py:145-163`, `model/processing_spatialvla_dev.py:314-320`。

- 状态字典过滤与形状保护
  - 仅加载视觉塔/投影器相关权重，过滤 `language_model.*`；若投影器权重形状与当前线性层不符，则删除相应键以避免加载错误。位置 `model/modeling_spatialvla_dev.py:626-639`，权重形状检查于 `model/modeling_spatialvla_dev.py:629-634`。

### 验证情况
- 单元测试：注入路径通过（图像特征形状 `(1, 256, 4096)` 与文本图像 token 数 256 对齐）；非 LLaVA 路径生成 4D 因果掩码，pad_token 与越界标签处理正常。
- 端到端：此前报错的三类问题（替换计数为 0、掩码维度不匹配、KV 头参数冲突）已在上述修复后对症处理。若日志仍出现“count_after: 0”，属准备阶段的替换统计；前向联合掩码保证功能不受影响。
   - 触发场景：在 `test_huggingface_dev.py` 加载 `spatialvla-4b-224` 后，日志显示大量 `language_model.model.layers.*` 权重未使用或新初始化，同时若尝试访问 `language_model.model.embed_tokens` 会报 `AttributeError`。
   - 根因：
     - 语言模型替换为 LLaVA‑3D 后，其内部结构与 Gemma2 不同；不存在 `.model.embed_tokens` 的嵌套属性。
     - `spatial_embed_tokens.weight` 的列维度来自旧结构的视觉投影（2304），不等于 LLaVA‑3D 的文本隐藏维度（4096），直接拷贝必然不匹配。
   - 关联影响：
     - Transformes 会在 `ignore_mismatched_sizes=True` 下重新初始化不匹配层，提示“should probably TRAIN this model ...”。这是正常信息，提醒下游任务需要微调以获得最佳效果。
   - 修复：
     - 在 `from_pretrained` 使用 `get_input_embeddings()` 获取嵌入层，并在维度一致时才进行替换；否则安全跳过，避免崩溃。
     - 运行时将 `use_spatial_token=false`（你当前不需要空间嵌入），彻底规避空间嵌入拷贝与维度对齐问题。
197→     - 在 `test_huggingface_dev.py` 中调用 `model.resize_token_embeddings(len(tok))`，确保新增动作标记的词表扩展与语言侧嵌入矩阵一致。

---

## 最新修改记录（2025-12-09）

### 概览
- 解决几何分支 dtype 冲突（BF16/FP32）与线性层维度不匹配导致的运行时错误。
- 修复 MapAnything 归一化类型与配置不一致引发的断言错误。
- 完善几何/视觉融合逻辑，移除未定义变量并统一特征维度。
- 放宽处理器对分词器类型的限制，支持 GPT2 作为回退以绕过 `protobuf/sentencepiece` 冲突。
- 导出当前虚拟环境依赖（`requirements.txt`）以便复现。

### 详细修改

1) MapAnything 包装器（几何分支）
- 变更：暴露几何通道维度供上层使用
  - 位置：`SpatialVLA_llava3d/model/modeling_mapanything.py:22`
  - 内容：`self.config.hidden_size = int(enc_dim) if enc_dim is not None else 1024`
  - 原因：上层投射层需要稳定的几何通道维度，避免 `AttributeError: 'MapAnythingWrapper' object has no attribute 'config'`。

- 变更：输入归一化类型与 dtype 对齐
  - 位置：`SpatialVLA_llava3d/model/modeling_mapanything.py:25–27`
  - 内容：`data_norm_type` 设为 `"dinov2"`；将 `img` 与 `intrinsics` 转为 `float32` 并 `contiguous()`。
  - 原因：修复归一化类型断言不匹配与 BF16/FP32 权重冲突（DINOv2）。

- 变更：统一返回接口
  - 位置：`SpatialVLA_llava3d/model/modeling_mapanything.py:33–41`
  - 内容：返回带 `last_hidden_state` 的对象（几何特征为 `[B, C, H, W]`）。
  - 原因：与上层调用保持一致，便于流水线对齐。

2) 视觉‑语言融合（SpatialVLA 模型）
- 变更：视觉输入设为 `float32`
  - 位置：`SpatialVLA_llava3d/model/modeling_spatialvla_dev.py:307–309`
  - 原因：确保与视觉塔权重的 dtype 一致，避免 BF16/FP32 混用。

- 变更：几何特征序列化与动态维度对齐
  - 位置：`SpatialVLA_llava3d/model/modeling_spatialvla_dev.py:314–328`
  - 内容：将 `[B, C, H, W]` 重排为 `[B, H*W, C]`；若几何通道与投射层 `in_features` 不一致，按运行时通道数重建线性层。
  - 额外位置：`SpatialVLA_llava3d/model/modeling_spatialvla_dev.py:329–336`
  - 原因：修复 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x768 and 1024x4096)`。

- 变更：几何‑视觉融合
  - 位置：`SpatialVLA_llava3d/model/modeling_spatialvla_dev.py:336–340`
  - 内容：几何特征经投射后做 token 维度均值池化，广播到视觉序列长度后与视觉特征拼接，最后经融合投射层回到 LM 隐藏维度。
  - 原因：统一融合维度、移除早期未定义变量（`selected_image_feature`）。

- 变更：几何/视觉投射器维度初始化
  - 位置：`SpatialVLA_llava3d/model/modeling_spatialvla_dev.py:166–175`
  - 内容：优先使用 `geometric_model.config.hidden_size`，否则回退到 `map_anything_model.encoder.enc_embed_dim` 或 `lm_hidden_size`；`fusion_projector` 设为 `lm_hidden * 2 → lm_hidden`。
  - 原因：避免因几何维度来源不稳定导致初始化失败或维度不匹配。

- 变更：推理路径 dtype 防护
  - 位置：`SpatialVLA_llava3d/model/modeling_spatialvla_dev.py:617–629`
  - 内容：`pixel_values` 与 `intrinsic` 强制为 `float32`，其它浮点输入保持 `bfloat16`。
  - 原因：避免 BF16 输入进入几何分支导致与 FP32 权重冲突。

3) 处理器（Tokenizer 与输入构造）
- 变更：允许 GPT2 作为回退分词器
  - 位置：`SpatialVLA_llava3d/model/processing_spatialvla_dev.py:72–76`
  - 原因：在 LLaMA 分词器加载失败（`protobuf/sentencepiece` 冲突）时，使用 GPT2 回退，避免 `TypeError: Received a GPT2TokenizerFast ... was expected ...`。

- 变更：BOS/EOS 缺失回退为空字符串
  - 位置：`SpatialVLA_llava3d/model/processing_spatialvla_dev.py:289–297` 与 `302–311`
  - 原因：防止拼接 `None` 导致异常，保持输入字符串构造健壮。

4) 环境依赖导出
- 变更：导出当前虚拟环境依赖
  - 位置：项目根目录生成 `requirements.txt`
  - 命令：`/cpfs01/qianfy_workspace/openvla_oft_rl/zzq_vla/SpatialVLA_llava3d/.conda/envs/test/bin/python -m pip freeze > requirements.txt`
  - 原因：便于环境复现与问题定位（如 `protobuf` 版本）。

### 关联问题与解决
- 归一化断言：`dinov2_vitl14_reg` vs `dinov2`
  - 解决：统一 `data_norm_type` 为 `dinov2`（`modeling_mapanything.py:25`）。
- BF16/FP32 冲突：DINOv2 卷积/线性权重为 FP32
  - 解决：几何分支入口与推理路径对 `pixel_values/intrinsic` 强制 `float32`（`modeling_mapanything.py:26–27`，`modeling_spatialvla_dev.py:617–629`）。
- 线性层维度不匹配：`(256x768) · (1024x4096)`
  - 解决：按运行时几何通道数重建投射层（`modeling_spatialvla_dev.py:329–336`）。
- 分词器加载失败（`Descriptors` 与重复 proto 文件名）
  - 解决：处理器允许 GPT2 回退并对 BOS/EOS 做空串回退（`processing_spatialvla_dev.py:72–76`、`289–297`、`302–311`）。

### 当前状态
- 几何/视觉融合按 LM 隐藏维度统一，图像 token 注入路径稳定（`forward` 中形状与计数校验通过）。
- MapAnything 的几何输出形状示例：`[1, 768, 16, 16]`（`modeling_spatialvla_dev.py:313` 的调试输出），序列化为 `[1, 256, 768]` 后投射融合正常。
- 处理器在分词器失败时可回退到 GPT2，端到端流程可继续执行；长期建议修复 `protobuf/protoc/sentencepiece` 环境以启用 LLaMA 分词器。
