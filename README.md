# R²Mem：方法代码与两阶段训练

这是 **PyTorch 方法实现**，提供残差记忆、固定视觉 teacher、历史重建、一阶段记忆训练、二阶段联合训练和权重保存/加载。机器人环境、驱动、原始数据读取、动作底座和预处理由使用方已有代码提供，本仓库不配置具体真机平台。

不包含数据、checkpoint、图片视频、历史报告、运行结果、环境或认证文件。只保留这份说明、方法配置、必要测试和最小接入示例。本仓库发布独立的方法版，消融项目单独维护。

## 1. 使用对方现有环境

先激活已有的 **PyTorch** 训练环境，再安装方法代码：

```bash
git clone https://github.com/zqqqqqqqj/R-.git
cd R-
python -m pip install -e '.[test]'
python -m pytest
```

本仓库不会安装 OpenPI、机器人 SDK、CUDA 或仿真环境，也不自动转换 JAX 动作模型。提供的是可独立调用的方法模块和单进程参考训练循环；已有分布式训练器可以直接集成 `R2Mem` / `Stage2Model`。

## 2. 两个阶段都包含

| 阶段 | 输入 | 训练目标 | 输出 |
| --- | --- | --- | --- |
| Stage 1 | 完整 episode 的冻结视觉 patch tokens，按 episode 划分 train/validation | 历史残差重建，训练 writer 和共享 residual reader | `stage1/final/`，含记忆权重、固定 teacher 投影/归一化和架构 |
| Stage 2 | Stage 1 权重、已有动作底座、当前动作样本及完整因果历史 | 使用方原动作目标 + `history_weight` × 历史重建 | `stage2/final/`，含 EMA 动作底座、EMA 记忆及接口权重 |

默认用真机数据重新训练 Stage 1，再做 Stage 2；动作底座从使用方已有的预训练权重初始化。代码不要求下载作者的仿真记忆 ckpt。Stage 1 不训练动作底座，不能独立用于机器人动作推理。

Stage 1 的 teacher 归一化只拟合训练 episode，留出集不参与拟合；历史重建只查询已经观测且早于当前时刻的历史。Stage 2 保留完整小记忆 merge 图，动作损失和辅助损失都可更新 writer/shared reader。关闭辅助损失时，记忆仍通过动作损失更新。

## 3. 对方只需接入已有数据和模型

```bash
cp examples/adapter_template.py r2mem_adapter.py
cp configs/method.yaml configs/method.local.yaml
```

把 `configs/method.local.yaml` 的 `adapter` 改成：

```yaml
adapter: r2mem_adapter:build_adapter
```

然后在 `r2mem_adapter.py` 中连接已有代码。完整可运行的接口示例在 `examples/synthetic.py`；模板中的未实现方法需要使用方填入真实数据/模型调用，不需要修改 `src/r2mem/`。

| adapter 方法 | 使用方提供的内容 |
| --- | --- |
| `memory_config()` | 视觉 embedding 宽度、proprio 宽度、使用的相机 slot、粗摘要宽度及 patch 网格；不预设8维/7维或相机字段名 |
| `memory_episodes(split)` | 可重复迭代的 `VisualEpisode` 序列，完整观测时间序列与冻结 patch tokens；episode ID 在 train/validation 间不重叠 |
| `fingerprint()` | 数据版本、episode 划分、视觉编码器、动作底座和预处理/归一化版本的稳定标识 |
| `build_backbone(device=...)` | 已初始化的 PyTorch 动作模型；使用方设置非视觉全参数或已有 adapter/LoRA 的训练范围，冻结视觉编码器 |
| `action_sample_count(split)` | 动作训练/留出集的样本数量 |
| `prepare_action_sample(...)` | `ActionSample`：当前 native prefix、语言 embeddings、proprio、完整因果历史特征，以及原动作目标所需 payload |
| `action_loss(backbone, sample, augmented_prefix)` | 使用带记忆 token 的 prefix 计算原来的可微标量动作目标 |

`ActionSample.features` 必须由 Stage 1 保存的同一个固定 teacher 产生，从 episode 的第一个观测开始，包含当前观测且不含未来图像；不能把独立打乱的帧当成历史。`current_time` 必须等于 `times[-1]`。语言与当前 prefix 可保留梯度，历史固定特征必须 detached。

对于现有 PyTorch OpenPI 的 `embed_prefix` 接口，`examples/openpi_loss.py` 演示了如何复用已有 observation/actions/preprocessing 并注入记忆 token，不修改原动作单位、绝对/增量约定、horizon 或 loss mask。动作模型为 JAX 时，需要由使用方提供 PyTorch 接口，不能直接接到本实现。

## 4. 真实训练入口

完成 adapter 后，分别执行：

```bash
# 一阶段：训练记忆
python scripts/train_stage1.py \
  --config configs/method.local.yaml \
  --device cuda:0 \
  --out runs/stage1

# 二阶段：加载刚完成的一阶段记忆，联合训练动作与记忆
python scripts/train_stage2.py \
  --config configs/method.local.yaml \
  --device cuda:0 \
  --memory-init runs/stage1/final \
  --out runs/stage2
```

参考配置沿用 memory 60k、action 30k 的初始预算，使用方可根据真机数据量调整。`batch_size` 是每次更新累计的样本数，每个 microbatch 为一个完整历史前缀。参考循环为单进程，不支持直接 `torchrun`；多卡训练请集成方法模块到已有训练器。

Stage 1 要求非空历史残差能量，并达到配置中的留出集重建增益阈值；否则权重会保存，但 CLI 返回失败，Stage 2 拒绝使用。Stage 2 的验证指标是 **离线原动作损失**，不是物理机器人成功率；真机评测仍使用对方的流程。

Stage 2 使用 EMA（默认0.999），验证和最终导出都使用 EMA 权重；本地 restart checkpoint 保存 raw model、optimizer、EMA、采样器和 RNG。恢复时保持配置、输入身份、参数布局和源码不变：

```bash
python scripts/train_stage1.py --config configs/method.local.yaml \
  --device cuda:0 --out runs/stage1 --resume

python scripts/train_stage2.py --config configs/method.local.yaml \
  --device cuda:0 --memory-init runs/stage1/final --out runs/stage2 --resume
```

## 5. 无真机、无数据的接口检查

以下示例仅用小型合成 tensor 和线性动作模型，验证方法的两阶段计算、梯度和导出，不代表真机训练效果。所有测试产物写在本地 `runs/` 并被 Git 忽略：

```bash
python scripts/train_stage1.py --config examples/synthetic.yaml --out runs/demo_stage1
python scripts/train_stage2.py --config examples/synthetic.yaml \
  --memory-init runs/demo_stage1/final --out runs/demo_stage2
```

## 6. checkpoint / Hugging Face

- `stage1/final/`：`memory.safetensors` + `config.json`。包含可复用的记忆和固定 teacher 状态，不含动作底座、optimizer、RNG 或原始数据。
- `stage2/final/`：EMA `backbone.safetensors`、`receiver.safetensors`、`memory/` 和 `config.json`。含完整方法权重；使用方仍需其 native 架构、输入/输出 transforms 和动作归一化资产。
- `checkpoints/`：本地精确续训状态，不属于默认公开权重包。权重文件与训练状态分开保存。

```python
from r2mem.checkpoint import load_memory, load_action_model

memory, metadata = load_memory("runs/stage1/final", device="cuda:0")
policy, metadata = load_action_model("runs/stage2/final", adapter, device="cuda:0")
# 用 policy.augment_prefix(sample) 得到带记忆的 prefix，再调用自己的 native 推理接口。
```

加载器校验架构/版本与文件 SHA；Stage 2 额外检查 Stage 1 的数据/预处理身份、划分、预算、seed 和重建验收。模型卡应由权重发布者补充实际底座、数据来源、normalizer/transform 资产、训练配置及实测结果。
