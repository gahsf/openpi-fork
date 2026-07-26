# OCAE V1 LIBERO 训练与测评执行方案

> 日期：2026-07-20
> 仓库：`/data1/gqy/workspace/openpi-overlockv1`
> 分支：`overlockv1`
> 基线提交：`9d0ae87 feat(ocae): add matched experiment baselines`
> 目标：在真实 `physical-intelligence/libero` 数据上完成 OCAE V1 的公平训练、分阶段 rollout 和多 seed 研究验收。

## 1. 执行结论

当前可以开始真实数据训练 pilot，但不应直接并行启动全部配置的 30,000-step 正式训练。

已经满足：

```text
LIBERO LeRobot 数据完整可读
π0 extra_delta_transform=True normalization stats 可读
E0/E1/E2/E5/E6/E7 共享同一份 normalization stats
真实 E7 transform batch shape 和 finite 检查通过
本地 pi0_base checkpoint 可用
大容量 checkpoint/log/eval 根目录可用
Python 3.8 LIBERO/MuJoCo 隔离环境已安装
libero_spatial 真实任务、初态和离屏渲染初始化通过
评测入口支持唯一视频名、JSONL 结果和禁止覆盖
```

正式实验前仍需完成：

```text
真实数据 2–5 step loss/checkpoint sanity
E1/E2/E7 各 1,000-step pilot
策略 server 与 LIBERO client 端到端 1 trial/task smoke
```

## 2. 研究目标与边界

核心研究问题：sample-conditioned overview/state 是否优于同位置 static LoRA 和同参数树 constant-conditioned OCAE。

主要判断：

```text
E7 > E1：收益不只是 action-expert Q/O 位置的低秩容量。
E7 > E2：收益来自 per-sample contextual conditioning，而不是 controller/encoder 参数量。
E5 vs E6：区分 Query 与 Output 调制的贡献。
```

本方案只覆盖 JAX π0 OCAE V1，不扩展到 π0.5、PyTorch 或新的 contextual ablation。

synthetic pilot 和训练 loss 不能代替 LIBERO rollout 成功率。checkpoint 不得依据 test rollout 结果挑选。

## 3. 已验证的基础条件

### 3.1 数据

```text
HF_HOME: /data1/gqy/data/huggingface
LeRobot path: /data1/gqy/data/huggingface/lerobot/physical-intelligence/libero
dataset size: 33 GiB
format: LeRobot v2.0
episodes: 1,693
frames: 273,465
tasks: 40
fps: 10
```

LeRobot 对 v2.0 的兼容性警告不阻塞当前训练。当前不转换数据版本，避免对已验证数据做非必要写入。

### 3.2 Normalization stats

原始计算结果：

```text
assets/pi0_libero/physical-intelligence/libero/norm_stats.json
```

六个 P9 配置读取的共享位置：

```text
assets/pi0_libero_e7_ocae_qo/physical-intelligence/libero/norm_stats.json
```

两者 JSON 数值相同；原文件末尾没有换行，因此文件字节哈希不同，但加载后的统计完全一致。

统计契约：

```text
state mean/std/q01/q99:   8 dimensions
action mean/std/q01/q99: 7 dimensions
extra_delta_transform:   True
```

不能替换为 `/data1/gqy/model/pi05_libero` 中的 stats，因为 π0.5 LIBERO 使用 `extra_delta_transform=False`。

### 3.3 真实 transform batch

E7 使用共享 stats 的离线真实 batch 已验证：

```text
base_0_rgb:        (1, 224, 224, 3) float32
left_wrist_0_rgb:  (1, 224, 224, 3) float32
right_wrist_0_rgb: (1, 224, 224, 3) float32
state:             (1, 32) float32, finite
tokenized_prompt:  (1, 48) int32
actions:           (1, 50, 32) float32, finite
```

### 3.4 模型、磁盘和输出目录

```text
pi0_base params: /data1/gqy/model/pi0_base/params
pi05_libero:     /data1/gqy/model/pi05_libero
free space:      approximately 12 TiB under /data1/gqy

checkpoints: /data1/gqy/checkpoints/ocaev1
logs:        /data1/gqy/logs/ocaev1
rollouts:    /data1/gqy/evals/ocaev1
```

所有训练命令必须显式传入本地 `pi0_base`，避免回退到配置中的 GCS 默认路径。

### 3.5 LIBERO 仿真环境

```text
Python: 3.8.20
venv: examples/libero/.venv
LIBERO config: examples/libero/.libero/config.yaml
LIBERO submodule: f78abd68
MuJoCo: 3.2.3
robosuite: 1.4.1
torch: 1.11.0+cu113
```

已初始化 `libero_spatial` task 0、固定初态和两路 `(256, 256, 3)` uint8 图像。

当前 EGL 运行时只接受 `MUJOCO_EGL_DEVICE_ID=0`。该编号是 EGL 可见设备编号，不应直接填写物理 GPU 编号 4–7。

## 4. 实验矩阵

| 编号 | 配置 | 方法 | 可训练参数 | 主要用途 |
|---|---|---|---:|---|
| F0 | `pi0_libero` | π0 full fine-tune | 全模型 | 官方 full-FT reference |
| E0 | `pi0_libero_e0_action_lora` | action-expert LoRA | 22,118,400 | 原始 PEFT baseline |
| E1 | `pi0_libero_e1_static_qo` | Static Q/O LoRA | 5,898,240 | 位置/容量对照 |
| E2 | `pi0_libero_e2_constant_qo` | Constant-input OCAE Q/O | 18,823,456 | E7 exact parameter-tree control |
| E5 | `pi0_libero_e5_ocae_q` | Sample OCAE Q-only | 15,874,336 | Query contribution |
| E6 | `pi0_libero_e6_ocae_o` | Sample OCAE O-only | 15,874,336 | Output contribution |
| E7 | `pi0_libero_e7_ocae_qo` | Sample OCAE Q/O | 18,823,456 | 主方法 |

E0/E1/E2/E5/E6/E7 的共同协议：

```text
dataset: physical-intelligence/libero
prompt_from_task: true
extra_delta_transform: true
initial checkpoint: same local pi0_base
batch size: 32
steps: 30,000
optimizer: AdamW
warmup: 1,000
peak lr: 1e-5
decay lr: 1e-6
EMA: disabled
train seeds: 42, 43, 44
```

F0 使用仓库官方 `pi0_libero` recipe，作为强 reference 单独报告；它的默认 peak LR 和 EMA 与 PEFT 组不同，不能用于 E7 contextual-conditioning 的 matched causal claim。

## 5. 固定环境变量

训练 shell：

```bash
export HF_HOME=/data1/gqy/data/huggingface
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export OCAE_CHECKPOINT_ROOT=/data1/gqy/checkpoints/ocaev1
export OCAE_LOG_ROOT=/data1/gqy/logs/ocaev1
export OCAE_EVAL_ROOT=/data1/gqy/evals/ocaev1
```

LIBERO client shell：

```bash
export LIBERO_CONFIG_PATH=/data1/gqy/workspace/openpi-overlockv1/examples/libero/.libero
export PYTHONPATH=/data1/gqy/workspace/openpi-overlockv1/third_party/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

每次启动训练前重新运行：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
df -h /data1/gqy
```

不要根据 2026-07-20 的一次 GPU 快照永久写死卡号。

## 6. Phase R0：真实数据 2–5 step sanity

目标：证明真实数据、模型、loss、optimizer 和 checkpoint 在同一链路中可用，不比较方法性能。

先在一张空闲 GPU 上运行 E7：

```bash
CUDA_VISIBLE_DEVICES=<one_free_gpu> \
uv run scripts/train.py pi0_libero_e7_ocae_qo \
  --exp-name=realdata_sanity_seed42_20260720 \
  --checkpoint-base-dir=/data1/gqy/checkpoints/ocaev1 \
  --weight-loader.params-path=/data1/gqy/model/pi0_base/params \
  --seed=42 \
  --batch-size=1 \
  --num-workers=0 \
  --num-train-steps=5 \
  --save-interval=5 \
  --keep-period=5 \
  --no-wandb-enabled
```

验收：

```text
真实 batch 初始化成功
5 step loss 和 grad_norm finite
无 OOM
最终 checkpoint 完整写入
相同配置可 restore
restore 后 action sampling finite，shape=(1, 50, 32)
```

该 run 只做工程验证，不能纳入性能表。

### 6.1 2026-07-20 执行结果

```text
config: pi0_libero_e7_ocae_qo
GPU: physical GPU 4
batch size: 1
steps: 5
loss: 0.2121, 0.1904, 0.1650, 0.1382, 0.1002
grad norm: 0.0094, 0.0059, 0.0063, 0.0059, 0.0052
checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo/realdata_sanity_seed42_20260720/4
checkpoint size: approximately 5.0 GiB
```

从该 checkpoint 重新加载模型，并用真实 LIBERO episode 0 observation、固定零噪声和 2 个 flow sampling steps 验证：

```text
prompt: put the white mug on the left plate and put the yellow and white mug on the right plate
actions shape: (50, 7)
actions finite: true
```

R0 验收通过。上述 5-step loss 只证明工程链路正常，不作为收敛或方法性能结论。

## 7. Phase R1：E1/E2/E7 1,000-step pilot

### 7.1 资源配置

正式 batch 32 建议预留四张空闲 RTX 5090：

```text
CUDA_VISIBLE_DEVICES contains exactly 4 GPUs
batch_size=32
fsdp_devices=4
local batch per device=8
one training job at a time during the first pilot
```

如果 4-GPU pilot OOM，可以降低 batch 做工程定位，但研究用 pilot 和正式 run 必须恢复相同 global batch 32。不能只为某个方法改变 batch、LR 或 gradient accumulation。

### 7.2 配置顺序

```text
1. pi0_libero_e1_static_qo
2. pi0_libero_e2_constant_qo
3. pi0_libero_e7_ocae_qo
```

命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu0,gpu1,gpu2,gpu3> \
uv run scripts/train.py <config_name> \
  --exp-name=pilot_seed42_steps1000 \
  --checkpoint-base-dir=/data1/gqy/checkpoints/ocaev1 \
  --weight-loader.params-path=/data1/gqy/model/pi0_base/params \
  --seed=42 \
  --fsdp-devices=4 \
  --num-train-steps=1000 \
  --save-interval=500 \
  --keep-period=500 \
  --no-wandb-enabled
```

每个方法必须使用独立的配置目录；相同 `exp-name` 不会发生冲突，因为 checkpoint 路径包含配置名。

### 7.3 Pilot 验收

```text
loss、grad_norm、param_norm 全程 finite
loss 平滑后有合理下降趋势
没有持续数据加载 stall
E1 static LoRA 梯度顺序仍符合 B-first
E2/E7 controller-first、encoder/LoRA-second 梯度顺序符合预期
gate 与 effective delta ratio 不饱和
冻结参数 fingerprint 不变
checkpoint restore 和 sampling 通过
记录实际 step/s、峰值显存、checkpoint 大小
```

Go：三组均通过工程验收。
No-Go：任一组出现 gate 饱和、effective delta 异常、freeze 破坏、NaN、restore 失败或无法解释的显存异常；先诊断，不启动 30k。

### 7.4 2026-07-21 执行结果

三组使用完全相同的训练协议顺序完成：

```text
GPU: 1 x RTX 5090
global batch size: 32
fsdp_devices: 1
seed: 42
steps: 1,000
log interval: 50
checkpoint steps: 500, 999
WandB: disabled
```

最初尝试了两张 RTX 5090。两卡 FSDP 在首次训练计算触发 CUDA illegal address；两卡纯数据并行在完整训练图的首次 collective/JIT 调用停止推进。`NCCL_CUMEM_HOST_ENABLE=0` 可以让独立两卡 `psum` 探针通过，但不能让完整训练图稳定执行。因此三组统一改用已验证稳定的单卡路径；global batch、seed、schedule 和训练步数均未改变。

| 配置 | step 0 loss | step 950 loss | step 950 grad norm | 稳态速度 | 峰值显存 | checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| E1 `pi0_libero_e1_static_qo` | 0.1809 | 0.1104 | 0.0276 | 约 1.4–1.6 s/step | 约 17.1 GiB | 4.8 GiB/个 |
| E2 `pi0_libero_e2_constant_qo` | 0.1809 | 0.1134 | 0.0264 | 约 1.1–1.2 step/s | 约 17.1 GiB | 5.0 GiB/个 |
| E7 `pi0_libero_e7_ocae_qo` | 0.1809 | 0.1134 | 0.0264 | 约 1.0–1.2 step/s | 约 17.1 GiB | 5.0 GiB/个 |

验收结果：

```text
三组 1,000 steps 完成
全部已记录 loss、grad_norm、param_norm finite
无 OOM、NaN 或 CUDA 运行错误
三组 step 500 和 step 999 checkpoint 完整落盘
三组 step 999 checkpoint 使用对应配置恢复成功
同一真实 LIBERO episode 0 observation、固定零噪声、2 个 flow steps 采样成功
三组 actions shape: (50, 7)
三组 actions finite: true
```

checkpoint 根目录：

```text
/data1/gqy/checkpoints/ocaev1/pi0_libero_e1_static_qo/pilot_seed42_steps1000
/data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo/pilot_seed42_steps1000
/data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo/pilot_seed42_steps1000
```

训练、保存、恢复和推理链路通过。标准训练日志没有记录 gate、effective delta ratio 和冻结参数 fingerprint，因此不能仅根据相近的 1,000-step training loss 比较 E1/E2/E7 方法效果；完整诊断和最终判定见下一节。

### 7.5 Pilot checkpoint 诊断与 Go / No-Go

使用 `scripts/diagnose_ocaev1_pilot.py` 对 step 999 checkpoint 做统一检查：

```text
真实 LIBERO batch size: 4
冻结参数: 与 pi0_base 在 checkpoint dtype cast 后逐叶 exact equal
gate 饱和阈值: |g| > 0.9 的比例必须 < 1%
effective delta ratio: finite、非零且 < 1
ratio 定义: ||g * (A @ B) * alpha/rank||F / ||W_base||F
```

| 配置 | 冻结叶 exact equal | Q `|g|>0.9` | O `|g|>0.9` | Q delta ratio | O delta ratio | 判定 |
|---|---:|---:|---:|---:|---:|---|
| E1 static Q/O | 50/50 | static gate=1，不适用 | static gate=1，不适用 | 0.400% | 0.572% | Pass |
| E2 constant Q/O | 50/50 | 100% | 100% | 1.322% | 1.763% | Fail |
| E7 sample Q/O | 50/50 | 100% | 100% | 1.321% | 1.762% | Fail |

E2 每个样本使用相同 reference input，样本间 gate 距离为 0 符合基线设计；失败原因是 gate 全饱和，不是冻结参数或数值错误。

E7 做了逐层详细追踪，确认饱和不是诊断取样假象：

```text
真实 prefix input 相对样本距离: 9.10%
scene context 相对样本距离: 1.93%
state context 相对样本距离: 1.24%
controller hidden 相对样本距离: 0.52%
step 999 raw Q/O gate 范围: [-4.875, 4.8125]
step 999 raw Q |g|>3: 100%
step 999 raw O |g|>3: 98.61%
tanh 后 Q/O gate 样本距离: 0
```

输入、encoder context 和 raw gate 都存在样本差异；但 raw gate 量级过大，`tanh` 将差异压成相同的 `±1`。E7 因而失去本实验要验证的 sample-conditioned gate 行为。

饱和随训练加重：

| E7 checkpoint | Q `|g|>0.9` | O `|g|>0.9` | Q raw max abs | O raw max abs |
|---|---:|---:|---:|---:|
| step 500 | 58.33% | 84.03% | 2.016 | 1.930 |
| step 999 | 100% | 100% | 4.875 | 4.875 |

诊断输出：

```text
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e1.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e2.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_detail.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_step500_detail.json
```

R1 最终判定：**No-Go**。checkpoint 可恢复、输出 finite、冻结参数完整、effective delta 比例有限，但 E2/E7 gate 验收失败。不得使用当前配置启动七组 30k。下一步应先做带逐步 gate 统计的短程受控实验，定位饱和开始的 step，并只验证一个最小的 gate 尺度约束方案；新 pilot 通过同一诊断后才能恢复 30k 计划。

### 7.6 Gate scale 0.25 受控短程试验

只引入一个变量：在进入 `tanh` 前将 controller gate logit 乘以 `0.25`。原 E1/E2/E7 配置继续保持 `1.0`，新配置为 `pi0_libero_e7_ocae_qo_gate025`。选择 `0.25` 的依据是原 E7 step 999 raw gate 最大绝对值 `4.875`；缩放后为 `1.219`，低于 `atanh(0.9)=1.472`。

同时将以下指标接入正常训练前向，每 10 steps 记录一次，不额外执行 prefix 前向：

```text
q/o_gate_abs_max
q/o_gate_abs_gt_0_9
q/o_raw_gate_abs_max
```

500-step 受控 pilot：

```text
GPU: physical GPU 0, 1 x RTX 5090
global batch size: 32
seed: 42
schedule: 与原 E7 完全相同
steps: 500
checkpoints: 250, 499
wall time: 约 11 分 39 秒（含首次 JIT，不含最终异步保存）
```

训练期 gate 轨迹：

| step | Q gate max abs | O gate max abs | Q/O `|g|>0.9` |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0% / 0% |
| 250 | 0.0373 | 0.0338 | 0% / 0% |
| 400 | 0.2025 | 0.1939 | 0% / 0% |
| 490 | 0.4223 | 0.4096 | 0% / 0% |

checkpoint 独立诊断：

| checkpoint | 冻结叶 exact equal | Q gate max | O gate max | Q delta ratio | O delta ratio | 判定 |
|---|---:|---:|---:|---:|---:|---|
| step 250 | 50/50 | 0.0378 | 0.0339 | 0.0231% | 0.0291% | Pass |
| step 499 | 50/50 | 0.4551 | 0.4414 | 0.4070% | 0.5904% | Pass |

step 499 的 Q/O gate 样本间 L2 距离分别为 `0.00720`/`0.00737`，均非零；实际 tanh logit 最大绝对值分别为 `0.4922`/`0.4746`。因此该 checkpoint 同时满足 finite、非饱和、sample-varying、effective delta 非零和冻结参数完整。

产物：

```text
checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo_gate025/gate025_pilot_seed42_steps500_20260721_v2/{250,499}
log: /data1/gqy/logs/ocaev1/e7_gate025_pilot_single_gpu_batch32_steps500_20260721_v2.log
diagnostic: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_gate025_step250_detail.json
diagnostic: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_gate025_step499_detail.json
```

本阶段判定：**500-step gate025 修复候选 Go**。它解决了原 E7 在 step 500 已有 58.33%/84.03% gate 饱和的问题，但尚不等于完整 R1 Go。下一关是按同协议验证 E7 gate025 到 1,000 steps，并补齐 constant E2 gate025 对照；两者通过后才能讨论 30k。

### 7.7 Gate025 扩展到 1,000 steps 的结果

E7 gate025 按相同 seed、global batch、学习率和单卡协议扩展训练。训练在 step 770 左右提前停止，因为 gate 已明确重新饱和：

```text
step 660: Q/O |g|>0.9 = 0.02% / 0.00%
step 680: Q/O |g|>0.9 = 2.12% / 4.59%
step 700: Q/O |g|>0.9 = 17.53% / 36.15%
step 720: Q/O |g|>0.9 = 45.41% / 77.80%
step 740: Q/O |g|>0.9 = 70.04% / 96.40%
step 760: Q/O |g|>0.9 = 82.69% / 98.88%
step 770: Q/O |g|>0.9 = 86.67% / 99.68%
```

step 770 时 Q/O gate 最大值为 `0.9883`/`0.9863`，raw gate 最大值为 `2.55`/`2.49`。训练过程中未出现 NaN、OOM 或 CUDA 错误；失败原因单纯是 gate 饱和。该进程已停止，避免无意义地继续到 step 1000。

本阶段判定：**gate025 1000-step No-Go**。`0.25` 只把饱和从原 E7 的约 step 500 延后到约 step 660–770，不能作为正式配置。由于 E7 主方法已经否证该单一尺度方案，不再启动同尺度 E2 gate025；E1/E2/E7 小规模实验仍未全部通过，30k 继续禁止。

### 7.8 Raw gate L2 正则受控试验

固定 gate scale 会被 controller 权重增长补偿，因此改为直接约束进入 `tanh` 前的 raw Q/O gate logits：

```text
total_loss = task_loss + coefficient * 0.5 * (mean(q_raw^2) + mean(o_raw^2))
```

训练日志保留原 `loss` 作为 task loss，并新增 `gate_regularization_loss`、`total_loss` 和 Q/O gate 指标。scale 恢复为 `1.0`。静态检查、10 个 controller 测试、5 个配置测试和 Pi0 aux-loss 前向测试均通过；真实 1-step GPU smoke 也已通过。

| 系数 / checkpoint | Q `|g|>0.9` | O `|g|>0.9` | Q/O delta ratio | 判定 |
|---|---:|---:|---:|---|
| `1e-3`, step 499 | 6.944% | 11.111% | 0.870% / 1.339% | No-Go，约束过弱 |
| `1e-2`, step 499 | 0% | 0% | 0.353% / 0.605% | Pass |
| `1e-2`, step 999 | 1.3889% | 3.4722% | 0.612% / 1.005% | 严格 No-Go |

step 999 的冻结叶仍为 `50/50 exact equal`，Q/O 样本距离非零，说明机制有效且 conditioning 没有退化为常量；但后期仍轻微超过 `<1%` 的严格阈值。因此 `1e-2` 不能进入 30k。

产物：

```text
1e-3 checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo_gate_l2_1e3/gate_l2_1e3_pilot_seed42_steps500_20260722/499
1e-3 log: /data1/gqy/logs/ocaev1/e7_gate_l2_1e3_pilot_single_gpu_batch32_steps500_20260722.log
1e-3 diagnostic: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260722/e7_gate_l2_1e3_step499_detail.json
1e-2 checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo_gate_l2_1e2/gate_l2_1e2_pilot_seed42_steps500_20260722/999
1e-2 logs: /data1/gqy/logs/ocaev1/e7_gate_l2_1e2_pilot_single_gpu_batch32_steps500_20260722.log
             /data1/gqy/logs/ocaev1/e7_gate_l2_1e2_resume_step500_to1000_20260722.log
1e-2 diagnostic: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260722/e7_gate_l2_1e2_step999_detail.json
```

`gate_l2_regularization=3e-2` 已按独立 1,000-step 协议完成 E7 sample 和 E2 constant 对照：

| 配置 | Q `|g|>0.9` | O `|g|>0.9` | Q/O delta ratio | freeze | 判定 |
|---|---:|---:|---:|---:|---|
| E7 sample, step 999 | 0.6944% | 0% | 0.443% / 0.705% | 50/50 | Pass |
| E2 constant, step 999 | 0.6944% | 0% | 0.518% / 0.852% | 50/50 | Pass |

E7 Q/O sample distance 非零；E2 sample distance 为零符合 constant 对照定义。两组诊断均为 `passed: true`，训练末期正则损失约占 task loss 的 5%，未见 NaN、OOM、CUDA 或 checkpoint 错误。

```text
E7 checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo_gate_l2_3e2/gate_l2_3e2_pilot_seed42_steps1000_20260722/999
E7 log: /data1/gqy/logs/ocaev1/e7_gate_l2_3e2_pilot_single_gpu_batch32_steps1000_20260722.log
E7 diagnostic: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260722/e7_gate_l2_3e2_step999_detail.json
E2 checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo_gate_l2_3e2/gate_l2_3e2_pilot_seed42_steps1000_20260722/999
E2 log: /data1/gqy/logs/ocaev1/e2_gate_l2_3e2_pilot_single_gpu_batch32_steps1000_20260722.log
E2 diagnostic: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260722/e2_gate_l2_3e2_step999_detail.json
```

截至 2026-07-23，E1 static、修正后的 E2 constant、修正后的 E7 sample 小规模验收全部通过。正式 `pi0_libero_e2_constant_qo` 和 `pi0_libero_e7_ocae_qo` 已统一使用 L2 `3e-2`；R1 gate 准入由 No-Go 更新为 Go，可以进入 seed 42 的 30k staged training，但仍需逐阶段监控 gate 和 task loss。

### 7.9 Seed 42 正式 30k 第一批与 E2 意外恢复

2026-07-23 在三张 GPU 上并行启动 E1/E2/E7 正式 30k。E1 和 E7 均于 step 29999 正常完成并保存 final checkpoint；E7 step 29900 的 Q/O `|g|>0.9` 均为 0%，gate L2 `3e-2` 在长程训练末期仍有效。

E2 于约 step 3990 遇到 `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`。错误前 loss、gate 和正则均正常，step 3900 Q/O 饱和率为 0%，因此判定为 GPU/运行时意外，不是模型数值或 gate 失败。原实验目录保留 step 3000 checkpoint 和完整失败日志。

为避免 Orbax 后续自动保留策略删除原始失败现场，2026-07-24 将原 E2 checkpoint 目录完整复制到独立续训目录，再从 step 3000 resume：

```text
原失败目录: /data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo/seed42_30k_20260723
原失败日志: /data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_20260723.log
续训目录: /data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo/seed42_30k_resume_from3000_20260724
续训日志: /data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_resume_from3000_20260724.log
GPU: physical GPU 2
launcher PID: 2216
```

日志已确认 Orbax 从 step 3000 完整恢复约 6.1 GiB 参数和 143.6 MiB train state，并继续进入 step 3000 之后的训练循环。原始失败目录未被修改或覆盖。

## 8. Phase R2：策略 server—simulator smoke

在 OCAE checkpoint 产生前，先使用已知可用的 π0.5 LIBERO checkpoint 验证服务链路。

Server：

```bash
CUDA_VISIBLE_DEVICES=<server_gpu> \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/data1/gqy/model/pi05_libero
```

Client：

```bash
LIBERO_CONFIG_PATH=/data1/gqy/workspace/openpi-overlockv1/examples/libero/.libero \
PYTHONPATH=/data1/gqy/workspace/openpi-overlockv1/third_party/libero \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name=libero_spatial \
  --args.num-trials-per-task=1 \
  --args.seed=7 \
  --args.video-out-path=/data1/gqy/evals/ocaev1/smoke_pi05_libero_spatial/videos \
  --args.results-out-path=/data1/gqy/evals/ocaev1/smoke_pi05_libero_spatial/results.jsonl
```

验收：

```text
client 连接 server
10 个 task 各完成 1 episode
图像、state、prompt、action shape 正常
无连续 websocket/MuJoCo exception
JSONL 包含 10 条 episode 和 1 条 summary
视频文件名不发生覆盖
```

Smoke 的成功率只用于发现严重 inference mismatch，不作为模型排名。

### 8.1 2026-07-20 执行结果

```text
checkpoint: /data1/gqy/model/pi05_libero
policy config: pi05_libero
server GPU: physical GPU 3
suite: libero_spatial
rollout seed: 7
trials per task: 1
total tasks / episodes: 10 / 10
successes: 10
episode errors: 0
elapsed: approximately 14 minutes
```

各 task 完成步数：

```text
113, 118, 122, 97, 171, 105, 119, 131, 110, 125
```

结果证据：

```text
/data1/gqy/evals/ocaev1/smoke_pi05_libero_spatial_20260720/results.jsonl
10 episode records + 1 summary record
10 uniquely named success videos
```

旧版 robosuite 与 MuJoCo 的 EGL device 编号检查存在冲突：client 设置 `CUDA_VISIBLE_DEVICES=3` 时，robosuite 要求物理编号 3，而 MuJoCo 要求 EGL 重映射编号 0。最终采用 server 固定 `CUDA_VISIBLE_DEVICES=3`、client 不设置 `CUDA_VISIBLE_DEVICES`、`MUJOCO_EGL_DEVICE_ID=0`，链路正常。

server 日志中有一次由 TCP 端口 readiness probe 引起的无效 WebSocket handshake；它发生在 rollout client 连接前。10 个正式 episode 的 JSONL `error` 均为 null。

R2 验收通过，server 已正常停止，GPU 3 已释放。该 π0.5 10/10 结果只证明 benchmark 链路正常，不作为 OCAE 性能结果。

## 9. Phase R3：Seed 42 正式训练

R1 和 R2 均通过后，依次运行：

```text
F0, E0, E1, E2, E5, E6, E7
```

PEFT/OCAE 命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu0,gpu1,gpu2,gpu3> \
WANDB_MODE=offline \
WANDB_DIR=/data1/gqy/logs/ocaev1 \
uv run scripts/train.py <config_name> \
  --exp-name=seed42 \
  --checkpoint-base-dir=/data1/gqy/checkpoints/ocaev1 \
  --weight-loader.params-path=/data1/gqy/model/pi0_base/params \
  --seed=42 \
  --fsdp-devices=4
```

F0 使用：

```bash
CUDA_VISIBLE_DEVICES=<gpu0,gpu1,gpu2,gpu3> \
WANDB_MODE=offline \
WANDB_DIR=/data1/gqy/logs/ocaev1 \
uv run scripts/train.py pi0_libero \
  --exp-name=seed42 \
  --checkpoint-base-dir=/data1/gqy/checkpoints/ocaev1 \
  --weight-loader.params-path=/data1/gqy/model/pi0_base/params \
  --seed=42 \
  --fsdp-devices=4
```

禁止使用 `--overwrite`。当前训练实现会递归清空同名 checkpoint 目录。

中断恢复：使用原命令和相同配置，增加：

```bash
--resume
```

不得同时传入 `--resume` 和 `--overwrite`。

### 9.1 Checkpoint 策略

```text
save interval: 1,000
keep period: 5,000
fixed comparison checkpoint: final 30k run checkpoint
current loop的最终 checkpoint 目录通常为 29999
```

如果正式引入独立 validation 集，应在所有方法开始前固定划分和选择规则。否则统一使用 final checkpoint，不能用 test rollout 选择 5k/10k/15k/20k/25k/30k。

## 10. Phase R4：Seed 42 分阶段 rollout

每个 checkpoint 必须使用训练时完全相同的配置加载。E7 示例：

```bash
CUDA_VISIBLE_DEVICES=<server_gpu> \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_e7_ocae_qo \
  --policy.dir=/data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo/seed42/29999
```

固定 rollout 协议：

```text
suites: libero_spatial, libero_object, libero_goal, libero_10
resize_size: 224
replan_steps: 5
num_steps_wait: 10
rollout seed: 7
same initial states and task order for every method
```

### 10.1 OCAE smoke

E1/E2/E7 每个 suite 各 `1 trial/task`，保存视频。

### 10.2 Seed 42 screening

E0/E1/E2/E5/E6/E7 每个 suite 各 `10 trials/task`。使用唯一结果路径，大规模 screening 可以关闭视频：

```bash
--args.no-save-videos
```

每个方法共：

```text
4 suites × 10 tasks × 10 trials = 400 episodes
```

screening 只作为资源 Go/No-Go 和严重回归检查，不用于选择 checkpoint。

### 10.3 Final rollout

最终报告使用：

```text
4 suites × 10 tasks × 50 trials = 2,000 episodes / checkpoint
```

结果路径命名：

```text
/data1/gqy/evals/ocaev1/<config>/seed<seed>/step29999/<suite>/results.jsonl
```

评测入口发现已有 `results.jsonl` 会直接报错，防止重复运行覆盖证据。需要重跑时创建新的显式 run 目录，不能删除旧结果。

### 10.4 2026-07-25 E1/E2/E7 完整 benchmark 启动记录

按 final 协议直接启动 E1/E2/E7 三组完整 LIBERO benchmark。每组依次运行 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，每 task 50 trials，固定 seed 7，关闭视频；每 checkpoint 2,000 episodes，合计 6,000 episodes。

```text
E1 server: GPU 2, port 8001, PID 11048
E2 server: GPU 3, port 8002, PID 11049
E7 server: GPU 4, port 8003, PID 11050
E1 orchestrator PID: 12315, start 2026-07-25T04:38:36Z
E2 orchestrator PID: 12465, start 2026-07-25T04:38:54Z
E7 orchestrator PID: 12647, start 2026-07-25T04:39:14Z
```

结果根目录：

```text
/data1/gqy/evals/ocaev1/full_benchmark_seed42_20260725
```

日志根目录：

```text
/data1/gqy/logs/ocaev1/libero_full_benchmark_seed42_20260725
```

三个 client 均未设置 `CUDA_VISIBLE_DEVICES`，使用历史已验证的 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`MUJOCO_EGL_DEVICE_ID=0`。三个 WebSocket 连接、LIBERO 环境初始化和 JSONL 写入均已验证。首个 Spatial task 0 episode 三组均运行到 230 steps 后失败，`error=null`；当前只有 1 episode/model，不用于判断最终性能。三路并发下失败 episode 约 159–171 秒，完整运行预计约四天。

### 10.5 2026-07-25 Full benchmark Early Stop

因 Spatial 前两项已出现明显负面趋势，用户要求停止完整 benchmark。已向三个 simulator client、三个 orchestrator 和三个 policy server 发送正常终止信号；所有进程均已退出，GPU server 计算进程已释放，checkpoint、日志和 JSONL 均保留，未删除结果。

停止时最终有效 episode：

| 模型 | episodes | successes | success rate | errors | task 分布 |
|---|---:|---:|---:|---:|---|
| E1 static | 105 | 2 | 1.90% | 0 | task0=50, task1=50, task2=5 |
| E2 constant | 104 | 0 | 0% | 0 | task0=50, task1=50, task2=4 |
| E7 sample | 103 | 7 | 6.80% | 0 | task0=50, task1=50, task2=3 |

已完整完成的前两个 Spatial task 上，E1 为 2/100、E2 为 0/100、E7 为 7/100。所有失败记录均 `error=null`，说明不是 EGL、WebSocket 或 simulator 异常，而是策略未完成任务。当前结果不足以报告完整 LIBERO benchmark，但已足以判定继续消耗约四天运行 6,000 episodes 的收益过低，因此 Early Stop 合理。

保留证据：

```text
/data1/gqy/evals/ocaev1/full_benchmark_seed42_20260725
/data1/gqy/logs/ocaev1/libero_full_benchmark_seed42_20260725
```

## 11. Phase R5：Go / No-Go 与多 seed

Seed 42 screening 的继续条件：

```text
E7 > E1
E7 > E2
E5 或 E6 至少一个显示正向贡献
不存在只由单一 task 或异常 episode 驱动的假趋势
```

趋势成立后，优先补：

```text
E1/E2/E7: seeds 43 and 44
```

三者趋势稳定后，再补 E0/E5/E6 的 seeds 43/44，形成完整三 seed 报告。F0 至少报告 seed 42；资源允许时同样补齐三个 seed，并明确它是 official full-FT reference。

趋势不成立时停止扩展性消融，先检查：

```text
训练是否真正收敛
gate/delta ratio 是否过小或饱和
E2/E7 conditioning 是否按预期区分
serve config 与 checkpoint 是否匹配
rollout exception 和 task-level 分布
```

E8–E13 contextual contribution 消融不属于当前执行范围。

## 12. 报告指标

训练侧：

```text
loss curve
grad norm
step time / samples per second
peak GPU memory
checkpoint size
gate magnitude
effective Q/O delta ratio
frozen fingerprint
restore/sampling result
```

Rollout 侧：

```text
success / trials per task
suite success rate
four-suite macro average
exception count and type
mean and standard deviation across training seeds
episode-level binomial confidence interval as supplemental evidence
```

训练 seed 是模型结论的主要统计单位；不能把同一 checkpoint 的 2,000 个 episode 当作 2,000 个独立模型重复。

## 13. 资源与风险控制

### 13.1 存储

`/data1/gqy` 当前空间充足，但正式实验仍应按 pilot 实测 checkpoint 大小规划。建议至少为本项目保留 1 TiB，所有 checkpoint、WandB offline 文件和 rollout 结果写到 `/data1/gqy`。

根文件系统只剩约 80 GiB：

```text
/root/.cache/uv: approximately 13 GiB
/root/.cache/jax: currently small
```

每个正式 run 前后检查根盘；不要把 checkpoint 或 rollout 视频写到 `/tmp` 或根盘。

### 13.2 GPU

```text
pilot start: one job only
formal train: four free GPUs, fsdp_devices=4
policy server and MuJoCo client: separate processes
do not assume memory.used=0 solely from utilization=0
```

### 13.3 可恢复性

```text
唯一 exp-name
禁止 --overwrite
中断后 --resume
结果 JSONL 使用唯一路径
任何重跑保留旧 checkpoint、日志和 rollout
```

### 13.4 公平性

```text
相同 pi0_base 初始化
相同 dataset/stats/transforms
相同 batch/steps/schedule/seed
相同 checkpoint selection rule
相同 rollout seed/task order/initial states
不根据 synthetic loss 或 test rollout 选择方法超参
```

## 14. 执行清单

### 已完成

- [x] 下载并核对 LIBERO LeRobot 数据。
- [x] 核对 π0 extra-delta normalization stats。
- [x] 将共享 stats 接到 E0/E1/E2/E5/E6/E7 预期目录。
- [x] 读取一个真实 E7 transformed batch。
- [x] 建立 checkpoint/log/eval 大盘目录。
- [x] 安装 Python 3.8 LIBERO 评测环境。
- [x] 配置非交互 LIBERO path。
- [x] 初始化真实 LIBERO task、固定初态和离屏相机。
- [x] 加入结构化 JSONL 和不覆盖的 rollout 输出。
- [x] π0.5 policy server—LIBERO client 1 trial/task smoke。

### 下一步

- [x] E7 真实数据 2–5 step sanity。
- [x] checkpoint restore 和真实 observation sampling。
- [x] E1/E2/E7 1,000-step pilot 训练、checkpoint 和 restore/sampling。
- [ ] 补齐 pilot gate、effective delta ratio 和冻结参数 fingerprint 诊断。
- [ ] Seed 42 七组训练。
- [ ] Seed 42 staged rollout。
- [ ] 根据 Go / No-Go 补 seed 43/44。
- [ ] Final 50 trials/task benchmark 和统计报告。

## 15. 最近的执行顺序

严格按以下顺序继续：

```text
1. 重新检查四张空闲 GPU。
2. 跑 E7 真实数据 5-step sanity。
3. restore checkpoint 并做 finite action sampling。
4. 跑 π0.5 policy server 与 LIBERO client smoke。
5. 跑 E1/E2/E7 各 1,000-step pilot。（已完成）
6. 汇总吞吐、显存、loss 和 checkpoint 数据，并补齐 gate、freeze 诊断。
7. 通过后启动 seed 42 正式训练。
```

在第 6 步完整通过前，不启动六配置全量 30k；在 seed 42 rollout 显示合理趋势前，不启动新消融。
