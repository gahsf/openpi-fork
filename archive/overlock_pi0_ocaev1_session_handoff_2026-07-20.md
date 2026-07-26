# OpenPI OCAE V1 会话交接文档

> 更新时间：2026-07-25
> 用途：下次会话开始时优先读取本文，快速恢复项目目标、设计约束、代码进度、验证证据和下一步工作。
> 当前结论：E1/E2/E7 30k 与 checkpoint 工程验收通过，但完整 LIBERO rollout 在 Spatial 前两个 task 上成功率仅 E1=2%、E2=0%、E7=7%。2026-07-25 已按用户要求 Early Stop；所有测评进程已退出，结果与日志完整保留。当前主要问题已从 gate 饱和转为策略实际任务成功率过低。

## 1. 仓库与运行环境

```text
工作目录: /data1/gqy/workspace/openpi-overlockv1
当前分支: overlockv1
远端仓库: git@github.com:gahsf/openpi-fork.git
远端分支: origin/overlockv1
本轮实验基点: 9d0ae87 feat(ocae): add matched experiment baselines
最新提交: 使用 git log -1 查看本轮 LIBERO 记录提交
上游 main 基点: 15a9616 update output objects to support batching (#975)
```

本地模型：

```text
π0 base 参数:   /data1/gqy/model/pi0_base/params
π0.5 base 参数: /data1/gqy/model/pi05_base/params
π0.5 LIBERO:    /data1/gqy/model/pi05_libero
```

GPU 和资源情况：

```text
默认验证 GPU: 按空闲情况选择
RTX 5090 一共 8 张，编号 0-7，每张约 32 GiB
最近真实 LIBERO pilot 使用物理 GPU 3
单卡 global batch 32 峰值显存约 17.1 GiB
```

仓库规则：

1. 禁止删除任何文件。
2. 不得覆盖或回退用户已有改动。
3. `uv.lock` 是用户本地镜像源改动，不要修改、提交或还原它。
4. 修改应小而可验证，每阶段单独提交。
5. 源码修改已经获得用户授权。

本轮提交后预期保留的本地状态：

```text
 M uv.lock
?? examples/libero/.libero/
```

`examples/libero/.libero/config.yaml` 含本机绝对路径，只在本机使用，不提交。

## 2. 项目目标

项目目标是在 OpenPI 的 JAX π0 中实现：

```text
Overview-Conditioned Action Expert (OCAE)
```

核心思想：

1. 先让冻结的 VLM prefix 完成视觉—语言联合编码。
2. 从 contextualized prefix hidden states 提取场景—任务 overview。
3. 将 overview 与机器人 state 编码后送入 controller。
4. controller 生成逐层、逐 head 的 Q/O 条件 gate。
5. 通过低秩分支调制 action expert 的 Query 和/或 Output projection。
6. 保持 prefix K/V、VLM expert、原 action expert 权重和原 flow matching 主损失不变。

一句话公式：

```text
H_prefix, KV_prefix = VLM(prefix_embeddings)
c = OverviewContextEncoder(H_prefix, prefix layout/masks)
s = StateContextEncoder(obs.state)
g_q, g_o = Controller(c, s)
v_theta = ActionExpert(suffix, KV_prefix, g_q, g_o)
```

主要研究问题：sample-conditioned overview/state 是否能比普通 static LoRA 或 constant-conditioned OCAE 更有效地适配 action expert。

## 3. 当前 V1 范围与明确不做的内容

V1 支持：

```text
框架: JAX / Flax NNX + Linen bridge
模型: Pi0Config(pi05=False)
主干: PaliGemma VLM expert + Gemma action expert
context 来源: contextualized prefix_out
调制位置: action expert Q 和/或 O
训练参数: OCAE 新增参数
训练损失: 原始 flow matching loss
```

V1 不支持：

```text
π0.5
PyTorch 训练/推理版本
context token 插入
prefix K/V 修改
suffix K/V 调制
attention-logit bias
SigLIP 或 VLM expert 微调
完整 QKV hypernetwork
依赖 noisy action 或 timestep 的 overview
额外辅助监督损失
```

关键框架结论：当前仓库本次实现路径是 JAX；没有为 OCAE V1 实现 PyTorch 对应版本。不要在后续实验中混合修改 JAX 和 PyTorch。

为什么先做 π0 而非 π0.5：

1. π0 的连续 state 位于 action expert suffix，新增 StateContext 的语义清晰。
2. π0.5 默认把 state 离散化写入语言 prompt，再加入连续 state 路径会引入重复条件。
3. 先在 π0 上验证方法，可避免把 OCAE 收益与 π0.5 输入语义变化混在一起。

## 4. 关键架构与实现决策

### 4.1 训练前向改为 prefix cache + suffix

原训练路径：

```text
[prefix, suffix] -> joint Gemma forward
```

当前训练路径：

```text
prefix-only forward -> contextualized prefix_out + KV cache
suffix-only forward with cached prefix KV
```

因为 attention mask 禁止 prefix 读取 suffix，所以 prefix hidden states 和 prefix K/V 不依赖 suffix。OCAE disabled 时，拆分前向与原 joint forward 已验证等价。

### 4.2 Context 必须来自 contextualized prefix

没有直接对 Gemma 输入前的 image/language embedding 做简单 mean pooling。overview 来自 prefix-only Gemma 的输出，使其已经包含视觉、语言以及跨模态关系。

### 4.3 只调制 action expert Q/O

OCAE conditional low-rank branch 只作用于 expert index 1：

```text
支持 target: q / o / q_o
不修改 expert 0
不修改 base expert Q/K/V/O 权重
不修改 cached prefix K/V
不修改 attention mask、RoPE、flow head
```

### 4.4 Identity initialization 与梯度顺序

Sample/constant OCAE：

```text
controller Q/O gate 零初始化
初始输出与 base 模型一致
第 1 步通常 controller 获得梯度
gate 离开零后，第 2 步 encoder 和 conditional LoRA 获得梯度
```

Static Q/O LoRA：

```text
gate 固定为 1
LoRA A 随机初始化
LoRA B 零初始化
初始 A @ B = 0，保持 identity
第 1 步 B 有梯度、A 为 0
第 2 步 A 开始有梯度
```

### 4.5 Checkpoint 合并与冻结

OCAE 模型从本地 `pi0_base` 加载：

```text
/data1/gqy/model/pi0_base/params
```

base checkpoint 中不存在的 OCAE 参数由当前初始化参数树精确补入；非 OCAE 参数如果缺失仍应报错。训练时 freeze filter 只允许 OCAE 新参数更新，冻结参数 fingerprint 在 P8 已验证不变。

## 5. 分步计划与完成状态

原计划依赖链：

```text
P0 基线记录
 -> P1 训练前向拆分
 -> P2 Prefix layout 结构化
 -> P3 Overview/State/Controller
 -> P4 Conditional Query LoRA
 -> P5 Conditional Output LoRA
 -> P6 Checkpoint 与 freeze
 -> P7 端到端 JAX/checkpoint 回归
 -> P8 Smoke training
 -> P9 核心实验准备与 pilot
```

当前状态：P0–P9 均已完成其工程验收范围；P9 的真实 benchmark 研究验收未完成。

## 6. 提交历史与每阶段产物

| 阶段 | Commit | 内容 | 状态 |
|---|---|---|---|
| P0/计划 | `012348d` | 新方案、分步计划、P0 基线和讲解文档 | 已推送 |
| P1 | `c10a420` | split training prefix/suffix forward | 已推送 |
| P2 | `a384df9` | prefix layout 静态 metadata | 已推送 |
| P3 | `0058a8e` | overview/state encoder 与 zero-gate controller | 已推送 |
| P4 | `52dbbd6` | conditional Query LoRA | 已推送 |
| P5 | `b712987` | conditional Output LoRA，支持 Q/O | 已推送 |
| P6 | `a237ab5` | base checkpoint merge 与 OCAE-only freeze | 已推送 |
| P7 | `cbd37d4` | 端到端 JAX、训练、采样与 checkpoint roundtrip | 已推送 |
| P8 | `427adab` | smoke training 配置与验证 | 已推送 |
| P9 | `9d0ae87` | matched baselines 与六组核心实验配置 | 已推送 |

远端已同步到：

```text
origin/overlockv1 -> 9d0ae87
```

主要生产代码落点：

```text
src/openpi/models/pi0.py
src/openpi/models/pi0_config.py
src/openpi/models/gemma.py
src/openpi/models/model.py
src/openpi/models/overview_action_conditioning.py
src/openpi/training/config.py
src/openpi/training/weight_loaders.py
```

主要测试落点：

```text
src/openpi/models/pi0_split_forward_test.py
src/openpi/models/prefix_layout_test.py
src/openpi/models/overview_action_conditioning_test.py
src/openpi/models/conditional_action_lora_test.py
src/openpi/models/pi0_test.py
src/openpi/models/model_test.py
src/openpi/training/weight_loaders_test.py
```

## 7. P8 Smoke Training 最终证据

最终采用的 smoke schedule：

```text
steps: 50
batch size: 1
warmup: 5
peak lr: 1e-5
decay lr: 1e-6
freeze: OCAE-only
base checkpoint: /data1/gqy/model/pi0_base/params
```

结果：

```text
所有 loss finite
controller 从 step 1 起有梯度
overview/state/Q-A/Q-B/O-A/O-B 从 step 2 起均有非零梯度
最终 Q max gate: 约 0.223
最终 O max gate: 约 0.216
最终 Q effective delta ratio: 约 0.159%
最终 O effective delta ratio: 约 0.218%
gate 未饱和
action-expert frozen fingerprint 未改变
checkpoint restore 后 fixed-noise 输出逐位一致
peak memory: 约 9.845 GiB
```

P8 结论：训练链路、梯度链路、冻结、保存/恢复和资源占用均符合预期，可以进入实验阶段。

保留的 P8 checkpoint（因项目禁止删除文件）：

```text
/tmp/openpi_ocaev1_p8_smoke_step50_params            约 4.8 GiB
/tmp/openpi_ocaev1_p8_smoke_lr1e5_step50_params      约 4.8 GiB
```

## 8. P9 核心实验设计

已经加入的配置：

| 编号 | 配置名 | 作用 |
|---|---|---|
| E0 | `pi0_libero_e0_action_lora` | 原始 action-expert LoRA baseline |
| E1 | `pi0_libero_e1_static_qo` | Static Q/O LoRA matched baseline |
| E2 | `pi0_libero_e2_constant_qo` | Constant-input OCAE Q/O matched baseline |
| E5 | `pi0_libero_e5_ocae_q` | Sample-conditioned OCAE Q-only |
| E6 | `pi0_libero_e6_ocae_o` | Sample-conditioned OCAE O-only |
| E7 | `pi0_libero_e7_ocae_qo` | Sample-conditioned OCAE Q/O，主方法 |

共同训练设置：

```text
dataset: physical-intelligence/libero
prompt_from_task: true
extra_delta_transform: true
initializer: pi0_base
batch size: 32
steps: 30,000
optimizer: AdamW
warmup: 1,000
peak lr: 1e-5
decay lr: 1e-6
EMA: disabled
default seed: 42
```

共享 normalization assets：

```text
./assets/pi0_libero_e7_ocae_qo/physical-intelligence/libero
```

参数量审计：

| 实验 | 可训练参数 |
|---|---:|
| E0 action LoRA | 22,118,400 |
| E1 Static Q/O | 5,898,240 |
| E2 Constant Q/O | 18,823,456 |
| E5 OCAE Q | 15,874,336 |
| E6 OCAE O | 15,874,336 |
| E7 OCAE Q/O | 18,823,456 |

最重要的 matched-control 事实：

```text
E2 与 E7 的总参数量、可训练参数量和参数树完全一致。
两者唯一关键差异是 OCAE conditioning input：
E2 对所有 sample 使用相同 deterministic nonzero reference；
E7 使用每个 sample 自身的 prefix/state。
```

研究判据：

```text
E7 > E1：收益不只是来自 action-expert Q/O 位置上的低秩容量。
E7 > E2：收益来自 sample-conditioned context，而不是同参数量 controller/encoder 本身。
E5 vs E6：区分 Query 与 Output 调制的贡献。
```

建议正式实验还加入 full fine-tune π0 baseline。已有 `pi0_libero` checkpoint 只能作为一个已有微调 baseline；不能拿它作为 E0/E1/E2/E5/E6/E7 的共同初始化。所有方法必须从同一个 `pi0_base` 独立训练，才能公平比较。

## 9. P9 全尺寸 Synthetic Pilot 结果

统一 pilot 条件：

```text
GPU: physical GPU 0
base checkpoint: /data1/gqy/model/pi0_base/params
batch size: 1
steps: 2
input: synthetic fixed-shape batch
warmup: 5
peak/decay lr: 1e-5 / 1e-6
sampling flow steps: 2
checkpoint saving: disabled
```

验证结果：

1. 六配置均能从本地完整 π0 base 初始化。
2. 六配置均完成两步训练且 loss finite。
3. 六配置均完成 finite action sampling，输出 shape 为 `(1, 50, 32)`。
4. E1 的梯度顺序符合预期：step 1 只有 B，step 2 A 开始有梯度。
5. E2/E5/E6/E7 的梯度顺序符合预期：step 1 controller；step 2 encoder 和对应 Q/O LoRA。
6. E5 只有 Q 分支梯度；E6 只有 O 分支梯度；E7 同时有 Q/O 梯度。
7. 六配置同进程 JAX peak memory high-water mark 约 6.792 GiB，无 OOM。

回归结果：

```text
P1-P9 CPU focused regression: 39 passed
OCAE-disabled 默认完整 π0 GPU regression: 1 passed
最终配置契约复查: 6 passed
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

重要边界：这些结果只证明工程链路正确，不能用 synthetic loss 数值比较方法优劣，也不能据此声称 OCAE 提升了 LIBERO 成功率。

## 10. LIBERO 数据和实验现状

当前数据与资源：

```text
third_party/libero 子模块
examples/libero/convert_libero_data_to_lerobot.py
/data1/gqy/data/huggingface/lerobot/physical-intelligence/libero
/data1/gqy/checkpoints/ocaev1
/data1/gqy/logs/ocaev1
/data1/gqy/evals/ocaev1
/data1/gqy/model/pi05_libero
assets/pi0_libero_e7_ocae_qo/physical-intelligence/libero/norm_stats.json
```

已完成的真实数据与评测链路：

```text
E7 真实数据 5-step sanity: passed
π0.5 LIBERO spatial smoke: 10/10 successes, 0 episode errors
E1/E2/E7 1,000-step pilot: completed
三组 step 500/999 checkpoint: completed
三组 step 999 restore + real observation sampling: passed
```

Pilot 统一协议：

```text
GPU: 1 x RTX 5090
global batch size: 32
fsdp_devices: 1
seed: 42
steps: 1,000
WandB: disabled
```

Pilot 结果：

| 配置 | step 0 loss | step 950 loss | step 950 grad norm | checkpoint |
|---|---:|---:|---:|---:|
| E1 `pi0_libero_e1_static_qo` | 0.1809 | 0.1104 | 0.0276 | 4.8 GiB/个 |
| E2 `pi0_libero_e2_constant_qo` | 0.1809 | 0.1134 | 0.0264 | 5.0 GiB/个 |
| E7 `pi0_libero_e7_ocae_qo` | 0.1809 | 0.1134 | 0.0264 | 5.0 GiB/个 |

所有已记录 loss、grad norm 和 param norm 均 finite。三组 final checkpoint 使用同一真实 LIBERO episode 0 observation、固定零噪声和 2 个 flow steps 采样，输出 shape 均为 `(50, 7)` 且全部 finite。

checkpoint：

```text
/data1/gqy/checkpoints/ocaev1/pi0_libero_e1_static_qo/pilot_seed42_steps1000/{500,999}
/data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo/pilot_seed42_steps1000/{500,999}
/data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo/pilot_seed42_steps1000/{500,999}
```

日志：

```text
/data1/gqy/logs/ocaev1/e1_pilot_single_gpu_batch32_20260721.log
/data1/gqy/logs/ocaev1/e2_pilot_single_gpu_batch32_20260721.log
/data1/gqy/logs/ocaev1/e7_pilot_single_gpu_batch32_20260721.log
```

多卡结论：两卡 FSDP 在 RTX 5090 上触发 CUDA illegal address；两卡纯数据并行在完整训练图首次 collective/JIT 调用停止推进。`NCCL_CUMEM_HOST_ENABLE=0` 仅让独立 `psum` 探针通过，未解决完整训练图。因此三组统一使用单卡，同时保持 global batch、seed、schedule 和 steps 不变。

Pilot checkpoint 诊断：

| 配置 | 冻结叶 exact equal | Q `|g|>0.9` | O `|g|>0.9` | Q delta ratio | O delta ratio | 判定 |
|---|---:|---:|---:|---:|---:|---|
| E1 static Q/O | 50/50 | static，不适用 | static，不适用 | 0.400% | 0.572% | Pass |
| E2 constant Q/O | 50/50 | 100% | 100% | 1.322% | 1.763% | Fail |
| E7 sample Q/O | 50/50 | 100% | 100% | 1.321% | 1.762% | Fail |

E7 真实输入和 encoder 输出存在样本差异；step 999 raw Q/O gate 已达到约 `[-4.875, 4.8125]`，经过 `tanh` 后样本 gate 全部变为相同的 `±1`。这不是取样或恢复假象。step 500 时 Q/O 的 `|g|>0.9` 比例已经分别达到 58.33%/84.03%，说明饱和随训练加重。

R1 最终判定：**No-Go**。冻结参数完整、effective delta 有限、checkpoint restore 和 sampling 正常，但 E2/E7 gate 验收失败。当前配置不得启动七组 30k。

诊断脚本和输出：

```text
scripts/diagnose_ocaev1_pilot.py
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e1.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e2.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_detail.json
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_step500_detail.json
```

Gate scale 0.25 修复候选：

```text
config: pi0_libero_e7_ocae_qo_gate025
唯一训练行为变化: tanh 前 gate logit 乘以 0.25
global batch size: 32
seed: 42
steps: 500
GPU: physical GPU 0
checkpoints: 250, 499
```

训练日志每 10 steps 直接记录 Q/O gate max、`|g|>0.9` 比例和 raw gate max。step 490 时 Q/O gate max 为 `0.4223`/`0.4096`，饱和比例始终为 0。

| checkpoint | 冻结叶 exact equal | Q gate max | O gate max | Q delta ratio | O delta ratio | 判定 |
|---|---:|---:|---:|---:|---:|---|
| step 250 | 50/50 | 0.0378 | 0.0339 | 0.0231% | 0.0291% | Pass |
| step 499 | 50/50 | 0.4551 | 0.4414 | 0.4070% | 0.5904% | Pass |

step 499 的 Q/O gate 样本距离非零，说明 sample conditioning 没有被 `tanh` 抹平。本阶段为 **500-step gate025 修复候选 Go**；完整 R1 仍是 No-Go，必须继续验证 E7 gate025 1,000 steps 并补 E2 gate025。

```text
checkpoint: /data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo_gate025/gate025_pilot_seed42_steps500_20260721_v2/{250,499}
log: /data1/gqy/logs/ocaev1/e7_gate025_pilot_single_gpu_batch32_steps500_20260721_v2.log
diagnostics: /data1/gqy/evals/ocaev1/pilot_diagnostics_20260721/e7_gate025_step{250,499}_detail.json
```

E7 gate025 扩展训练失败记录：

```text
exp: gate025_pilot_seed42_steps1000_20260721_v3
log: /data1/gqy/logs/ocaev1/e7_gate025_pilot_single_gpu_batch32_steps1000_20260721_v3.log
step 700: Q/O |g|>0.9 = 17.53% / 36.15%
step 750: Q/O |g|>0.9 = 76.83% / 98.28%
step 770: Q/O |g|>0.9 = 86.67% / 99.68%
结论: gate025 延迟但没有消除饱和；训练已停止，不启动同尺度 E2。
```

Raw gate L2 正则试验（2026-07-22）：

```text
实现: total_loss = task_loss + gate_regularization_loss；loss 继续表示 task loss
1e-3 step 499: Q/O |g|>0.9 = 6.944% / 11.111%，No-Go
1e-2 step 499: Q/O |g|>0.9 = 0% / 0%，Pass
1e-2 step 999: Q/O |g|>0.9 = 1.3889% / 3.4722%，严格 No-Go
step 999: freeze 50/50 exact equal，Q/O sample distance 非零
验证: ruff passed；controller 10 tests、gate config 5 tests、Pi0 aux-loss test passed；真实 1-step GPU smoke passed
```

`3e-2` 最终结果：E7 step 999 Q/O 饱和率为 0.6944%/0%，delta 为 0.443%/0.705%；E2 step 999 为 0.6944%/0%，delta 为 0.518%/0.852%。两组 freeze 均为 50/50，诊断均 `passed: true`。E7 sample distance 非零；E2 为零符合 constant 定义。正式 `pi0_libero_e2_constant_qo` 和 `pi0_libero_e7_ocae_qo` 已切换到 L2 `3e-2`。

Seed 42 正式训练最新记录（2026-07-24）：

```text
E1: 30k 完成，final checkpoint step 29999
E7: 30k 完成，final checkpoint step 29999；step 29900 Q/O |g|>0.9 = 0% / 0%
E2: 原进程约 step 3990 遇到 CUDA_ERROR_LAUNCH_FAILED
E2 原 checkpoint: .../pi0_libero_e2_constant_qo/seed42_30k_20260723/3000
E2 续训 checkpoint root: .../pi0_libero_e2_constant_qo/seed42_30k_resume_from3000_20260724
E2 续训日志: /data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_resume_from3000_20260724.log
E2 resume: GPU 2，launcher PID 2216，已确认从 step 3000 restore 并继续递增
```

恢复采用“复制原 checkpoint 到新实验目录后 resume”，因此原始失败目录与日志保持不变，后续 Orbax 清理只影响续训副本。

完整 LIBERO benchmark 最新记录（2026-07-25）：

```text
协议: spatial/object/goal/libero_10；50 trials/task；seed 7；无视频
E1: server GPU2 port8001 PID11048；orchestrator PID12315
E2: server GPU3 port8002 PID11049；orchestrator PID12465
E7: server GPU4 port8003 PID11050；orchestrator PID12647
results: /data1/gqy/evals/ocaev1/full_benchmark_seed42_20260725
logs: /data1/gqy/logs/ocaev1/libero_full_benchmark_seed42_20260725
```

三组均已连接 server、初始化 Spatial task 0 并写入首条 episode JSONL。首条均为 230 steps failure、`error=null`；样本数仅 1，不作效果判断。按当前三路并发速度，完整 6,000 episodes 预计约四天。

LIBERO Early Stop 最终记录（2026-07-25）：

```text
E1: 105 episodes, 2 successes, 1.90%, errors=0
E2: 104 episodes, 0 successes, 0%, errors=0
E7: 103 episodes, 7 successes, 6.80%, errors=0
completed task0+task1: E1 2/100, E2 0/100, E7 7/100
```

三个 simulator client、orchestrator 和 policy server 已全部停止。失败 episode 均 `error=null`，属于策略失败而非测评链路失败。保留目录：

```text
/data1/gqy/evals/ocaev1/full_benchmark_seed42_20260725
/data1/gqy/logs/ocaev1/libero_full_benchmark_seed42_20260725
```

详细方案和本轮证据见：

```text
archive/overlock_pi0_ocaev1_libero_training_evaluation_plan_2026-07-20.md
```

## 11. LIBERO 数据获取记录

以下是历史获取方案；当前 LeRobot 数据已经下载到 `/data1/gqy/data/huggingface/lerobot/physical-intelligence/libero`，无需重复下载。

优先方案：从 Hugging Face 获取已经转换好的 LeRobot 数据集：

```bash
export HF_HOME=/large_disk/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
uv run hf download physical-intelligence/libero --repo-type dataset
```

要求将 cache 放在大容量磁盘，不要继续挤占当前只剩几十 GiB 的文件系统。

备选方案：获取 raw/RLDS 数据，例如 `openvla/modified_libero_rlds`，再使用：

```text
examples/libero/convert_libero_data_to_lerobot.py
```

转换成 OpenPI 所需的 LeRobot 格式。

## 12. 恢复真实 LIBERO 实验后的执行顺序

### 12.1 前置条件

开始训练前必须确认：

```text
1. 可访问的 LIBERO LeRobot 数据路径或 Hugging Face cache
2. 至少约 150 GiB 的 checkpoint/cache 可用空间，越多越好
3. GPU 编号和可见卡列表
4. 一个真实数据 batch 能完成 transform、loss 和 sampling
```

### 12.2 只计算一次 π0 normalization stats

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_libero_e7_ocae_qo
```

预期生成：

```text
assets/pi0_libero_e7_ocae_qo/physical-intelligence/libero
```

其余五个 P9 配置读取同一份 stats。

### 12.3 先做短程真实数据 pilot

不要直接启动所有配置 30,000 steps。推荐先跑：

```text
E1 Static Q/O
E2 Constant Q/O
E7 Sample-conditioned Q/O
每个 500–1,000 steps，单 seed
```

验证：

```text
真实数据 loss finite 且有下降趋势
gradient sequence 正常
gate/delta ratio 不饱和
frozen fingerprint 不变
checkpoint 能恢复和采样
吞吐、显存和磁盘增长可接受
```

### 12.4 单 seed 核心比较

短程 pilot 通过后，先跑 seed 42：

```text
full fine-tune π0 baseline
E0 action LoRA
E1 static Q/O
E2 constant Q/O
E5 OCAE Q-only
E6 OCAE O-only
E7 OCAE Q/O
```

示例训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,3,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_e7_ocae_qo \
  --exp-name=seed42 \
  --checkpoint-base-dir=/large_disk/p9_checkpoints \
  --seed=42 \
  --overwrite
```

配置名依次替换为需要运行的实验。不要把 checkpoint 写到当前空间不足的目录。

### 12.5 多 seed

只有 E7 相对 E1/E2 显示合理趋势后，再补：

```text
seed 42
seed 43
seed 44
```

至少报告 3 个训练 seed 的均值和离散程度。checkpoint 应按固定 step 或独立 validation 指标选择，不能根据 test rollout 结果挑 checkpoint。

### 12.6 Serve 和 rollout

每个 checkpoint 必须用与训练完全相同的配置加载。例如 E7：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_e7_ocae_qo \
  --policy.dir=/large_disk/p9_checkpoints/pi0_libero_e7_ocae_qo/seed42/<step>
```

不要用 `pi0_libero` 配置加载 E7 checkpoint。

最终 benchmark：

```text
libero_spatial
libero_object
libero_goal
libero_10
final run: 50 trials/task
所有方法使用相同 rollout seeds
```

若 checkpoint 位于仓库外的大磁盘，Docker/compose 需要显式 mount 该路径。

## 13. 下一批消融的 Go / No-Go

在真实 rollout 前不要开始扩展性消融。建议只有观察到以下趋势才继续：

```text
E7 > E1
E7 > E2
E5 或 E6 至少一个显示稳定贡献
```

趋势成立后可考虑 contextual contribution 消融：

```text
E8: no visual context
E9: no language context
E10: no state context
E11: shuffled language
E12: shuffled state
E13: shuffled images/views
```

这些不属于当前已经完成的 P9 工程范围。

## 14. 当前最合理的下一步

1. 检查最终代码差异和测试结果，不要提交或还原用户的 `uv.lock`、`AGENTS.md` 和 `.libero/`。
2. 核验七组正式配置，确认只有 E2/E7 使用已验收的 gate L2 `3e-2`，其他实验变量保持不变。
3. 使用 seed 42 启动 30k staged training；先按资源情况安排作业，最多三张 GPU，不使用 `--overwrite`，每个配置使用独立 checkpoint 目录。
4. 训练期持续监控 task `loss`、`gate_regularization_loss`、`total_loss`、Q/O 饱和率、raw gate max、显存和错误日志。
5. 中间 checkpoint 必须执行 freeze、gate、effective delta 和 restore 诊断；出现 gate 超过阈值或 task loss 明显恶化时停止对应作业。
6. seed 42 训练和 checkpoint 验收完成后，使用固定 rollout seed、task order 和 initial states 做 staged rollout；趋势合理后再补 seed 43/44。

当前 `3e-2` 已是正式 E2/E7 配置，不再是未验收候选；但 1,000-step Go 不等于 30k 最终成功，长程训练仍需阶段性验收。

## 15. 常用检查命令

查看分支和工作树：

```bash
git status --short
git log --oneline --decorate -15
```

CPU 快速测试时显式禁用 CUDA JAX：

```bash
JAX_PLATFORMS=cpu uv run pytest <test_path> -q
```

GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 uv run pytest <test_path> -q
```

静态检查只针对本次修改文件：

```bash
uv run ruff check <changed_python_files>
uv run ruff format --check <changed_python_files>
git diff --check
```

不要对整个仓库做无关格式化，也不要处理 `uv.lock`。

## 16. 关键原始文档索引

总体设计：

```text
archive/overlock_pi0_ocaev1_contextualized_plan.md
archive/overlock_pi0_ocaev1_stepwise_implementation_validation_plan.md
archive/overlock_pi0_action_expert_new_plan.md
archive/overlock_pi0_v1_v2_explanation.md
```

阶段报告：

```text
archive/overlock_pi0_ocaev1_p0_baseline_report.md
archive/overlock_pi0_ocaev1_p1_split_forward_report.md
archive/overlock_pi0_ocaev1_p2_prefix_layout_report.md
archive/overlock_pi0_ocaev1_p3_conditioning_modules_report.md
archive/overlock_pi0_ocaev1_p4_conditional_query_report.md
archive/overlock_pi0_ocaev1_p5_conditional_output_report.md
archive/overlock_pi0_ocaev1_p6_checkpoint_freeze_report.md
archive/overlock_pi0_ocaev1_p7_end_to_end_report.md
archive/overlock_pi0_ocaev1_p8_smoke_training_report.md
archive/overlock_pi0_ocaev1_p9_core_experiment_pilot_report.md
```

## 17. 下次会话可直接使用的上下文提示

```text
请先读取 archive/overlock_pi0_ocaev1_session_handoff_2026-07-20.md，
再按其中“当前最合理的下一步”继续。

仓库是 /data1/gqy/workspace/openpi-overlockv1，分支 overlockv1。
P1-P9 已实现；真实 LIBERO E1/E2/E7 1,000-step pilot 及 final checkpoint restore/sampling 已完成。
不要修改或提交现有 uv.lock，也不要删除任何文件。
π0 base 参数位于 /data1/gqy/model/pi0_base/params。
下一步先补 gate、effective delta ratio 和冻结参数 fingerprint 诊断；
通过后再启动 seed 42 正式训练，不要把 pilot training loss 当成性能结论。
```
