# OpenPI OCAE V1 当前会话交接文档

> 更新时间：2026-07-26 UTC
> 工作目录：`/data1/gqy/workspace/openpi-overlockv1`
> 用途：新对话开始时先读取本文，即可快速恢复实验目标、实现状态、训练与测评结果、日志路径、当前判断和下一步计划。
> 当前总判断：OCAE V1 工程链路和 E1/E2/E7 30k checkpoint 验收通过，但真实 LIBERO 闭环执行能力很弱。E2 为 0%，E7 约 7%；逐步动作记录显示旋转动作相对训练 demonstration 严重放大。最高优先级不再是调 gate，而是验证 `extra_delta_transform=False` 和 full fine-tuning/标准 action LoRA 阳性对照。

## 0. 一分钟摘要

项目是在 JAX π0 action expert 上实现 Overview-Conditioned Action Expert（OCAE）：从 contextualized vision-language prefix 和机器人 state 生成逐层、逐 head 的 gate，控制 action expert Q/O 低秩分支。

已完成：

1. P0-P9 工程实现、测试、checkpoint merge/freeze、smoke training 和实验配置。
2. E1、E2、E7 的 seed 42、30k 单卡正式训练。
3. 三个 final checkpoint 的冻结、gate、effective delta、restore 和 sampling 验收。
4. 三模型完整 LIBERO benchmark 启动后，根据明显负面趋势 Early Stop。
5. E2/E7 四个标准 suite、每 task 只跑一次的快速全任务复测。
6. 保存 80 个视频和全部逐步 state/action trace。
7. 将 rollout 最终环境动作与 273,465 条训练动作做统计对比。

最终性能：

```text
完整 benchmark Early Stop:
E1: 2 / 105 = 1.90%
E2: 0 / 104 = 0%
E7: 7 / 103 = 6.80%

E2/E7 单次全任务复测:
E2: 0 / 40 = 0%
E7: 3 / 40 = 7.5%
```

所有失败 episode 均 `error=null`，属于策略失败，不是 simulator、WebSocket、EGL 或 OOM 链路失败。

最关键的新证据：

```text
训练 rotation action std: 0.039 / 0.063 / 0.078
E2 rollout rotation std:  1.883 / 1.499 / 1.657
E7 rollout rotation std:  1.395 / 1.082 / 1.180
```

E2/E7 rollout 旋转动作波动约为训练数据的 15-48 倍。任意维超出训练 q01-q99 区间的步骤比例为 E2 81.4%、E7 73.0%。

## 1. 仓库、分支和环境

```text
仓库: /data1/gqy/workspace/openpi-overlockv1
分支: overlockv1
HEAD: 4e77e31 feat(libero): record pilot training validation
远端: git@github.com:gahsf/openpi-fork.git
远端分支: origin/overlockv1
远端当前也在: 4e77e31
上游 main 基点: 15a9616
```

本地模型：

```text
π0 base:       /data1/gqy/model/pi0_base/params
π0.5 base:     /data1/gqy/model/pi05_base/params
π0.5 LIBERO:   /data1/gqy/model/pi05_libero
```

数据集：

```text
/data1/gqy/data/huggingface/lerobot/physical-intelligence/libero
```

数据统计：

```text
parquet 文件: 1,693
动作帧:       273,465
state:        8D
actions:      7D
```

GPU：8 张 RTX 5090，每张约 32 GiB。当前没有本项目的训练、LIBERO client 或 policy server 在运行。本轮 E2/E7 复测服务已经停止；GPU 2 已释放，GPU 5 原有的约 13.7 GiB 其他占用不属于本轮进程。

## 2. 实验目标和研究问题

OCAE V1 的目标是：

1. 冻结 VLM 和原 action expert 主干。
2. 用 contextualized prefix hidden states 表示视觉—语言任务上下文。
3. 编码机器人连续 state。
4. 由 controller 生成逐层、逐 head 的 Query/Output gate。
5. 只用条件低秩分支修改 action expert Q/O。
6. 判断 sample-conditioned context 是否优于 static LoRA 和 constant-conditioned matched control。

核心关系：

```text
prefix -> frozen VLM -> contextualized prefix + KV cache
contextualized prefix + state -> OCAE controller -> Q/O gates
suffix action expert + cached prefix KV + Q/O gates -> flow action prediction
```

研究对照：

| 编号 | 配置 | 目的 |
|---|---|---|
| E0 | `pi0_libero_e0_action_lora` | 标准 action-expert LoRA baseline |
| E1 | `pi0_libero_e1_static_qo` | Static Q/O LoRA matched baseline |
| E2 | `pi0_libero_e2_constant_qo` | 同参数量 constant-input OCAE |
| E5 | `pi0_libero_e5_ocae_q` | Sample-conditioned Q-only |
| E6 | `pi0_libero_e6_ocae_o` | Sample-conditioned O-only |
| E7 | `pi0_libero_e7_ocae_qo` | Sample-conditioned Q/O 主方法 |

本轮只正式训练了 E1、E2、E7。

重要：E1/E2/E7 都不是 fully fine-tuning。

| 实验 | 可训练参数 | 占 π0 约 32.38 亿参数 |
|---|---:|---:|
| E1 | 5,898,240 | 约 0.18% |
| E2 | 18,823,456 | 约 0.58% |
| E7 | 18,823,456 | 约 0.58% |

E2/E7 新增参数中相当一部分属于 overview/state encoder 和 controller；真正修改 action expert 的位置仍只有每层 attention Q/O rank-16 低秩分支。K/V、FFN、state/action input/output projection、VLM 主干和原 action expert 都冻结。

## 3. 工程实现进度

P0-P9 的工程范围已经完成：

```text
P0 基线和方案记录
P1 训练前向拆分为 prefix cache + suffix
P2 Prefix layout 静态结构
P3 Overview/State encoder 和 controller
P4 Conditional Query LoRA
P5 Conditional Output LoRA
P6 Base checkpoint merge 和 OCAE-only freeze
P7 端到端 JAX/checkpoint 回归
P8 Smoke training
P9 matched baselines、真实数据 pilot 和正式配置
```

已推送提交：

| 阶段 | Commit |
|---|---|
| P0/计划 | `012348d` |
| P1 | `c10a420` |
| P2 | `a384df9` |
| P3 | `0058a8e` |
| P4 | `52dbbd6` |
| P5 | `b712987` |
| P6 | `a237ab5` |
| P7 | `cbd37d4` |
| P8 | `427adab` |
| P9 matched configs | `9d0ae87` |
| LIBERO pilot 记录 | `4e77e31` |

主要实现文件：

```text
src/openpi/models/pi0.py
src/openpi/models/pi0_config.py
src/openpi/models/gemma.py
src/openpi/models/model.py
src/openpi/models/overview_action_conditioning.py
src/openpi/training/config.py
src/openpi/training/weight_loaders.py
scripts/train.py
```

## 4. Gate 饱和问题及修复经过

最初 1,000-step E2/E7 pilot 出现 controller gate 饱和：Q/O 的 `|g|>0.9` 达到 100%。冻结、checkpoint restore 和 sampling 正常，但 sample conditioning 被 `tanh` 饱和抹平，因此最初判定 No-Go。

尝试一：`gate_logit_scale=0.25`

- 500 steps 时 gate 正常。
- 延长到约 750-770 steps 后再次快速饱和。
- 结论：只能延迟，不能解决。

尝试二：raw gate L2 regularization

```text
total_loss = task_loss + gate_regularization_loss
日志中的 loss 仍表示 task loss
```

结果：

```text
1e-3: 不足
1e-2: 500 step 通过，但 999 step 严格标准失败
3e-2: E2/E7 999 step 均通过
```

`3e-2` 最终 pilot：

```text
E7 step 999: Q/O saturation 0.6944% / 0%; delta 0.443% / 0.705%
E2 step 999: Q/O saturation 0.6944% / 0%; delta 0.518% / 0.852%
freeze: 两组均 50/50 exact equal
E7 sample distance: 非零
E2 sample distance: 零，符合 constant 定义
```

因此正式 E2/E7 配置使用 `gate_l2_regularization=3e-2`。

当前判断：gate 饱和已经被控制，但最终 rollout 仍很差，所以 gate 不是当前主要矛盾。

## 5. 正式训练进度和结果

公共训练协议：

```text
dataset: physical-intelligence/libero
seed: 42
steps: 30,000
global batch: 32
optimizer: AdamW
warmup: 1,000
peak lr: 1e-5
decay lr: 1e-6
EMA: disabled
extra_delta_transform: true
initializer: /data1/gqy/model/pi0_base/params
```

### 5.1 多卡结论

尝试过两卡：

- 两卡 FSDP 在 RTX 5090 上出现 CUDA illegal address。
- 两卡纯数据并行在完整训练图的首次 collective/JIT 调用停止推进。
- `NCCL_CUMEM_HOST_ENABLE=0` 只让独立 `psum` 探针通过，没有解决完整训练图。

所以 E1/E2/E7 最终统一使用单卡训练，并通过三张不同 GPU 并行跑三个独立实验。不要直接重复之前失败的两卡方案。

### 5.2 30k checkpoint

E1：

```text
checkpoint:
/data1/gqy/checkpoints/ocaev1/pi0_libero_e1_static_qo/seed42_30k_20260723/29999

log:
/data1/gqy/logs/ocaev1/e1_static_qo_seed42_30k_20260723.log
```

E7：

```text
checkpoint:
/data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo/seed42_30k_20260723/29999

log:
/data1/gqy/logs/ocaev1/e7_sample_qo_gate_l2_3e2_seed42_30k_20260723.log

step 29900 Q/O |g|>0.9: 0% / 0%
```

E2：

```text
原训练在约 step 3990 遇到 CUDA_ERROR_LAUNCH_FAILED。
原目录保留，没有删除或覆盖。
从原 step 3000 checkpoint 复制到新实验目录后 resume。

final checkpoint:
/data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo/seed42_30k_resume_from3000_20260724/29999

原日志:
/data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_20260723.log

续训日志:
/data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_resume_from3000_20260724.log
```

三组 30k 训练最终都完成。loss 下降、梯度和参数 finite；final checkpoint 的冻结参数、effective delta、gate、restore 和真实 observation sampling 验收通过。

这只能证明“训练和 checkpoint 工程正常”，不能证明策略会完成任务。

## 6. 第一次完整 LIBERO benchmark：已 Early Stop

原计划：

```text
suites: spatial / object / goal / libero_10
50 trials per task
seed: 7
三模型并行
总计计划 6,000 episodes
```

结果目录：

```text
/data1/gqy/evals/ocaev1/full_benchmark_seed42_20260725
```

日志目录：

```text
/data1/gqy/logs/ocaev1/libero_full_benchmark_seed42_20260725
```

停止时结果：

| 模型 | Episodes | Success | 成功率 | 程序错误 |
|---|---:|---:|---:|---:|
| E1 | 105 | 2 | 1.90% | 0 |
| E2 | 104 | 0 | 0% | 0 |
| E7 | 103 | 7 | 6.80% | 0 |

其中完整跑完的 Spatial task 0 和 task 1：

```text
E1: 2 / 100
E2: 0 / 100
E7: 7 / 100
```

负面趋势已经非常明显，用户要求停止，避免继续消耗约四天 GPU。所有 client、orchestrator 和 server 已正常停止，结果与日志全部保留。

## 7. E2/E7 单次全任务动作轨迹复测

为了覆盖全部四个 suite 并分析失败动作，又用两张卡执行了快速复测。

协议：

```text
模型: E2 final、E7 final
suites: libero_spatial / libero_object / libero_goal / libero_10
每 suite: 10 tasks
每 task: 1 episode
每模型: 40 episodes
seed: 7
resize: 224
replan_steps: 5
num_steps_wait: 10
save_videos: true
```

结果根目录：

```text
/data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725
```

日志根目录：

```text
/data1/gqy/logs/ocaev1/action_trace_smoke_seed42_20260725
```

目录结构：

```text
<model>/<suite>/results.jsonl
<model>/<suite>/videos/*.mp4
<model>/<suite>/traces/action_trace.jsonl
```

最终成功率：

| 模型 | Spatial | Object | Goal | LIBERO-10 | 总计 |
|---|---:|---:|---:|---:|---:|
| E2 constant Q/O | 0/10 | 0/10 | 0/10 | 0/10 | 0/40（0%） |
| E7 sample Q/O | 0/10 | 1/10 | 2/10 | 0/10 | 3/40（7.5%） |

E7 成功任务：

```text
Object task 8: pick up the chocolate pudding and place it in the basket, 162 steps
Goal task 7: turn on the stove, 226 steps
Goal task 8: put the bowl on the plate, 91 steps
```

完整性：

```text
80 / 80 episode results 完整
80 / 80 MP4 完整
80 / 80 episode_end trace 完整
所有 error=null
E2 action steps: 13,200，全部跑到最大步数
E7 action steps: 12,772，3 个成功 episode 提前结束
```

## 8. Action trace 记录内容

`examples/libero/main.py` 新增：

```text
--args.action-trace-out-path
```

每个控制步记录：

```json
{
  "record_type": "action",
  "task_suite": "libero_spatial",
  "task_id": 0,
  "episode_idx": 0,
  "seed": 7,
  "t": 10,
  "state": ["8D policy input state"],
  "action": ["7D final env.step action"],
  "replanned": true,
  "planned_actions": [["5 x 7 actions, only on replanning steps"]]
}
```

episode 结束记录 `episode_end`、success、steps 和 error。

关键语义：记录的 `action` 已经过 policy output transform，即 `Unnormalize + AbsoluteActions`，是最终实际传入 `env.step` 的 raw LIBERO 7D action。因此可以直接与训练 parquet 的 `actions` 对比。

action trace 文件拒绝覆盖已有路径。

验证：

```text
Ruff passed
Python py_compile passed
LIBERO Python 3.8 CLI help passed
真实 80 episodes 写入通过
```

## 9. Rollout 动作与训练数据对比

分析脚本：

```text
scripts/compare_libero_rollout_actions.py
```

完整输出：

```text
/data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725/action_comparison.json
```

训练动作统计：

```text
count: 273,465
rotation std: 0.039 / 0.063 / 0.078
rotation q01: -0.115 / -0.164 / -0.224
rotation q99:  0.132 /  0.193 /  0.335
gripper positive fraction: 47.5%
```

Rollout：

| 数据 | Rotation std 3 | Rotation std 4 | Rotation std 5 | 任意维越出训练 q01-q99 |
|---|---:|---:|---:|---:|
| E2 | 1.883 | 1.499 | 1.657 | 81.4% |
| E7 | 1.395 | 1.082 | 1.180 | 73.0% |
| E7 成功 episodes | 0.605 | 0.769 | 0.753 | 59.7% |

逐维 rotation 越界比例：

```text
E2: 52.0% / 58.3% / 42.0%
E7: 36.8% / 40.7% / 28.6%
```

平移前三维越界约 1%-5%，异常明显集中在旋转和 gripper，不是全部维度同时发生统一尺度错误。

## 10. 当前根因判断

### 10.1 没有发现确定的 observation/rollout 硬错

已核对：

- 图像旋转和 resize 与官方 LIBERO 路径一致。
- state 是 position 3 + axis-angle 3 + gripper 2，共 8 维。
- prompt 与数据集 task 文本一致。
- server 加载正确 config、checkpoint 和 checkpoint norm stats。
- seed、initial states、max steps、replan 5 均符合协议。
- `Unnormalize` 和 `AbsoluteActions` 与训练变换数学对称。
- policy 输出没有维度错误，simulator/API 链路无异常。

所以没有证据支持“漏反归一化”或“评测代码把动作维度传错”这种简单硬错误。

### 10.2 最大嫌疑：动作表示增加了窄 PEFT 的学习难度

LIBERO parquet 的 raw action 本身已经是 delta action，但当前训练配置启用：

```text
extra_delta_transform=True
```

训练时：

```text
target = raw LIBERO action - current state
```

推理时：

```text
raw action = predicted target + current state
```

两步数学上互逆，因此不是确定的 train/eval mismatch；但 transformed target 被绝对末端姿态主导。此前抽样发现：

```text
raw action mean absolute:         约 0.184
transformed target mean absolute: 约 0.883
旋转 transformed target 与 -state 相关性: 约 0.955-0.991
```

当前 Q/O-only PEFT 必须非常精确地预测并抵消绝对姿态。小的 transformed-space 误差经过加回 state 后，会变成很大的 raw rotation action。本次 trace 中 15-48 倍 rotation std 是这一风险的直接表现。

### 10.3 第二嫌疑：PEFT 容量和插入位置过窄

训练 loss 明显下降，effective delta 非零，说明 adapter 不是完全没学到；但它只能改 action expert attention Q/O rank-16：

```text
K/V frozen
FFN frozen
state_proj frozen
action_in_proj frozen
action_out_proj frozen
VLM frozen
```

这可能足以拟合 flow-matching 训练 loss，却不足以获得可靠的视觉定位、动作动力学和闭环纠错能力。

### 10.4 Gate 不是共同主因

E2/E7 final gate 已通过饱和验收，而 rollout 仍差；无动态 gate 的 E1 也只有约 2%。因此不应继续把主要资源投入 gate 超参数。

## 11. 当前代码与 Git 状态

当前 HEAD 和远端同步在 `4e77e31`。2026-07-26 的 gate L2、30k/benchmark 记录、action trace 和动作分析工作仍未提交。

当前 tracked modifications：

```text
M AGENTS.md
M archive/overlock_pi0_ocaev1_libero_training_evaluation_plan_2026-07-20.md
M archive/overlock_pi0_ocaev1_session_handoff_2026-07-20.md
M examples/libero/main.py
M scripts/train.py
M src/openpi/models/model.py
M src/openpi/models/model_test.py
M src/openpi/models/overview_action_conditioning.py
M src/openpi/models/overview_action_conditioning_test.py
M src/openpi/models/pi0.py
M src/openpi/training/config.py
M uv.lock
```

当前 untracked：

```text
archive/overlock_pi0_ocaev1_e2_e7_action_trace_eval_report_2026-07-26.md
archive/overlock_pi0_ocaev1_session_handoff_2026-07-26.md
examples/libero/.libero/
scripts/compare_libero_rollout_actions.py
scripts/diagnose_ocaev1_pilot.py
```

必须保留并谨慎处理：

- `AGENTS.md`：用户/环境规则文件，不要擅自还原或提交。
- `uv.lock`：用户本地镜像源变化，不要还原、不要纳入项目提交。
- `examples/libero/.libero/`：包含本机绝对路径，不要提交。
- 工作树中其他改动属于连续实验实现，不要用 reset/checkout 覆盖。

提交前应逐文件审计并只 stage 本轮需要的代码、测试和文档，不能使用无选择的 `git add -A`。

## 12. 关键日志与记录索引

### 12.1 总体方案和交接

```text
archive/overlock_pi0_ocaev1_contextualized_plan.md
archive/overlock_pi0_ocaev1_stepwise_implementation_validation_plan.md
archive/overlock_pi0_ocaev1_libero_training_evaluation_plan_2026-07-20.md
archive/overlock_pi0_ocaev1_session_handoff_2026-07-20.md
archive/overlock_pi0_ocaev1_session_handoff_2026-07-26.md
```

### 12.2 本次动作复测报告

```text
archive/overlock_pi0_ocaev1_e2_e7_action_trace_eval_report_2026-07-26.md
```

### 12.3 Pilot 诊断

```text
scripts/diagnose_ocaev1_pilot.py
/data1/gqy/evals/ocaev1/pilot_diagnostics_20260721
```

### 12.4 30k checkpoint 和日志

```text
E1 checkpoint:
/data1/gqy/checkpoints/ocaev1/pi0_libero_e1_static_qo/seed42_30k_20260723/29999
E1 log:
/data1/gqy/logs/ocaev1/e1_static_qo_seed42_30k_20260723.log

E2 checkpoint:
/data1/gqy/checkpoints/ocaev1/pi0_libero_e2_constant_qo/seed42_30k_resume_from3000_20260724/29999
E2 original log:
/data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_20260723.log
E2 resume log:
/data1/gqy/logs/ocaev1/e2_constant_qo_gate_l2_3e2_seed42_30k_resume_from3000_20260724.log

E7 checkpoint:
/data1/gqy/checkpoints/ocaev1/pi0_libero_e7_ocae_qo/seed42_30k_20260723/29999
E7 log:
/data1/gqy/logs/ocaev1/e7_sample_qo_gate_l2_3e2_seed42_30k_20260723.log
```

### 12.5 Benchmark

```text
完整 benchmark Early Stop results:
/data1/gqy/evals/ocaev1/full_benchmark_seed42_20260725

完整 benchmark logs:
/data1/gqy/logs/ocaev1/libero_full_benchmark_seed42_20260725

E2/E7 单次全任务 results、videos、traces:
/data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725

E2/E7 单次全任务 logs:
/data1/gqy/logs/ocaev1/action_trace_smoke_seed42_20260725

动作统计 JSON:
/data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725/action_comparison.json
```

## 13. 已验证的命令和分析复现

动作分布对比：

```bash
uv run python scripts/compare_libero_rollout_actions.py \
  --train-glob '/data1/gqy/data/huggingface/lerobot/physical-intelligence/libero/data/**/*.parquet' \
  --eval-root /data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725 \
  --output /data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725/action_comparison.json
```

相关校验：

```text
scripts/compare_libero_rollout_actions.py: Ruff passed, py_compile passed
examples/libero/main.py action trace: Ruff passed, Python 3.8 py_compile passed
所有相关文件: git diff --check passed
```

## 14. 下一步建议：按判别力排序

### A. 先保存和提交当前工作

在继续训练前，先审计未提交 diff，把 gate L2 实现、测试、训练/评测记录、action trace 和比较脚本做成清晰提交并推送。不要包含 `AGENTS.md`、`uv.lock`、`.libero/`。

### B. 做 `extra_delta_transform=False` 最小 A/B

这是当前最高优先级实验：

1. 新增独立配置，不修改已有 E1/E2/E7 配置语义。
2. 关闭 `extra_delta_transform`。
3. 重新计算独立 normalization stats，不能复用现有 transformed stats。
4. 先跑小规模训练和少量固定 observation 推理。
5. 保存 raw rollout action，确认 rotation 分布是否回到 demonstration 范围。
6. 再做每 task 一次的四-suite 快测，不直接启动 50 trials/task。

判据：rotation std、q01/q99 和越界率应明显改善；若动作分布改善但成功率仍低，再重点看模型容量和视觉定位。

### C. 同时补阳性对照

至少选择一个，最好都做：

```text
F0: 官方 pi0_libero fully fine-tuning
E0: 标准 action-expert LoRA，rank-32，attention + FFN
```

解释方式：

| 结果 | 判断 |
|---|---|
| F0/E0 成功，OCAE 失败 | OCAE 容量或插入位置不足 |
| F0 成功，E0 失败 | 当前任务可能需要更广泛 full fine-tuning |
| F0 也失败 | 优先怀疑动作表示、环境版本或整个训练/评测链路 |
| no-delta 明显改善 | `extra_delta_transform` 是主要放大因素 |

### D. 暂时不要做

- 不要立刻恢复 50 trials/task 全量 benchmark。
- 不要先补 seed 43/44。
- 不要继续优先调 gate L2 或 gate scale。
- 不要直接启动 E5/E6 长训练。
- 不要再次使用已失败的两卡 FSDP/DP 路径，除非先单独解决多卡稳定性。

## 15. 新对话可直接使用的提示

```text
请先完整读取：
archive/overlock_pi0_ocaev1_session_handoff_2026-07-26.md

仓库：/data1/gqy/workspace/openpi-overlockv1
分支：overlockv1
HEAD/origin：4e77e31

OCAE P0-P9、E1/E2/E7 seed42 30k 和 checkpoint 验收已完成。
真实 LIBERO 成功率很低：E1≈2%、E2=0%、E7≈7%。
E2/E7 四 suite 每 task 一次的复测已完成，80 个视频和逐步 action trace 已保存。
动作对比显示 rollout rotation std 比训练数据高 15-48 倍。

当前优先工作：
1. 先审计并提交尚未提交的 gate L2、action trace、分析脚本和文档；不要提交 AGENTS.md、uv.lock、examples/libero/.libero/。
2. 新增 extra_delta_transform=False 的独立配置，重新计算 norm stats，做短程 A/B。
3. 补官方 full fine-tuning 或标准 action-expert LoRA 阳性对照。

禁止删除 checkpoint、日志和结果；不要覆盖用户已有改动。
```
