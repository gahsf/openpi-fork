# OCAE V1 P9 核心实验准备与 Pilot 报告

## 1. 当前结论

P9 的工程准备与全尺寸模型 pilot 已完成，但研究验收尚未完成。

已完成：

```text
E0/E1/E2/E5/E6/E7 Libero 配置
E1 Static Q/O LoRA matched baseline
E2 Constant-input OCAE matched baseline
六配置参数量和 freeze 审计
六配置本地 pi0_base 全尺寸两步训练 pilot
六配置 finite action sampling
P1-P9 聚焦回归和默认 pi0 GPU 回归
```

尚未完成：

```text
真实 Libero 数据训练
每配置至少 3 个 seed
Libero spatial/object/goal/10 rollout
E7 > E1 和 E7 > E2 的研究判断
```

不能用本报告的 synthetic loss 排名替代 Libero benchmark 结果。

P8 前置提交：

```text
427adab chore(ocae): add smoke training configuration
```

## 2. 新增实验配置

| 编号 | 配置名 | 方法 |
|---|---|---|
| E0 | `pi0_libero_e0_action_lora` | 原始 action-expert LoRA |
| E1 | `pi0_libero_e1_static_qo` | Static Q/O LoRA |
| E2 | `pi0_libero_e2_constant_qo` | Constant-input OCAE Q/O |
| E5 | `pi0_libero_e5_ocae_q` | Sample-conditioned OCAE Q-only |
| E6 | `pi0_libero_e6_ocae_o` | Sample-conditioned OCAE O-only |
| E7 | `pi0_libero_e7_ocae_qo` | Sample-conditioned OCAE Q/O |

所有配置共同使用：

```text
dataset: physical-intelligence/libero
prompt_from_task: true
extra_delta_transform: true
initial checkpoint: pi0_base
batch size: 32
steps: 30,000
optimizer: AdamW
warmup: 1,000
peak lr: 1e-5
decay lr: 1e-6
EMA: disabled
seed default: 42
```

六配置共享由 E7 生成的 normalization assets：

```text
./assets/pi0_libero_e7_ocae_qo/physical-intelligence/libero
```

因此恢复数据连接后只需计算一次 normalization stats。

## 3. Matched Baseline 实现

### 3.1 E1 Static Q/O LoRA

E1 使用与 E7 相同的 action-expert Q/O 位置和 `rank=16, alpha=16`，但不创建 overview/state/controller。

```text
gate: fixed one
LoRA A: random initialization
LoRA B: zero initialization
```

这样初始化时 `A @ B = 0`，保证输出与 base 模型一致；第一步只有 B 获得梯度，B 更新后 A 从第二步开始获得梯度。这是普通 static LoRA 的标准 identity 初始化，而不是把 zero gate 固定为零导致无法学习。

### 3.2 E2 Constant-input OCAE

E2 与 E7 保持完全相同的参数树：

```text
overview encoder
state encoder
controller
conditional Q LoRA
conditional O LoRA
```

区别仅为送入 OCAE 的 reference input：

```text
prefix reference: deterministic nonzero sinusoidal tensor
image masks: fixed valid
language mask: fixed valid
state reference: deterministic nonzero vector
```

所有 batch sample 收到相同 reference，但 reference 仍经过 overview/state encoder 和 controller，因此这些参数继续参与反向传播。E2 没有直接绕过 encoder 塞常量 context。

E2 仍执行正常 sample-dependent base prefix KV forward；只有新增 OCAE conditioning signal 被固定。这保证它控制的是 sample-conditioned controller 的贡献，而不是破坏原 π0 的视觉语言输入。

## 4. 失败先行测试

生产代码修改前，新增测试得到 10 个预期失败：

```text
constant reference helper 不存在
Static Q/O zero-init B 不存在
static 参数树模式不存在
六个 Libero 实验配置不存在
```

实现后同一组测试：

```text
10 passed
12 warnings
elapsed: 15.30 s
```

覆盖：

```text
constant reference 与 sample 内容无关且非零
Static Query/Output LoRA 初始化输出为零
Static LoRA 第一步 B 梯度非零、A 梯度为零
E1 仅 conditional Q/O 参数可训练
六配置的数据、模型、freeze、训练预算一致
```

## 5. 参数量审计

| 实验 | 总参数 | 可训练参数 | 可训练比例 |
|---|---:|---:|---:|
| E0 action LoRA | 3,260,166,928 | 22,118,400 | 0.67844% |
| E1 Static Q/O | 3,243,946,768 | 5,898,240 | 0.18182% |
| E2 Constant Q/O | 3,256,871,984 | 18,823,456 | 0.57796% |
| E5 OCAE Q | 3,253,922,864 | 15,874,336 | 0.48785% |
| E6 OCAE O | 3,253,922,864 | 15,874,336 | 0.48785% |
| E7 OCAE Q/O | 3,256,871,984 | 18,823,456 | 0.57796% |

关键一致性：

```text
E2 total/trainable params == E7 total/trainable params
E1 只含 conditional Q/O LoRA
E0 只训练 action-expert 标准 LoRA
E5 只有 conditional Q
E6 只有 conditional O
```

## 6. Libero 数据与环境审计

已存在：

```text
third_party/libero submodule
/data1/gqy/model/pi05_libero checkpoint
/data1/gqy/model/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
```

缺失：

```text
physical-intelligence/libero LeRobot dataset cache
π0 extra_delta_transform=True 对应的 normalization stats
```

连接验证：

```text
LeRobotDatasetMetadata request: 120 s timeout
curl Hugging Face dataset API: 20 s timeout
local physical-intelligence/libero dataset search: not found
```

不能直接复用 `pi05_libero` stats：π0 核心配置使用 `extra_delta_transform=True`，而 `pi05_libero` 使用 `extra_delta_transform=False`，动作统计定义不同。

## 7. 存储审计

P9 开始时：

```text
filesystem available: approximately 35 GiB
P8 retained checkpoint 1: approximately 4.8 GiB
P8 retained checkpoint 2: approximately 4.8 GiB
```

单个完整 OCAE 参数 checkpoint 约 4.8 GiB。仅六配置单 seed 就至少需要约 29 GiB，尚未计入 optimizer state、assets、日志和多个 seed。因此当前磁盘不适合启动 P9 全量训练。

本阶段六配置 pilot 没有写 checkpoint。

## 8. GPU 0 全尺寸 Pilot

统一条件：

```text
physical GPU: 0
checkpoint: /data1/gqy/model/pi0_base/params
batch size: 1
steps: 2
synthetic fixed-shape input
pilot warmup: 5
pilot peak/decay lr: 1e-5 / 1e-6
sample flow steps: 2
checkpoint saving: disabled
```

不同实验使用不同 RNG，因此 loss 数值不能横向排名；这里只验证各配置的真实 loader、freeze、梯度和推理链路。

### 8.1 E0 Action-expert LoRA

```text
init: 93.25 s
step 1 loss/global LoRA grad: 1.29139 / 0.97068
step 2 loss/global LoRA grad: 1.27619 / 0.92686
step 2 steady time: 0.04996 s
actions: (1, 50, 32), finite
```

### 8.2 E1 Static Q/O LoRA

```text
init: 30.71 s
step 1 Q-A/O-A grad: 0 / 0
step 1 Q-B/O-B grad: 0.63751 / 2.19840
step 2 Q-A/O-A grad: 4.258e-4 / 2.980e-3
step 2 Q-B/O-B grad: 0.22346 / 0.75622
step 2 steady time: 0.04368 s
actions: (1, 50, 32), finite
```

E1 梯度顺序符合 static LoRA identity 初始化预期。

### 8.3 E2 Constant-input OCAE Q/O

```text
step 1 controller grad: 3.696e-2
step 1 encoder/Q/O LoRA grad: 0
step 2 overview/state grad: 1.497e-4 / 5.448e-5
step 2 Q-A/Q-B grad: 2.405e-4 / 2.214e-4
step 2 O-A/O-B grad: 7.663e-4 / 7.616e-4
step 2 steady time: 0.04823 s
actions: (1, 50, 32), finite
```

E2 完整参数树均能从 constant reference 获得梯度。

### 8.4 E5 OCAE Q-only

```text
step 1 controller grad: 6.177e-2
step 2 overview/state grad: 2.598e-7 / 8.098e-8
step 2 Q-A/Q-B grad: 3.068e-5 / 2.974e-5
O-A/O-B grad: 0 / 0
step 2 steady time: 0.04478 s
actions: (1, 50, 32), finite
```

### 8.5 E6 OCAE O-only

```text
step 1 controller grad: 2.640e-2
step 2 overview/state grad: 1.247e-5 / 3.471e-6
step 2 O-A/O-B grad: 7.861e-5 / 8.013e-5
Q-A/Q-B grad: 0 / 0
step 2 steady time: 0.04639 s
actions: (1, 50, 32), finite
```

### 8.6 E7 OCAE Q/O

```text
step 1 controller grad: 2.892e-2
step 2 overview/state grad: 4.955e-5 / 1.574e-5
step 2 Q-A/Q-B grad: 4.561e-5 / 4.392e-5
step 2 O-A/O-B grad: 1.468e-4 / 1.476e-4
step 2 steady time: 0.04636 s
actions: (1, 50, 32), finite
```

六配置同进程观察到的 JAX peak memory high-water mark 为约 6.792 GiB。该值用于确认 pilot 无 OOM，不作为各实验独立峰值比较。

## 9. 回归验证

P1-P9 CPU 聚焦回归：

```text
39 passed
1,255 warnings
elapsed: 183.02 s
```

OCAE-disabled 默认 π0 GPU 回归：

```text
1 passed
322 warnings
elapsed: 103.49 s
```

最终配置契约复查：

```text
6 passed
elapsed: 8.23 s
```

静态检查：

```text
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

warnings 来自现有 JAX/Flax deprecated API。

## 10. 数据恢复后的执行顺序

### 10.1 只计算一次 normalization stats

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_libero_e7_ocae_qo
```

生成目录：

```text
assets/pi0_libero_e7_ocae_qo/physical-intelligence/libero
```

其余五个 P9 配置会读取同一目录。

### 10.2 先跑单 seed 趋势

建议使用有足够空间的 checkpoint 根目录，例如 `/large_disk/p9_checkpoints`：

```bash
CUDA_VISIBLE_DEVICES=0,1,3,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_e0_action_lora \
  --exp-name=seed42 --checkpoint-base-dir=/large_disk/p9_checkpoints --overwrite
```

然后依次替换配置名：

```text
pi0_libero_e1_static_qo
pi0_libero_e2_constant_qo
pi0_libero_e5_ocae_q
pi0_libero_e6_ocae_o
pi0_libero_e7_ocae_qo
```

### 10.3 趋势成立后补 3 seed

```text
seed 42
seed 43
seed 44
```

不要根据 Libero test rollout 挑 checkpoint；应使用固定 step 或独立 validation loss 选择。

### 10.4 Rollout

每个训练 checkpoint 必须用对应配置名加载，例如 E7：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_e7_ocae_qo \
  --policy.dir=/large_disk/p9_checkpoints/pi0_libero_e7_ocae_qo/seed42/<step>
```

Docker client 对每个 checkpoint 依次运行：

```text
libero_spatial
libero_object
libero_goal
libero_10
50 trials per task
相同 rollout seed
```

## 11. P9 Go / No-Go

只有真实 rollout 满足以下趋势才能继续第二批消融：

```text
E7 > E1
E7 > E2
E5/E6 至少一个显示稳定正向贡献
多个训练 seed 趋势一致
```

当前状态：

```text
工程 Go: true
全尺寸 synthetic pilot Go: true
Libero research Go/No-Go: 未判定
阻塞 1: physical-intelligence/libero 数据不可达
阻塞 2: π0 Libero norm stats 缺失
阻塞 3: 当前 35 GiB 空间不足以保存多配置多 seed checkpoint
```

因此当前不能声称 P9 或论文核心结论已经验证通过。
