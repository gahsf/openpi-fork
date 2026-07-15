# OCAE V1 P7 端到端 JAX 验证报告

## 1. 结论

P7 通过：

```text
新增独立 pi0_ocaev1_debug 训练配置
本地 base pi0 checkpoint 可通过真实训练初始化路径加载
完整 JIT loss、反向传播和参数更新成功
固定 noise 的推理可重复
Orbax 参数 checkpoint 保存/恢复前后输出逐位一致
真实 OCAE 训练状态可通过默认 BaseModelConfig.load() 重建
OCAE disabled 的默认完整 pi0 GPU 回归通过
额外推理延迟和显存已测量
```

P6 前置提交：

```text
a237ab5 feat(ocae): support base checkpoints and ocae-only freezing
```

## 2. 本阶段修改

```text
src/openpi/training/config.py
src/openpi/models/model.py
src/openpi/models/model_test.py
```

### 2.1 训练配置

新增：

```text
pi0_ocaev1_debug
```

关键配置：

```text
model: pi0
pi05: false
paligemma_variant: gemma_2b
action_expert_variant: gemma_300m
OCAE enabled: true
target: q_o
rank: 16
lora_alpha: 16.0
freeze: OCAE-only
weight loader: gs://openpi-assets/checkpoints/pi0_base/params
batch size: 1
EMA: disabled
```

正式配置保留可移植的 GCS base checkpoint 地址；本机 GPU 验证通过 `dataclasses.replace` 仅把 loader 地址替换为：

```text
/data1/gqy/model/pi0_base/params
```

### 2.2 NNX no-bias 参数恢复兼容

P6 诊断发现，no-bias `nnx.LayerNorm` 在期望状态树中包含：

```text
bias: None
```

Orbax `intersect_trees()` 会删除该静态叶子，导致默认 `BaseModelConfig.load()` 在结构验证时失败。

P7 的最小修复为：

1. 继续使用 `intersect_trees()` 删除 checkpoint 多余参数。
2. 只从模型期望树补回值为 `None` 的静态叶子。
3. 继续执行完整结构和 shape 验证。

该修复不允许缺失的数组参数绕过验证，也不改变 checkpoint 中已有数组值。

## 3. 失败先行验证

生产代码修改前，新测试得到两个预期失败：

```text
pi0_ocaev1_debug 不存在
checkpoint 恢复后 source_norm 缺失 bias: None
```

第二个测试在失败前已经成功完成：

```text
dummy OCAE model 创建
JIT 单步 loss/gradient/update
gradient finite/nonzero 检查
固定 noise 采样
Orbax checkpoint 写入
```

说明失败点被精确定位在恢复树结构，而不是训练、推理或 checkpoint 写入。

修复后聚焦结果：

```text
2 passed
509 warnings
elapsed: 81.15 s
```

## 4. 新增端到端自动化测试

`test_pi0_ocaev1_train_sample_and_checkpoint_roundtrip` 使用 dummy π0 降低持续集成成本，但覆盖真实 JAX/NNX/Orbax 接口：

1. 创建 `q_o/rank=2` OCAE 模型。
2. JIT 执行一次只针对 OCAE 参数的梯度更新。
3. 验证 loss、所有梯度 finite，且至少一个梯度非零。
4. 使用固定 noise 采样。
5. 通过 `PyTreeCheckpointer` 保存参数 checkpoint。
6. 通过仓库 `restore_params()` 和默认 `config.load()` 恢复。
7. 使用相同 noise 再次采样。
8. 恢复前后 actions 逐位相等。

## 5. P1-P7 CPU 聚焦回归

命令：

```bash
JAX_PLATFORMS=cpu uv run pytest \
  src/openpi/training/weight_loaders_test.py \
  src/openpi/models/pi0_test.py \
  src/openpi/models/overview_action_conditioning_test.py \
  src/openpi/models/conditional_action_lora_test.py \
  src/openpi/models/pi0_split_forward_test.py \
  src/openpi/models/model_test.py::test_pi0_ocaev1_qo_model \
  src/openpi/models/model_test.py::test_pi0_ocaev1_training_config \
  src/openpi/models/model_test.py::test_pi0_ocaev1_train_sample_and_checkpoint_roundtrip -q
```

结果：

```text
28 passed
1,243 warnings
elapsed: 177.35 s
```

warnings 均来自现有 JAX/Flax deprecated API。

## 6. GPU 0 真实训练初始化与单步更新

环境：

```text
physical GPU: 0
GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
checkpoint: /data1/gqy/model/pi0_base/params
model dtype: bfloat16 compute
batch size: 1
image views: 3
action horizon: 50
action dim: 32
target: q_o
rank: 16
```

使用仓库真实接口：

```text
scripts/train.py::init_train_state
scripts/train.py::train_step
```

初始化结果：

```text
init seconds: 36.8640
step before: 0
trainable parameter count: 18,823,456
GPU resident after init: 8,745 MiB (nvidia-smi process-level observation)
```

zero-gate 固定 noise 采样：

```text
compile + first run: 25.2719 s
steady run: 0.038823 s
actions shape: (1, 50, 32)
actions finite: true
repeat exactly equal: true
```

一次真实训练更新：

```text
train compile + first step: 46.7156 s
step after: 1
loss: 1.2981908321
loss finite: true
grad norm: 0.0253329258
grad norm finite: true
changed trainable leaves: 2
max trainable delta: 2.4974476e-08
```

发生更新的两个叶子对应 zero-init gate projection。默认 cosine schedule 在 warmup 首步的学习率约为 `2.5e-8`，因此更新后 bfloat16 固定 noise actions 与 zero-gate actions 仍逐位相同：

```text
zero vs one-step-trained max abs action delta: 0.0
```

这不影响 P7 的“单步更新链路闭合”结论；是否在多步后形成可测输出变化属于 P8 smoke training 的验收内容。

## 7. 真实状态默认加载验证

训练 step 后直接使用完整真实参数树调用：

```text
config.model.load(state.params.to_pure_dict())
```

结果：

```text
default config load: success
restored actions shape: (1, 50, 32)
restored actions finite: true
```

这验证了 `bias: None` 静态槽修复不仅能通过 dummy checkpoint 测试，也能作用于完整 OCAE 模型。

## 8. 推理性能对比

固定条件：

```text
physical GPU: 0
dtype: bfloat16
batch size: 1
image views: 3
action horizon: 50
flow steps: 2
fixed noise: true
XLA preallocation: disabled
```

JAX allocator 测量：

| 模型 | compile + first sample | steady sample | sample peak bytes | sample peak GiB |
|---|---:|---:|---:|---:|
| base pi0 | 20.3306 s | 0.037607 s | 6,999,062,016 | 6.5184 |
| OCAE zero-gate | 23.6786 s | 0.038717 s | 7,134,259,456 | 6.6443 |
| OCAE after one update | 17.8020 s* | 0.038635 s | - | - |

`*` 该 compile 数据来自同一训练进程中的后续编译，受到 compilation cache 影响，不与独立进程 cold compile 直接比较。

zero-gate 相对 base：

```text
steady latency delta: +2.95%
sample peak memory delta: +135,197,440 bytes
sample peak memory delta: +0.1259 GiB / +1.93%
```

包含一次反向传播的 OCAE 全流程峰值：

```text
7,188,550,912 bytes
6.6949 GiB
```

Overview encoder 和 controller 在 `sample_actions()` 的 denoising `while_loop` 之前只计算一次；steady 延迟的增量主要来自一次 prefix conditioning 以及每层 conditional Q/O 低秩分支。

## 9. 默认 pi0 GPU 回归

命令：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 105.39 s
```

说明 OCAE disabled 的完整默认 π0 loss 和 sampling 未回归。

## 10. 静态检查

执行：

```text
ruff check
ruff format --check
git diff --check
```

结果全部通过。无关的 `uv.lock` 修改未纳入 P7。

## 11. P7 最终状态

```text
训练配置完成
失败先行测试完成
JIT 单步更新完成
固定 noise 推理完成
checkpoint 保存/恢复闭环完成
真实 base checkpoint 初始化完成
默认加载接口修复并验证完成
性能和显存测量完成
disabled 模式回归完成
可以进入 P8 smoke training
```
