# OCAE V1 P8 Smoke Training 验证报告

## 1. 结论

P8 通过。最终 smoke 配置在物理 GPU 0 上连续完成 50 个真实训练 step：

```text
所有 loss finite
controller gate projection 在第 1 步获得非零梯度
overview/state encoder 在第 2 步获得非零梯度
conditional Q/O LoRA A/B 在第 2 步获得非零梯度
Q/O gate 从零离开且未饱和
Q/O effective delta ratio 有界增长
冻结参数抽样指纹不变
checkpoint 恢复前后固定-noise actions 逐位一致
```

P7 前置提交：

```text
cbd37d4 test(ocae): add end-to-end jax pi0 validation
```

## 2. 本阶段修改

```text
src/openpi/training/config.py
src/openpi/models/model_test.py
```

新增独立配置：

```text
pi0_ocaev1_smoke
```

最终配置：

```text
model: pi0 OCAE V1
target: q_o
rank: 16
lora_alpha: 16.0
freeze: OCAE-only
batch size: 1
seed: 42
steps: 50
warmup steps: 5
peak lr: 1e-5
decay lr: 1e-6
decay steps: 50
EMA: disabled
log interval: 1
```

正式配置使用 `gs://openpi-assets/checkpoints/pi0_base/params`；本地执行时仅把 loader 替换为 `/data1/gqy/model/pi0_base/params`。

## 3. 配置失败先行测试

新增配置测试后，生产配置修改前得到预期失败：

```text
Config 'pi0_ocaev1_smoke' not found
```

配置实现后测试转绿，并验证模型、freeze、loader、训练预算和学习率 schedule。

## 4. 学习率探索

首轮使用 `peak=1e-4, decay=1e-5`。所有训练和 checkpoint 链路均通过，但约 step 25 时 gate 已达到 `max abs=1.0`，不适合作为最终 smoke 超参。

首轮 step 50：

```text
Q/O gate std: approximately 1.0 / 1.0
Q delta ratio: 1.260%
O delta ratio: 1.668%
```

最终将 peak/decay learning rate 均降低 10 倍。最终轮 gate 未饱和，delta ratio 平稳增长，因此以下结论均以 `1e-5 -> 1e-6` 轮次为准。

首轮 checkpoint 依照仓库“不删除文件”的约束保留：

```text
/tmp/openpi_ocaev1_p8_smoke_step50_params
du size: 4.8G
```

## 5. 最终运行环境

```text
physical GPU: 0
GPU: NVIDIA GeForce RTX 5090
checkpoint: /data1/gqy/model/pi0_base/params
compute dtype: bfloat16
trainable params: 18,823,456
batch size: 1
synthetic subset: one fixed fake observation/action batch
training RNG: seed 42 folded by step
steps: 50
init seconds: 41.5811
```

初始化指标：

```text
Q/O gate mean/std/max abs: 0
Q/O effective delta ratio: 0
```

## 6. 指标定义

每一步采集：

```text
loss
global grad norm
overview encoder grad norm
state encoder grad norm
controller final gate projection grad norm
conditional Q LoRA A/B grad norm
conditional O LoRA A/B grad norm
Q/O gate mean/std/max abs
Q/O effective kernel delta ratio
step time
```

effective kernel delta ratio：

```text
Q ratio = ||g_q * (A_q @ B_q) * alpha/rank||_F / ||W_q_base||_F
O ratio = ||g_o * (A_o @ B_o) * alpha/rank||_F / ||W_o_base||_F
```

本配置 `alpha/rank = 16/16 = 1`。

## 7. 梯度传播顺序

| 模块 | 首次非零 step |
|---|---:|
| controller gate projection | 1 |
| overview encoder | 2 |
| state encoder | 2 |
| conditional Q LoRA A | 2 |
| conditional Q LoRA B | 2 |
| conditional O LoRA A | 2 |
| conditional O LoRA B | 2 |

这符合 zero-gate 设计：初始只有 gate projection 获得梯度；第一次更新后 gate 离开零；下一步 encoder 和 Q/O LoRA 开始获得有效梯度。

## 8. 关键 step 指标

### Step 1

```text
loss: 1.2988397
global/controller grad norm: 2.5499e-2 / 2.5499e-2
overview/state/Q-A/Q-B/O-A/O-B grad norm: 0
Q/O gate max abs: 6.5231e-4 / 6.5231e-4
Q/O delta ratio: 8.0151e-6 / 1.0465e-5
```

### Step 2

```text
loss: 1.2832375
overview/state grad norm: 2.8145e-5 / 8.6327e-6
controller grad norm: 2.3577e-2
Q A/B grad norm: 3.5027e-5 / 3.3985e-5
O A/B grad norm: 8.7956e-5 / 8.9044e-5
Q/O gate max abs: 1.9836e-3 / 1.9836e-3
Q/O delta ratio: 2.1326e-5 / 2.8687e-5
```

### Step 10

```text
loss: 1.4809707
Q gate std/max abs: 0.02063 / 0.03760
O gate std/max abs: 0.02161 / 0.03857
Q/O delta ratio: 0.02556% / 0.03518%
```

### Step 30

```text
loss: 8.1273394
Q gate std/max abs: 0.09326 / 0.16699
O gate std/max abs: 0.09912 / 0.16406
Q/O delta ratio: 0.11572% / 0.15846%
```

### Step 50

```text
loss: 2.8742297
global grad norm: 0.3337457
overview/state grad norm: 9.7336e-3 / 7.7542e-3
controller grad norm: 0.3241687
Q A/B grad norm: 0.0140098 / 0.0149212
O A/B grad norm: 0.0537448 / 0.0507038
Q gate mean/std/max abs: -0.001358 / 0.128906 / 0.222656
O gate mean/std/max abs: 0.005035 / 0.135742 / 0.215820
Q/O delta ratio: 0.15930% / 0.21783%
```

最终 gate 距离 `tanh` 的 ±1 饱和区仍有充分余量，Q/O delta 均远低于 base kernel norm。

## 9. Loss 与冻结参数

50 step loss：

```text
all finite: true
minimum: 0.9098324
maximum: 17.3095989
NaN/Inf: none
```

固定 synthetic batch 的 flow noise 和 timestep 随 step RNG 变化，因此单步 loss 方差明显；P8 不据此判断收敛或任务性能。

对全部 50 个冻结参数叶子分别抽取 16 个稳定位置，训练前后逐位比较：

```text
frozen fingerprint equal: true
```

训练的 `DiffState` 和 optimizer params 同时使用 `config.trainable_filter`；P6 已验证该 filter 只覆盖 OCAE 新参数。

## 10. 性能与显存

最终轮每步额外执行 prefix gate forward、模块梯度统计和 effective kernel ratio 计算：

```text
first diagnostic step including compile: 60.1777 s
steady diagnostic step mean: 0.083732 s
steady diagnostic step max: 0.110487 s
JAX peak bytes in use: 10,570,591,232
JAX peak GiB: 9.8446
```

50 step 连续运行未发生 OOM。

## 11. Checkpoint 保存与恢复

最终 checkpoint：

```text
path: /tmp/openpi_ocaev1_p8_smoke_lr1e5_step50_params
bytes: 5,146,516,254
du size: 4.8G
save seconds: 24.7542
restore + compile + sample seconds: 26.4654
```

固定 noise 恢复验证：

```text
pre-save actions finite: true
restored actions finite: true
restored actions exactly equal: true
max abs delta: 0.0
```

两个训练 checkpoint 均位于 `/tmp`，不会纳入 Git。

## 12. 回归与静态检查

P1-P8 CPU 聚焦回归：

```text
29 passed
1,243 warnings
elapsed: 185.58 s
```

OCAE disabled 的默认完整 pi0 GPU 回归：

```text
1 passed
322 warnings
elapsed: 106.66 s
```

静态检查：

```text
ruff check: passed
ruff format --check: passed
git diff --check: passed
```

warnings 来自现有 JAX/Flax deprecated API。无关的 `uv.lock` 修改未纳入 P8。

## 13. P8 最终状态

```text
50-step smoke training 完成
loss 全部 finite
gate 从零离开且最终未饱和
所有 OCAE 目标模块最终均有非零梯度
Q/O delta ratio 有界
冻结参数未改变
checkpoint 可恢复且输出逐位一致
性能和显存可接受
可以进入 P9 核心实验
```
