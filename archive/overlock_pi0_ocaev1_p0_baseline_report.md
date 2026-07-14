# OCAE V1 P0 基线验证报告

## 1. 验证范围

本报告对应：

```text
archive/overlock_pi0_ocaev1_stepwise_implementation_validation_plan.md
第 3 节：通用验证环境
第 4 节：P0 基线记录
```

本阶段未修改模型源码。

## 2. 仓库状态

```text
commit: 15a9616a00943ada6c20a0f158e3adb39df2ccac
```

执行验证前已有未提交内容：

```text
M  uv.lock
?? AGENTS.md
?? archive/
```

这些内容不是 P0 验证产生的模型源码修改。

## 3. 最终 GPU 环境

按用户要求，最终本地 checkpoint 验证使用物理 GPU 7：

```text
GPU: NVIDIA GeForce RTX 5090
physical GPU index: 7
GPU UUID: GPU-63a0c8b5-587b-0f6c-9fe9-bc38a1363f20
GPU memory: 32607 MiB
JAX version: 0.5.3
JAX visible device: CudaDevice(id=0)
JAX backend: gpu
```

`CudaDevice(id=0)` 是 `CUDA_VISIBLE_DEVICES=7` 后的逻辑编号。

最终 checkpoint 命令使用：

```text
CUDA_VISIBLE_DEVICES=7
JAX_PLATFORMS=cuda
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## 4. 参数树与 Freeze Filter 测试

执行：

```bash
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda uv run pytest src/openpi/models/pi0_test.py -q
```

结果：

```text
4 passed
16 warnings
elapsed: 12.93 s
```

覆盖：

```text
π0 full finetune freeze filter
PaliGemma LoRA freeze filter
action expert LoRA freeze filter
PaliGemma + action expert LoRA freeze filter
```

警告来自 JAX/Flax deprecated API，不影响测试结论。

该测试是参数树/过滤规则验证，与具体 GPU 数值结果无关，因此切换到 GPU 7 后未重复执行。

## 5. 完整默认 π0 测试

执行：

```bash
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
321 warnings
elapsed: 105.08 s
```

覆盖：

```text
完整默认 Pi0 模型创建
compute_loss()
sample_actions(num_steps=10)
loss/action 输出 shape
```

警告同样是 JAX/Flax deprecated API，不是模型错误。

## 6. 本地 Checkpoint 结构验证

本地路径：

```text
/data1/gqy/model/pi0_base/params
```

CPU/NumPy restore 结果：

```text
parameter leaves: 50
parameter count: 3,238,048,528
original dtype: float32
parameter bytes: 12,952,194,112
parameter size: 12.0627 GiB
```

结论：

```text
Orbax checkpoint metadata 和参数数据可正常解析
本地路径正确
checkpoint 不是损坏或不完整状态
```

## 7. GPU 7 本地 Float32 Checkpoint 验证

配置：

```text
model config: Pi0Config()
checkpoint: /data1/gqy/model/pi0_base/params
batch size: 1
sample steps: 10
restore dtype: original float32
```

恢复结果：

```text
parameter count: 3,238,048,528
restored parameter size: 12.0627 GiB
restore dtype set: [float32]
restore time: 19.134 s
```

Flow loss：

```text
loss shape: (1, 50)
loss mean: 6.9216618538
loss std: 0.3572238982
loss finite: true
first call time including compile: 59.745 s
```

Action sampling：

```text
action shape: (1, 50, 32)
action mean: 0.0543918610
action std: 0.2530734837
actions finite: true
10-step first call time including compile: 32.829 s
```

## 8. GPU 3 诊断记录

初始验证使用 GPU 3。参数树测试和完整默认 π0 测试通过后，GPU 3 出现约 18 GiB 的不可见显存占用，`nvidia-smi` 未列出对应进程。

在该状态下：

```text
float32 checkpoint restore: OOM，缺少约 1.96 GiB 连续分配
bfloat16 checkpoint restore: 成功
bfloat16 loss/action: finite
```

该问题是 GPU 3 当时的显存状态，不是 checkpoint 或模型错误。切换到空闲 GPU 7 后，原始 float32 checkpoint 完整验证成功。

## 9. P0 结论

P0 基线验证通过：

```text
JAX CUDA 环境正常
现有 π0 参数树与 freeze filter 测试通过
完整默认 π0 loss 和采样测试通过
本地 pi0_base checkpoint 可正常解析和加载
GPU 7 上原始 float32 loss 和 10 步采样均成功且 finite
输出 shape 符合 Pi0Config 默认 action_horizon/action_dim
```

可以进入 P1：

```text
只把 Pi0.compute_loss() 从 joint forward 拆成 prefix KV cache + suffix forward，
暂不加入 Overview、Controller 或 Conditional Q/O LoRA。
```
