# OCAE V1 P6 Checkpoint Loader 与 Freeze Filter 验证报告

## 1. 结论

P6 在计划边界内通过：

```text
base pi0 checkpoint 能精确补入 OCAE V1 新参数
非 OCAE base 参数缺失和 shape 不匹配仍会失败
OCAE enabled 时只有新增 OCAE 参数可训练
OCAE disabled 时原有 full-finetune / LoRA freeze 行为不变
真实 q_o/rank=16 模型的 loss 与 2-step sampling 均成功且 finite
```

P5 前置提交：

```text
b712987 feat(ocae): condition action expert attention outputs
```

## 2. 本阶段范围

修改：

```text
src/openpi/training/weight_loaders.py
src/openpi/training/weight_loaders_test.py
src/openpi/models/pi0_config.py
src/openpi/models/pi0_test.py
```

本阶段不新增训练配置，不执行训练 step，也不验证训练 checkpoint 的保存/恢复闭环；这些属于 P7。

## 3. Checkpoint Loader

`CheckpointWeightLoader` 的 missing regex 从：

```text
.*lora.*
```

扩展为：

```text
.*(lora|overview_action_conditioning).*
```

它精确覆盖：

```text
overview_action_conditioning/...
conditional_q_lora_1/...
conditional_o_lora_1/...
```

conditional Q/O 的参数路径已包含 `lora`，因此沿用原有 LoRA missing 规则；外置的 overview/state/controller 模块由新增分支覆盖。普通 base 参数仍不能从 reference tree 静默补入。

## 4. Freeze Filter

OCAE enabled 时，可训练参数匹配：

```text
.*(overview_action_conditioning|conditional_(q|o)_lora_1).*
```

freeze filter 为上述匹配的 `Not`。因此：

```text
overview/state/controller 可训练
conditional Q/O LoRA 按 target=q/o/q_o 可训练
SigLIP/VLM expert 冻结
action expert base 参数冻结
state_proj/action_in_proj/action_out_proj 冻结
flow head 冻结
```

OCAE disabled 时继续使用原有 freeze filter 分支。

## 5. 失败先行验证

新增测试在修改生产代码前正确暴露两个问题：

```text
loader 只允许缺失 .*lora.*，无法补入外置 overview_action_conditioning 参数
原 freeze filter 会让 base 模型参数保持可训练
```

生产代码修改后，loader/freeze 聚焦测试结果：

```text
6 passed
```

覆盖：

1. 缺失的 OCAE 参数从 reference params 补入。
2. dtype 转换行为保持不变。
3. 普通 base 参数缺失仍在结构验证时失败。
4. shape mismatch 仍在结构验证时失败。
5. `target=q/o/q_o` 的可训练路径分别正确。
6. 所有非 OCAE 参数均被冻结。

## 6. P1-P6 聚焦回归

命令：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest \
  src/openpi/training/weight_loaders_test.py \
  src/openpi/models/conditional_action_lora_test.py \
  src/openpi/models/overview_action_conditioning_test.py \
  src/openpi/models/prefix_layout_test.py \
  src/openpi/models/pi0_split_forward_test.py \
  src/openpi/models/pi0_test.py -q
```

结果：

```text
28 passed
396 warnings
elapsed: 232.31 s
```

warnings 来自现有 JAX/Flax deprecated API，没有新增测试失败。

## 7. 真实本地 Base Checkpoint 合并

配置：

```text
physical GPU: 0
checkpoint: /data1/gqy/model/pi0_base/params
model: pi0 OCAE V1
target: q_o
rank: 16
lora_alpha: 16.0
```

结构与 dtype 验证：

```text
merge seconds: 10.8933
base array leaves: 50
base arrays loaded from checkpoint: true
new non-None leaves: 21
new parameter count: 18,823,456
new params supplied by reference initialization: true
full structure/dtype equality: true
```

这证明本地 base checkpoint 的所有原参数均被加载，且只有允许缺失的 OCAE 参数由初始化参考树补入。

## 8. 参数量

`q_o/rank=16` 的精确统计：

| 类别 | 参数量 |
|---|---:|
| 完整模型 | 3,256,871,984 |
| overview/state/controller | 12,925,216 |
| conditional Q | 2,949,120 |
| conditional O | 2,949,120 |
| OCAE 可训练总数 | 18,823,456 |

一致性：

```text
12,925,216 + 2,949,120 + 2,949,120 = 18,823,456
trainable / total = 0.577961%
```

## 9. GPU 0 真实模型运行

为控制单卡峰值显存，运行验证将 base checkpoint 恢复为 bfloat16，同时保留真实 OCAE 随机初始化和 zero-init gate。结果：

```text
new parameter initialization: 53.7993 s

loss compile + run: 27.6322 s
loss shape: (1, 50)
loss finite: true
loss mean: 6.9245934486

2-step sampling compile + run: 18.5527 s
actions shape: (1, 50, 32)
actions finite: true
actions mean: 0.0578011684
```

## 10. 默认 pi0 回归

命令：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 97.17 s
```

这验证了 OCAE disabled 的默认完整 pi0 loss/sampling 行为未被 P6 破坏。

## 11. P7 衔接注意项

P6 验证的是训练初始化所使用的 `CheckpointWeightLoader` 路径。额外诊断发现，`source_norm` 的 no-bias LayerNorm 在完整 NNX 状态树中以 `bias: None` 表示；`BaseModelConfig.load()` 默认执行的 Orbax tree intersection 会移除该静态槽位。P6 的训练初始化路径不经过这个 intersection，且真实运行已用完整状态树通过。

P7 在验证推理加载和 checkpoint 保存/恢复闭环时，应加入默认 `BaseModelConfig.load()` 回归，并对该 `None` 静态槽位做最小兼容处理。该事项不应被误判为本地 checkpoint 损坏或 P6 missing regex 失败。

## 12. P6 最终状态

```text
实现完成
loader/freeze 测试通过
P1-P6 聚焦回归通过
真实本地 checkpoint 精确合并通过
GPU 0 真实 OCAE loss 与采样通过
默认 pi0 完整回归通过
可以进入 P7
```
