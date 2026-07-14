# OCAE V1 P1 Split Forward 验证报告

## 1. 阶段目标

本报告对应分步计划 P1：

```text
把 Pi0.compute_loss() 从 joint prefix+suffix forward
改为 prefix-only KV cache + suffix-only forward。
```

本阶段不包含：

```text
Prefix 结构体
Overview Context Encoder
State Context Encoder
Conditioning Controller
Conditional Q/O LoRA
checkpoint/freeze 修改
```

## 2. 修改文件

生产代码：

```text
src/openpi/models/pi0.py
```

新增测试：

```text
src/openpi/models/pi0_split_forward_test.py
```

## 3. 生产代码修改

原训练路径：

```text
[prefix_tokens, suffix_tokens]
-> one joint Gemma forward
-> suffix_out
```

P1 路径：

```text
prefix_tokens
-> prefix-only Gemma forward
-> prefix KV cache

suffix_tokens + prefix KV cache
-> suffix-only Gemma forward
-> suffix_out
```

suffix forward 复用现有 `sample_actions()` 的：

```text
suffix self-attention mask
suffix-to-prefix cross mask
prefix valid-length position offset
KV cache 调用方式
```

`embed_prefix()`、`embed_suffix()`、`sample_actions()`、Gemma、参数树和 flow loss 未修改。

## 4. Joint/Split 等价测试

新增 dummy Gemma 测试覆盖：

```text
两个 expert
batch size 2
prefix length 5
suffix length 3
第二个 batch 含两个 prefix padding token
π0 风格 state/action suffix ar_mask
```

比较：

```text
joint suffix hidden states
cached split suffix hidden states
joint/cache split 的有效 K/V entries
```

GPU 7 上 joint 与 split 不是逐位相同。不同 GEMM/softmax kernel shape 会改变浮点累积顺序，观测到的 suffix 最大绝对差异不超过约 `3.37e-3`。

测试使用：

```text
rtol = 5e-3
atol = 5e-3
```

cache 比较只覆盖 `input_mask=True` 的有效 token。无效 padding query 在全 mask 下的内部值可能不同，但不会被后续 attention 使用。

## 5. 反向传播验证

dummy test 对 cached split forward 的 suffix hidden state 构造标量 loss，并对完整 Gemma 参数树执行 `jax.grad`。

断言：

```text
所有 gradient leaves 均 finite
至少存在一个非零 gradient leaf
```

结果：通过。

这验证了 loss 可以跨 suffix forward 和 prefix KV cache 回传到两次 Gemma 调用。

## 6. 测试结果

### 6.1 Focused tests

执行：

```bash
CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/pi0_split_forward_test.py src/openpi/models/pi0_test.py -q
```

结果：

```text
5 passed
80 warnings
elapsed: 50.67 s
```

覆盖：

```text
joint/split 数值等价
有效 cache 数值等价
原 π0 参数树
原 full-finetune freeze filter
原 PaliGemma/action-expert LoRA freeze filter
```

加入 gradient assertion 后单独重跑 split test：

```text
1 passed
96 warnings
elapsed: 62.75 s
```

### 6.2 完整默认 π0 GPU 测试

执行：

```bash
CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 102.57 s
```

覆盖：

```text
完整默认 Pi0 创建
JIT compute_loss()
JIT sample_actions(num_steps=10)
loss/action shape
```

警告均来自现有 JAX/Flax deprecated API，不是 P1 错误。

## 7. 本地 Pi0 Base Checkpoint 验证

使用：

```text
GPU: physical GPU 7
checkpoint: /data1/gqy/model/pi0_base/params
dtype: checkpoint original float32
batch size: 1
RNG: jax.random.key(0)
```

P0 joint-forward 基线：

```text
loss mean: 6.9216618538
```

P1 split-forward：

```text
loss shape: (1, 50)
loss mean: 6.9274382591
loss std: 0.3569163084
loss finite: true
first call including compile: 65.60 s
```

均值变化：

```text
absolute delta: 0.0057764053
relative to baseline mean: approximately 0.083%
```

该差异与 dummy GPU 测试观察到的浮点 kernel 顺序变化一致。模型输出 shape 正确且全部 finite。

`sample_actions()` 本阶段未修改；完整默认模型的 10 步采样回归已经通过，因此没有重复执行本地 checkpoint 采样。

## 8. 静态检查

执行：

```text
ruff check
ruff format --check
git diff --check
```

结果：全部通过。

## 9. 参数树结论

P1 没有新增或删除模型参数。

现有 `pi0_test.py` 全部通过，说明：

```text
普通 π0 参数结构不变
现有 LoRA 参数结构不变
现有 freeze filter 行为不变
```

## 10. P1 结论

P1 验收通过：

```text
compute_loss 已改为 prefix cache + suffix forward
joint/split 在 GPU 数值容差内等价
padding token 已正确处理
split forward 梯度 finite 且非零
完整 π0 JIT loss 和采样测试通过
本地 float32 pi0_base loss finite
参数树和 freeze filter 无变化
静态检查通过
```

可以进入 P2：

```text
把 embed_prefix() 返回值结构化为带静态 layout metadata 的 PrefixEmbeddings，
仍不加入 Overview、Controller 或 Conditional Q/O LoRA。
```
