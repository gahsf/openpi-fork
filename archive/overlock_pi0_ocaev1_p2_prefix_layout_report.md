# OCAE V1 P2 Prefix Layout 验证报告

## 1. 阶段目标

本报告对应分步计划 P2：

```text
把 Pi0.embed_prefix() 的 tuple 返回值改为 PrefixEmbeddings，
增加 image/language segment 的静态 layout metadata。
```

本阶段不增加模型参数，也不实现任何 Overview、State、Controller 或 Conditional LoRA 计算。

## 2. 修改文件

生产代码：

```text
src/openpi/models/pi0.py
src/openpi/models/overview_action_conditioning.py
```

新增测试：

```text
src/openpi/models/prefix_layout_test.py
```

## 3. PrefixEmbeddings

新增 `flax.struct.dataclass`：

```text
tokens: array leaf
input_mask: array leaf
ar_mask: array leaf
image_lengths: static tuple[int, ...]
language_length: static int
```

`image_lengths` 和 `language_length` 使用 `pytree_node=False`，不会成为 JAX array leaf，也不会在 JIT 中动态变化。

P2 没有增加用于切片的额外 helper；后续 Overview Context Encoder 可以按静态长度直接恢复各 segment。

## 4. embed_prefix() 修改

对每个 image view 记录：

```text
image_tokens.shape[1]
```

如果存在 language prompt，记录：

```text
language_length = tokenized_inputs.shape[1]
```

无 language prompt 时：

```text
language_length = 0
```

原始 token 拼接、input mask 和 ar_mask 计算保持不变。

## 5. 调用点修改

以下调用点从 tuple 解包改为字段访问：

```text
Pi0.compute_loss()
Pi0.sample_actions()
```

替换关系：

```text
prefix_tokens  -> prefix.tokens
prefix_mask    -> prefix.input_mask
prefix_ar_mask -> prefix.ar_mask
```

KV cache、attention mask、positions 和 flow head 没有数值逻辑修改。

## 6. 测试优先结果

在修改 `embed_prefix()` 前先运行新增测试，结果：

```text
2 failed, 1 passed
```

失败原因符合预期：

```text
旧 embed_prefix() 仍返回 tuple
不存在 image_lengths/language_length 字段
```

实现 PrefixEmbeddings 后：

```text
3 passed
```

## 7. Prefix Layout 测试覆盖

使用轻量 fake image embedder 和 language embedder，不构建 SigLIP。

覆盖：

1. 两个 image view 的 segment 长度为 `(2, 3)`。
2. language segment 长度为 4。
3. 各 segment slice 与原始 fake embedding 一致。
4. image validity mask 正确扩展到对应 token segment。
5. language padding mask 位于正确尾部 segment。
6. 整个 prefix ar_mask 保持 false。
7. 无 language prompt 时 `language_length == 0`。
8. `jax.jit(lambda x: x)` 后静态 layout 保持不变。
9. JAX tree leaves 只有 tokens/input_mask/ar_mask 三个数组。

## 8. Focused 回归

执行：

```bash
CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest \
  src/openpi/models/prefix_layout_test.py \
  src/openpi/models/pi0_split_forward_test.py \
  src/openpi/models/pi0_test.py -q
```

结果：

```text
8 passed
112 warnings
elapsed: 73.32 s
```

覆盖：

```text
P2 prefix layout
P1 joint/split 数值等价
P1 split gradient
原 π0 参数树
原 freeze filter
```

警告来自现有 JAX/Flax deprecated API，不是 P2 错误。

## 9. 完整默认 π0 GPU 回归

执行：

```bash
CUDA_VISIBLE_DEVICES=7 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 104.24 s
```

验证：

```text
完整默认 Pi0 创建
JIT compute_loss()
JIT sample_actions(num_steps=10)
结构化 prefix 能通过两条完整调用链
loss/action shape 正确
```

## 10. 静态检查

以下检查全部通过：

```text
ruff check
ruff format --check
git diff --check
```

## 11. 参数树与数值行为

P2 只新增无参数的 `flax.struct.dataclass`，没有新增 `nnx.Param` 或 Linen parameter。

`pi0_test.py` 与完整模型测试通过，说明：

```text
模型参数树不变
freeze filter 行为不变
prefix token/mask 数值不变
compute_loss/sample_actions 链路正常
```

由于 P2 不改变数值计算，只改变返回容器和静态 metadata，本阶段没有重复加载本地 12.06 GiB `pi0_base` checkpoint；P1 已完成真实 checkpoint 验证。

## 12. P2 结论

P2 验收通过：

```text
PrefixEmbeddings 已实现
image/language 静态 layout 正确
padding 和缺失 language 路径正确
结构可安全穿过 JIT
compute_loss/sample_actions 已改为字段访问
P1 等价和梯度测试无回归
参数树与 freeze filter 无变化
完整 GPU 回归通过
```

可以进入 P3：

```text
独立实现 Overview Context Encoder、State Context Encoder 和 zero-gate Controller，
暂不接入 Gemma Attention。
```
