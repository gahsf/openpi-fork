# OCAE V1 P4 Conditional Query LoRA 验证报告

## 1. 阶段目标

本报告对应分步计划 P4：

```text
把逐层逐 head conditioning gates 传入 Gemma nn.scan，
只为 expert 1 增加 Conditional Q LoRA，
并把 contextualized prefix/state conditioning 接入 Pi0 训练与推理。
```

本阶段不实现 O LoRA，不修改 K/V，也不处理 base checkpoint missing 参数和 OCAE-only freeze filter。

## 2. P3 提交

P4 开始前，P3 已提交并推送：

```text
commit: 0058a8ea60ab7da2d5f861fafb5aba0e10e8a6a6
branch: origin/overlockv1
```

## 3. 修改文件

生产代码：

```text
src/openpi/models/gemma.py
src/openpi/models/pi0.py
```

新增/修改测试：

```text
src/openpi/models/conditional_action_lora_test.py
src/openpi/models/pi0_test.py
src/openpi/models/model_test.py
```

## 4. Conditional Query LoRA

新增 Linen 子模块：

```text
ConditionalQueryLoRA
```

每层 expert 1 参数：

```text
A_q: [N,D_action,R]
B_q: [N,R,H]
```

计算：

```text
Q_low = einsum(x,A_q)                           # [B,T,N,R]
Q_low *= q_gate[:,None,:,None]
Q_delta = einsum(Q_low,B_q)                    # [B,T,N,H]
Q = Q_base + (lora_alpha/rank) * Q_delta
```

Q delta 在 RoPE 和 head-dim scaling 之前加入。

A/B 均使用 `normal(stddev=0.01)` 非零初始化；identity 由 zero gate 保证。

参数只在以下条件下创建：

```text
conditioning config enabled
target in {q, q_o}
expert index == 1
```

expert 0、action expert K/V 和 base Q/O 参数均不修改。

## 5. nn.scan 与 remat

公开输入 gates：

```text
q_gates/o_gates: [B,L,N]
```

Gemma Module 内部转换：

```text
stack -> [B,L,2,N]
transpose -> [L,B,2,N]
```

`nn.scan` 对 conditioning 使用 `in_axes=0`，每层 Block 收到：

```text
layer_action_conditioning: [B,2,N]
```

同步修改：

```text
Block signature
Attention signature
scan in_axes
remat static_argnums
Module.init dummy call
```

当 `action_conditioning=None` 时，Module 自动构造 `[L,B,2,N]` 零 gates，保持默认调用兼容。

## 6. Lazy Init 与参数命名

`gemma.Module.init()` 仍使用两个非空 dummy expert 输入，因此启用 Q conditioning 时会创建：

```text
layers/attn/conditional_q_lora_1/lora_a
layers/attn/conditional_q_lora_1/lora_b
```

scan 后真实参数 shape：

```text
lora_a: [L,N,D_action,R]
lora_b: [L,N,R,H]
```

默认 disabled 模型不创建任何 conditional Q 参数。

## 7. Pi0 接入

`Pi0.__init__()`：

1. 把静态 conditioning config 传入 Gemma。
2. 所有原始 Pi0 参数创建完成后，再创建 `OverviewActionConditioning`。
3. disabled 时该属性为 `None`，不新增 NNX 参数。

训练 `compute_loss()`：

```text
prefix-only Gemma -> contextualized prefix_out + KV cache
prefix_out + image/language masks -> scene context
obs.state -> state context
controller -> q/o gates
suffix-only Gemma with KV cache + gates
flow head -> loss
```

推理 `sample_actions()`：

```text
prefix_out/KV/context/gates 只计算一次
gates 位于 denoising loop 外
每个 flow step 复用固定 gates 和 prefix KV
```

## 8. 测试优先结果

修改 Gemma 前运行 Conditional Q 测试：

```text
2 failed
Module 不接受 action_conditioning_config
```

修改 Pi0 前运行参数树测试：

```text
1 failed
Pi0 参数树中不存在 overview_action_conditioning
```

两项失败均符合预期，证明测试覆盖了旧代码缺失能力。

## 9. Gemma 级 Identity/Cache/Gradient 测试

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/conditional_action_lora_test.py -q
```

结果：

```text
2 passed
114 warnings
elapsed: 86.54 s
```

验证：

1. disabled Gemma 无 conditional Q 参数。
2. enabled Gemma 创建正确 scan layer 参数。
3. `action_conditioning=None` 与显式 zero gates 输出逐位相等。
4. 非零 q_gate 改变 action suffix 输出。
5. 改变 action gates 不影响 prefix output。
6. 改变 action gates 不影响 prefix K/V cache。
7. 非零 gate 下 Q LoRA A/B gradient 全部 finite 且非零。

## 10. Pi0 参数树与端到端 JIT

参数树测试：

```text
overview_action_conditioning 参数存在
conditional_q_lora_1 参数存在
conditional_o_lora_1 参数不存在
```

结果：通过。

OCAE enabled dummy π0 完整测试：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_ocaev1_q_model -q
```

结果：

```text
1 passed
330 warnings
elapsed: 118.92 s
```

覆盖：

```text
完整 Pi0 OCAE 模型创建
contextualized prefix conditioning
JIT compute_loss()
JIT sample_actions(num_steps=2)
zero-init gates
```

## 11. P1-P4 Focused 回归

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest \
  src/openpi/models/conditional_action_lora_test.py \
  src/openpi/models/overview_action_conditioning_test.py \
  src/openpi/models/prefix_layout_test.py \
  src/openpi/models/pi0_split_forward_test.py \
  src/openpi/models/pi0_test.py -q
```

结果：

```text
18 passed
230 warnings
elapsed: 177.97 s
```

## 12. 默认 π0 回归

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 116.12 s
```

说明 Gemma 新 scan 输入与默认零 conditioning 没有破坏 OCAE-disabled 模型的 loss/采样链路。

## 13. 完整 OCAE 梯度链路诊断

配置：

```text
dummy PaliGemma/action expert
batch size 1
rank 2
target q
只对 overview_action_conditioning 和 conditional_q_lora_1 求导
```

Zero-gate 初始状态：

```text
loss: 2.9838857651
gate_proj: finite=true, nonzero=true, norm=5.16944e-4
conditional_q: finite=true, nonzero=false, norm=0
overview_encoder: finite=true, nonzero=false, norm=0
state_encoder: finite=true, nonzero=false, norm=0
```

这符合 zero-gate 初始化预期：第一步只有最终 gate projection 学习。

把 gate projection kernel/bias 设为 `1e-3` 后：

```text
loss: 2.9837572575
gate_proj: finite=true, nonzero=true, norm=5.16890e-4
conditional_q: finite=true, nonzero=true, norm=1.79959e-4
overview_encoder: finite=true, nonzero=true, norm=2.59002e-6
state_encoder: finite=true, nonzero=true, norm=1.53779e-6
```

说明 gate 离开零后，flow loss 可以回传到 Q LoRA A/B、Overview encoder 和 State encoder。

## 14. 参数量

按真实 π0 默认宽度、18 层、8 heads、rank 16 统计：

```text
Overview/State/Controller: 12,925,216
Conditional Q LoRA: 2,949,120
P4 新增总参数: 15,874,336
```

当前 `get_freeze_filter()` 尚未切换为 OCAE-only；精确可训练参数过滤属于 P6。

## 15. 静态检查

以下全部通过：

```text
ruff check
ruff format --check
git diff --check
```

## 16. P4 边界

P4 已完成：

```text
逐层逐 head gates 进入 scan
expert 1 Conditional Q LoRA
zero-gate identity
prefix KV 不变
训练/推理 Pi0 接入
完整梯度链路
```

P4 未完成且不应提前声称：

```text
Conditional O LoRA
target=o 的实际调制
base pi0 checkpoint 缺失新参数加载
OCAE-only freeze filter
正式训练配置
```

其中 checkpoint loader 和 freeze filter 按计划在 P6 处理，因此本阶段没有加载本地 `/data1/gqy/model/pi0_base/params` 到增强模型。

## 17. P4 结论

P4 验收通过：

```text
zero gate 与 bypass 等价
prefix output/KV 完全不受 action gates 影响
非零 Q gate 能改变 action suffix
Q LoRA scan 参数 shape 正确
Q LoRA A/B gradient finite 且非零
完整 flow loss 梯度可以到达所有 P4 新模块
OCAE enabled/disabled Pi0 均可 JIT loss 和采样
P1-P3 无回归
静态检查通过
```

可以进入 P5：

```text
在 attention encoded head 维仍存在时实现 Conditional O LoRA，
支持 target=q/o/q_o，并保持 attention logits/probs 和 prefix KV 不变。
```
