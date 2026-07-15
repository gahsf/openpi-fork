# OCAE V1 P5 Conditional Output LoRA 验证报告

## 1. 阶段目标

本报告对应分步计划 P5：

```text
在 attention encoded 仍保留 head 维时增加 Conditional O LoRA，
完整支持 target=q/o/q_o。
```

本阶段不修改 prefix K/V，不实现 checkpoint missing 参数合并和 OCAE-only freeze filter。

## 2. P4 提交

P5 开始前，P4 已提交并推送：

```text
commit: 52dbbd67437722d72c330d4bc7634c3ed9aa29be
branch: origin/overlockv1
```

## 3. 修改文件

生产代码：

```text
src/openpi/models/gemma.py
```

测试：

```text
src/openpi/models/conditional_action_lora_test.py
src/openpi/models/pi0_test.py
src/openpi/models/model_test.py
```

Pi0 conditioning 数据流在 P4 已完整接入，P5 无需修改 `pi0.py`。

## 4. Conditional Output LoRA

新增 Linen 子模块：

```text
ConditionalOutputLoRA
```

每层 expert 1 参数：

```text
A_o: [N,H,R]
B_o: [N,R,D_action]
```

计算：

```text
encoded: [B,T,N,H]
O_low = einsum(encoded,A_o)                     # [B,T,N,R]
O_low *= o_gate[:,None,:,None]
O_delta = einsum(O_low,B_o)                    # [B,T,D_action], sum over N
O = O_base + (lora_alpha/rank) * O_delta
```

A/B 均使用 `normal(stddev=0.01)` 非零初始化，identity 由 zero o_gate 保证。

参数只在以下条件下创建：

```text
conditioning config enabled
target in {o, q_o}
expert index == 1
```

## 5. 接入位置

O branch 位于：

```text
Q/K/V
-> attention logits
-> mask + softmax probabilities
-> value aggregation
-> encoded [B,T,N,H]
-> base O projection + Conditional O LoRA
```

因此 Conditional O 不修改当前层的：

```text
Q/K/V
attention logits
attention probabilities
encoded value aggregation
prefix KV cache
```

O delta 在 head 汇总前计算，per-head gate 语义被保留。

## 6. 测试优先结果

新增 O-only/q_o 测试后、修改 Gemma 前：

```text
2 passed, 2 failed
```

原 P4 Q 测试继续通过；失败项为：

```text
非零 o_gate 尚不改变 suffix
q_o 参数树缺少 conditional_o_lora_1
```

符合 P5 预期。

## 7. Gemma Q/O 测试

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/conditional_action_lora_test.py -q
```

结果：

```text
4 passed
248 warnings
elapsed: 129.88 s
```

O-only 验证：

1. `None` 与显式 zero gates 输出逐位相等。
2. target=o 时非零 q_gate 不改变 suffix。
3. target=o 时非零 o_gate 改变 suffix。
4. action gates 不影响 prefix output。
5. action gates 不影响 prefix KV cache。
6. O A/B scan 参数 shape 正确。
7. 非零 o_gate 下 O A/B gradient finite 且非零。

q_o 验证：

```text
conditional_q_lora_1 参数存在
conditional_o_lora_1 参数存在
```

## 8. q/o/q_o Pi0 参数树

参数化测试结果：

```text
target=q   -> Q params=true,  O params=false
target=o   -> Q params=false, O params=true
target=q_o -> Q params=true,  O params=true
```

执行结果：

```text
3 passed
12 warnings
elapsed: 15.53 s
```

所有 target 都包含外部 `overview_action_conditioning` 参数。

## 9. 完整 q_o Pi0 JIT

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_ocaev1_qo_model -q
```

结果：

```text
1 passed
338 warnings
elapsed: 95.03 s
```

覆盖：

```text
完整 q_o Pi0 模型创建
contextualized prefix/state conditioning
Q/O zero gates
JIT compute_loss()
JIT sample_actions(num_steps=2)
```

## 10. P1-P5 Focused 回归

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
22 passed
372 warnings
elapsed: 218.93 s
```

## 11. 默认 π0 回归

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 104.65 s
```

说明 disabled 配置不创建或执行 Conditional Q/O 分支，默认 π0 loss/采样正常。

## 12. 完整 q_o 梯度链路

配置：

```text
dummy PaliGemma/action expert
batch size 1
rank 2
target q_o
只对 overview_action_conditioning 和 conditional Q/O 参数求导
```

Zero-gate 初始状态：

```text
loss: 2.9841604233
gate_proj: finite=true, nonzero=true, norm=4.61363e-3
conditional_q: finite=true, nonzero=false, norm=0
conditional_o: finite=true, nonzero=false, norm=0
overview_encoder: finite=true, nonzero=false, norm=0
state_encoder: finite=true, nonzero=false, norm=0
```

符合预期：第一步只有最终 gate projection 获得梯度。

把 gate projection kernel/bias 设为 `1e-3` 后：

```text
loss: 2.9841876030
gate_proj: finite=true, nonzero=true, norm=4.61730e-3
conditional_q: finite=true, nonzero=true, norm=1.79509e-4
conditional_o: finite=true, nonzero=true, norm=2.27546e-3
overview_encoder: finite=true, nonzero=true, norm=1.07060e-4
state_encoder: finite=true, nonzero=true, norm=6.35024e-5
```

说明 gate 离开零后，flow loss 可以同时回传到 Q/O LoRA、Overview encoder 和 State encoder。

## 13. 参数量

按真实 π0 默认宽度、18 层、8 heads、rank 16 统计：

```text
Overview/State/Controller: 12,925,216
Conditional Q LoRA: 2,949,120
Conditional O LoRA: 2,949,120
P5 q_o 新增总参数: 18,823,456
```

## 14. 静态检查

以下全部通过：

```text
ruff check
ruff format --check
git diff --check
```

## 15. P5 边界

P5 已完成：

```text
target=q/o/q_o
Conditional Q/O LoRA
逐层逐 head gates
zero-gate identity
prefix output/KV 不变
Q/O A/B gradient
完整 q_o flow gradient
默认与增强模型 JIT
```

P5 尚未完成：

```text
base checkpoint missing OCAE 参数合并
OCAE-only freeze filter
正式训练配置
checkpoint 保存/恢复
smoke training
```

这些属于 P6-P8。

## 16. P5 结论

P5 验收通过：

```text
O gate 在 head 汇总前正确应用
O branch 不修改当前层 logits/probs/KV
q/o/q_o 参数树正确
zero gate 与 bypass 等价
非零 O gate 能改变 action suffix
Q/O A/B gradient finite 且非零
完整 flow loss 可到达所有 OCAE 模块
q_o 与默认 Pi0 均可 JIT loss/采样
P1-P4 无回归
静态检查通过
```

可以进入 P6：

```text
允许 base checkpoint 精确补入 OCAE 新参数，
并把阶段 1 trainable 参数严格限制为 Overview/State/Controller 和 Conditional Q/O LoRA。
```
