# OCAE V1 P3 Conditioning Modules 验证报告

## 1. 阶段目标

本报告对应分步计划 P3：

```text
独立实现 Overview Context Encoder、State Context Encoder、
zero-gate Action Conditioning Controller 和统一 wrapper。
```

本阶段只实现和验证 conditioning 模块，不在 `Pi0` 中实例化，也不接入 Gemma Attention。

## 2. 修改文件

生产代码：

```text
src/openpi/models/overview_action_conditioning.py
src/openpi/models/pi0_config.py
```

新增测试：

```text
src/openpi/models/overview_action_conditioning_test.py
```

## 3. 配置

新增：

```text
OverviewActionConditioningConfig
```

字段：

```text
enabled=False
rank=16
lora_alpha=16.0
target="q_o"
use_state_context=True
```

配置校验：

```text
rank > 0
lora_alpha > 0
target in {q, o, q_o}
启用 V1 时禁止 pi05
启用 V1 时禁止现有 PaliGemma/action-expert LoRA variant
```

`Pi0Config` 新增该配置字段，默认 disabled。

## 4. Overview Context Encoder

输入：

```text
contextualized prefix_out
PrefixEmbeddings static layout
ordered image masks
language token mask
```

处理：

```text
按 image_lengths 恢复每个 view segment
每个 view 独立 masked mean
language segment 独立 masked mean
无效 view 输出零 pooled vector
拼接 view validity bits
shared no-bias LayerNorm
Dense(action_width) -> SiLU -> Dense(action_width)
```

不同 view 不做平均，而是按稳定顺序拼接，因此保留 camera identity。

layout 覆盖 token 数与 `prefix_out.shape[1]` 不一致时会明确报错。

## 5. State Context Encoder

结构：

```text
state
-> Dense(action_width)
-> SiLU
-> Dense(action_width)
-> state context
```

当 `use_state_context=False` 时，wrapper 返回相同 shape/dtype 的零 context。State encoder 仍被创建，保持参数结构稳定。

## 6. Action Conditioning Controller

结构：

```text
LayerNorm(scene context)
+ LayerNorm(state context)
-> concatenate
-> Dense(action_width)
-> SiLU
-> zero-init Dense(num_layers * 2 * num_heads)
-> reshape [B,L,2,N]
-> tanh
-> q_gates/o_gates [B,L,N]
```

最终 gate projection 的 kernel 和 bias 均为零初始化，因此：

```text
initial q_gates == 0
initial o_gates == 0
```

## 7. 统一 Wrapper

新增：

```text
OverviewActionConditioning
```

提供：

```text
encode_overview()
encode_state()
make_gates()
__call__()
```

P4 可以直接复用这些接口，在 prefix-only forward 后计算固定 gates。

## 8. 测试优先结果

在实现 P3 模块前运行新增测试：

```text
collection error
OverviewActionConditioning 尚不存在
```

说明测试确实覆盖了旧代码缺失的 P3 能力。

首次实现后出现两个测试夹具问题：

1. view 特征只相差常数平移，LayerNorm 后相同，不能用于检验 view 顺序。
2. surrogate gate 权重 reshape 元素数错误。

修正测试数据后，生产实现无需修改即通过。

## 9. P3 单元测试覆盖

最终覆盖：

1. masked image hidden states 不影响 overview context。
2. language padding hidden states 不影响 overview context。
3. 所有 view 无效时输出 finite。
4. 交换非仿射 view 特征会改变 overview context。
5. scene/state context shape 正确。
6. q/o gate shape 为 `[B,L,N]`。
7. 初始 q/o gates 严格为零。
8. wrapper 可通过 JIT。
9. `use_state_context=False` 返回零 context。
10. zero-init gate projection kernel/bias gradient 非零。
11. float32 输出 dtype 正确。
12. bfloat16 context/gate 输出 dtype 正确。
13. rank/alpha/target 配置校验。
14. π0.5 与现有 LoRA variant 组合校验。

最终 P3 单测：

```text
7 passed
elapsed: 23.10 s
```

## 10. P1-P3 Focused 回归

GPU：物理 GPU 0，RTX 5090。

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest \
  src/openpi/models/overview_action_conditioning_test.py \
  src/openpi/models/prefix_layout_test.py \
  src/openpi/models/pi0_split_forward_test.py \
  src/openpi/models/pi0_test.py -q
```

结果：

```text
14 passed
112 warnings
elapsed: 110.29 s
```

该次组合回归发生在增加独立 bfloat16 dtype case 之前；新增 dtype case 随后在 P3 单测中通过，不影响其余模块。

覆盖：

```text
P3 mask/zero-gate/JIT/gradient/config
P2 Prefix layout
P1 joint/split 等价与梯度
原 π0 参数树与 freeze filter
```

## 11. 完整默认 π0 GPU 回归

执行：

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

结果：

```text
1 passed
322 warnings
elapsed: 126.83 s
```

验证：

```text
默认 OverviewActionConditioningConfig.enabled=False
完整默认 Pi0 创建正常
JIT compute_loss() 正常
JIT sample_actions(num_steps=10) 正常
loss/action shape 正确
```

警告来自现有 JAX/Flax deprecated API，不是 P3 错误。

## 12. 静态检查

以下检查全部通过：

```text
ruff check
ruff format --check
git diff --check
```

## 13. 参数树与行为边界

P3 conditioning 模块尚未在 `Pi0.__init__()` 中实例化。

因此默认模型：

```text
没有新增 OCAE 参数
现有 checkpoint 结构不变
现有 freeze filter 结果不变
compute_loss/sample_actions 数值路径不变
```

P3 只证明：

```text
conditioning 模块能独立正确计算
zero gate 初始化成立
gate projection 可以收到梯度
```

P3 尚未证明：

```text
gate 能穿过 Gemma nn.scan
Conditional Q/O LoRA 生效
flow loss 能回传到完整 OCAE 模块
```

这些属于 P4/P5。

## 14. P3 结论

P3 验收通过：

```text
Overview pooling 的 mask/layout 行为正确
多视角顺序被保留
State encoder 与 state-disabled 路径正确
初始 q/o gates 严格为零
controller final projection gradient 非零
float32/bfloat16 dtype 正确
模块可 JIT
配置边界正确
默认 Pi0 参数树与行为无回归
完整 GPU 回归通过
```

可以进入 P4：

```text
把 conditioning gates 沿 layer 轴传入 Gemma nn.scan，
只实现 expert 1 的 Conditional Q LoRA，暂不实现 O LoRA。
```
