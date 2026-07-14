# OCAE V1 分步修改与验证计划

## 1. 计划目标

本文把 `Contextualized Overview-Conditioned Action Expert` 的 V1 实现拆成可独立验证、可独立提交的阶段。

目标不是尽快一次性完成全部代码，而是保证每一步都满足：

```text
改动范围明确
行为变化可测
失败原因可定位
通过后再进入下一步
```

对应总体方案：

```text
archive/overlock_pi0_ocaev1_contextualized_plan.md
```

## 2. 总体原则

### 2.1 实施原则

1. 先重构行为等价的 prefix/cache 前向，再加入新模块。
2. 先实现 Q-only，再实现 O-only/Q-O。
3. identity、梯度和参数树验证必须早于正式训练。
4. 每一步单独提交，不把多个高风险修改塞进同一个 commit。
5. 任一步未达到通过标准时，停止进入下一阶段。
6. 不同时修改 JAX 与 PyTorch；V1 只处理 JAX π0。
7. 不在 V1 中加入 π0.5、context token、attention bias 或 K/V modulation。

### 2.2 风险分级

| 级别 | 内容 | 主要风险 |
|---|---|---|
| 低 | Prefix 返回结构、独立 encoder 单元测试 | shape、mask、静态 metadata |
| 中 | `compute_loss()` 拆分、checkpoint/freeze | mask、positions、参数路径 |
| 高 | `gemma.Attention`、`nn.scan/remat`、lazy init | 参数树不一致、JIT/scan 失败、identity 被破坏 |

### 2.3 阶段依赖

```text
P0 基线记录
  -> P1 训练前向拆分
  -> P2 Prefix layout 结构化
  -> P3 Overview/State/Controller 独立模块
  -> P4 Q-only 接入 Gemma
  -> P5 O-only 与 Q/O 接入
  -> P6 Checkpoint 与 freeze
  -> P7 端到端回归
  -> P8 Smoke training
  -> P9 核心实验
```

## 3. 通用验证环境

### 3.1 CPU 快速测试

仓库安装了 CUDA JAX plugin、但执行环境没有可见 GPU 时，需要显式指定 CPU：

```bash
JAX_PLATFORMS=cpu uv run pytest <test_path> -q
```

适合 CPU 的测试：

```text
dummy Gemma attention
Prefix layout/mask
Overview/State/Controller
参数树和 freeze filter
checkpoint merge 的纯结构测试
```

### 3.2 GPU 测试

以下验证建议在 GPU 上执行：

```text
完整 Pi0 compute_loss JIT
完整 Pi0 sample_actions JIT
bfloat16 identity
真实 checkpoint restore
smoke training
吞吐、显存和延迟
```

### 3.3 每阶段通用静态检查

只检查本阶段涉及的文件：

```bash
uv run ruff check <changed_python_files>
uv run ruff format --check <changed_python_files>
git diff --check
```

不对整个仓库做全量格式化。

## 4. P0：基线记录
### 4.1 目标

在修改源码前记录当前 JAX π0 的行为和参数结构，为后续 identity 与回归测试提供参考。

### 4.2 修改文件

```text
无生产代码修改
可新增临时分析测试或记录文件，但不进入最终代码
```

### 4.3 执行内容

记录：

1. 当前 commit。
2. 当前 worktree 中已有用户修改。
3. dummy Gemma 的参数树路径。
4. dummy π0 的 `compute_loss()` 输出 shape。
5. dummy π0 的 `sample_actions()` 输出 shape。
6. 当前 freeze filter 在普通 π0 和 action-expert LoRA 下的结果。
7. 固定随机种子下的基线输出摘要。

建议保存的摘要：

```text
loss shape
loss mean/std
sample shape
sample mean/std
parameter count
frozen/trainable parameter count
```

### 4.4 验证命令

CPU：

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_test.py -q
```

GPU：

```bash
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

`model_test.py::test_pi0_model` 会创建完整默认 π0，不应作为普通 CPU 快速测试。CPU 上优先使用 abstract parameter-tree 测试和后续新增的 dummy Gemma 测试。

### 4.5 通过标准

```text
当前基线测试通过
参数树已记录
没有把已有用户修改误认为本任务修改
```

### 4.6 停止条件

如果基线本身失败，应先记录已有失败，不在本阶段修复无关问题，也不把其归因于 OCAE。

### 4.7 建议提交

本阶段通常不提交。

## 5. P1：训练前向拆分为 Prefix Cache + Suffix

### 5.1 目标

只重构 `Pi0.compute_loss()` 的执行顺序，不增加任何 OCAE 参数或行为。

从：

```text
[prefix, suffix] -> joint forward
```

改为：

```text
prefix -> prefix_out + KV cache
suffix -> forward with KV cache
```

### 5.2 修改文件

```text
src/openpi/models/pi0.py
src/openpi/models/pi0_split_forward_test.py  # 新增
```

### 5.3 最小修改内容

1. 保持 `embed_prefix()` 当前 tuple 返回不变。
2. 保持 `embed_suffix()` 不变。
3. 在 `compute_loss()` 中执行 prefix-only forward。
4. 复用 `sample_actions()` 当前的 suffix mask、cross-prefix mask 和 positions 逻辑。
5. suffix forward 使用 prefix KV cache。
6. flow head和 loss 计算完全不变。
7. 暂时不要增加 Prefix dataclass、Overview module 或 conditioning 参数。

### 5.4 不在本阶段做

```text
不修改 gemma.py
不增加新配置
不修改 checkpoint loader
不修改 freeze filter
不重构 sample_actions()
```

### 5.5 单元验证

在 dummy Gemma 测试中保留一个只用于比较的原 joint forward 计算，使用同一个模型参数、输入和 mask 比较：

```text
joint suffix_out
split suffix_out
joint v_t
split v_t
joint loss
split loss
```

必须覆盖带 prefix padding 的 batch，因为 suffix position offset 依赖有效 prefix token 数。

### 5.6 关键断言

```python
assert joint_suffix.shape == split_suffix.shape
assert joint_v.shape == split_v.shape
assert joint_loss.shape == split_loss.shape
assert allclose(joint_suffix, split_suffix)
assert allclose(joint_v, split_v)
assert allclose(joint_loss, split_loss)
```

float32 dummy Gemma 优先要求 exact equality；bfloat16 使用明确容差。

### 5.7 验证命令

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_split_forward_test.py -q
```

完整 Pi0 回归放到 GPU：

```bash
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

### 5.8 通过标准

```text
dummy float32 joint/split 输出完全相等或最大差值为 0
带 padding 的 mask/positions 测试通过
compute_loss shape 不变
sample_actions 未受影响
无新增参数
```

### 5.9 常见失败定位

| 现象 | 优先检查 |
|---|---|
| suffix 输出差异 | suffix full mask 是否包含 prefix mask |
| padding batch 才失败 | suffix positions 是否使用有效 prefix 长度 |
| cache length 不符 | prefix cache token 轴与 mask 长度 |
| state token 行为变化 | suffix ar_mask 是否保持原逻辑 |

### 5.10 停止条件

joint/split 未等价前，不进入 Prefix layout 和 OCAE 实现。

### 5.11 建议提交

```text
refactor(pi0): split training prefix and suffix forwards
```

## 6. P2：Prefix 返回结构与 Layout

### 6.1 目标

把 `embed_prefix()` 的返回值结构化，使后续能够从 `prefix_out` 精确恢复各 image view 与 language 片段。

本阶段仍不计算 OCAE context，也不改变模型输出。

### 6.2 修改文件

```text
src/openpi/models/overview_action_conditioning.py  # 新增 PrefixEmbeddings
src/openpi/models/pi0.py
src/openpi/models/prefix_layout_test.py            # 新增
```

### 6.3 数据结构

```python
@flax.struct.dataclass
class PrefixEmbeddings:
    tokens: Array
    input_mask: Array
    ar_mask: Array
    image_lengths: tuple[int, ...] = flax.struct.field(pytree_node=False)
    language_length: int = flax.struct.field(pytree_node=False)
```

### 6.4 修改内容

1. `embed_prefix()` 返回 `PrefixEmbeddings`。
2. `compute_loss()` 改用字段访问。
3. `sample_actions()` 改用字段访问。
4. image view 顺序必须与当前 `obs.images` 稳定顺序一致。
5. `image_lengths` 来自各 view 实际 token shape。
6. `language_length` 是 language segment 的静态序列长度，不是有效 token 数。

### 6.5 不在本阶段做

```text
不增加 encoder 参数
不修改 Gemma
不增加 controller
不修改 checkpoint/freeze
```

### 6.6 单元验证

构造多视角 token，验证：

1. 各 view slice 的起止位置正确。
2. language slice 紧接最后一个 image view。
3. 拼接回去后等于原始 `tokens`。
4. language padding 只影响 mask，不改变静态 layout。
5. image view mask 不改变 token segment 长度。
6. 结构可以经过 JAX tree flatten/unflatten。

### 6.7 回归验证

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/prefix_layout_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_split_forward_test.py -q
```

GPU 完整回归：

```bash
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

### 6.8 通过标准

```text
layout slice 全部正确
JAX pytree 行为正确
P1 joint/split 等价仍成立
模型参数树没有变化
```

### 6.9 常见失败定位

| 现象 | 优先检查 |
|---|---|
| JIT 不识别返回对象 | 是否使用 flax.struct.dataclass |
| slice 越界 | image_lengths 是否按实际 view 数生成 |
| language padding 错位 | language_length 与有效 mask 数是否混淆 |

### 6.10 建议提交

```text
refactor(pi0): carry static prefix layout metadata
```

## 7. P3：Overview、State 与 Zero-Gate Controller 独立模块

### 7.1 目标

实现 OCAE 的外部 conditioning 模块，但暂时不接入 Gemma attention。

本阶段验证：

```text
context pooling 正确
state encoder 正确
controller gate shape 正确
zero-init identity 条件成立
controller final projection 有梯度
```

### 7.2 修改文件

```text
src/openpi/models/overview_action_conditioning.py
src/openpi/models/overview_action_conditioning_test.py
src/openpi/models/pi0_config.py
```

### 7.3 新增组件

```text
OverviewActionConditioningConfig
OverviewContextEncoder
StateContextEncoder
ActionConditioningController
OverviewActionConditioning
```

### 7.4 配置

```python
@dataclasses.dataclass(frozen=True)
class OverviewActionConditioningConfig:
    enabled: bool = False
    rank: int = 16
    lora_alpha: float = 16.0
    target: Literal["q", "o", "q_o"] = "q_o"
    use_state_context: bool = True
```

此配置定义在 `overview_action_conditioning.py`，由 `pi0_config.py` 和后续的 `gemma.py` 共用，避免循环依赖。

### 7.5 Overview pooling

输入：

```text
prefix_out: [B,P,D_vlm]
PrefixEmbeddings layout
image masks
language mask
```

处理：

```text
每个 view 单独 masked mean
language 单独 masked mean
缺失 view pooled vector 置零
拼接 view validity bits
LayerNorm + MLP
```

输出：

```text
c: [B,D_action]
```

### 7.6 State 与 controller

```text
s = StateEncoder(obs.state)                     # [B,D_action]
raw = ZeroInitDense(MLP([LN(c),LN(s)]))
raw -> [B,L,2,N]
g_q/g_o = tanh(raw split)                       # [B,L,N]
```

当 `use_state_context=False` 时，以同 shape 的零 state context 输入 controller，保持参数和 shape 稳定。

### 7.7 单元验证

#### Mask 测试

1. 修改 padding hidden state 不应改变 language pooled context。
2. 修改 masked view hidden state 不应改变 overview context。
3. 所有 view 无效时不得产生 NaN。
4. 不同 view 交换内容应改变拼接后的 overview input。

#### Shape 测试

```text
c: [B,D_action]
s: [B,D_action]
g_q/g_o: [B,L,N]
```

#### 初始化测试

```text
初始 g_q == 0
初始 g_o == 0
所有输出 finite
```

#### 梯度测试

构造简单 surrogate loss：

```text
loss = sum(g_q * fixed_nonzero_tensor) + sum(g_o * fixed_nonzero_tensor)
```

断言 controller zero-init final projection 的 kernel 或 bias gradient 非零。

### 7.8 验证命令

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/overview_action_conditioning_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/prefix_layout_test.py -q
```

### 7.9 通过标准

```text
mask 和 layout 行为正确
zero gate 全零
controller final projection 梯度非零
所有 shape 与 dtype 正确
模块可 JIT
尚未改变 Pi0/Gemma 输出
```

### 7.10 常见失败定位

| 现象 | 优先检查 |
|---|---|
| masked view 仍影响结果 | pooling 前是否乘 mask、输出是否 where 置零 |
| 全 mask 出 NaN | denominator 是否 clip 到至少 1 |
| 初始 gate 非零 | controller 最后一层 kernel/bias 是否 zero init |
| final projection 无梯度 | surrogate loss 是否对 gate 有非零导数 |

### 7.11 停止条件

zero gate 或梯度测试失败时，不接入 Gemma。

### 7.12 建议提交

```text
feat(ocae): add contextual overview and zero-gate controller
```

## 8. P4：Q-only Conditional LoRA 接入

### 8.1 目标

完成第一条真正影响 action expert 的路径：

```text
contextualized prefix/state
-> per-layer/per-head q_gate
-> action expert conditional Q LoRA
```

暂时不实现 O LoRA。

### 8.2 修改文件

```text
src/openpi/models/gemma.py
src/openpi/models/pi0.py
src/openpi/models/pi0_config.py
src/openpi/models/conditional_action_lora_test.py  # 新增
src/openpi/models/model_test.py
```

### 8.3 Gemma 接口一次到位

为避免 P5 再次大改函数签名，本阶段同时把 Q/O gate 都传入 scan，但 O gate 暂不使用。

调用链：

```text
gemma.Module.__call__
-> nn.scan
-> Block.__call__
-> Attention.__call__
```

controller gate 转置：

```text
[B,L,N] -> [L,B,N]
```

scan：

```text
q_gate in_axes=0
o_gate in_axes=0
positions/mask/adarms_cond/deterministic=broadcast
```

### 8.4 Conditional Q 参数

只在 expert index 1 创建：

```text
A_q: [N,D_action,R]
B_q: [N,R,H]
```

A/B 均使用小的非零随机初始化；identity 只由 q_gate=0 保证。

### 8.5 Q 计算位置

```text
Q_base = q_einsum(x)
Q_low = einsum(x,A_q)
Q_low *= q_gate[:,None,:,None]
Q_delta = einsum(Q_low,B_q)
Q = Q_base + lora_scale * Q_delta
Q -> RoPE -> head scaling
```

必须在 RoPE 之前加入 delta。

### 8.6 Pi0 接入

1. `Pi0.__init__()` 创建 `OverviewActionConditioning`。
2. 创建 Gemma 时传入静态 OCAE config。
3. prefix-only pass 返回 `prefix_out` 和 KV cache。
4. 在 suffix pass 前计算 c/s/gates。
5. 训练和推理都把 gates 传入 Gemma。
6. 推理中 c/s/gates 位于 denoising loop 外。

### 8.7 Lazy init 与参数树

必须同步修改 `gemma.Module.init()`：

```text
dummy expert 1 input 非 None
dummy q_gate/o_gate shape = [L,1,N]
conditional Q 参数在 init 时创建
```

参数创建只能由静态配置控制：

```text
config.enabled and target includes q and expert_index == 1
```

不能由运行时 gate 是否为 None 控制。

### 8.8 必做测试

#### Zero-gate identity

同一个 augmented model、同一参数比较：

```text
conditional Q branch bypass
conditional Q branch enabled with q_gate=0
```

断言：

```text
Q 相等
attention output 相等
suffix hidden 相等
flow output 相等
固定 noise 的 sample_actions 相等
```

#### Prefix cache 不变

同一 augmented model 比较 OCAE bypass/enable：

```text
prefix cache K 相等
prefix cache V 相等
prefix_out 相等
```

#### 非零 gate 有效

手动传入固定非零 q_gate：

```text
Q_delta norm > 0
action expert suffix output 与 zero-gate 不同
prefix cache 仍相同
```

#### 梯度测试

首次 backward：

```text
controller final projection grad > 0
loss finite
```

一次 optimizer update 后再次 backward：

```text
A_q 或 B_q grad norm > 0
overview encoder grad norm > 0
state encoder grad norm > 0
```

#### Scan/init 测试

```text
参数树含 conditional_q_lora_1
参数第一维含 layer scan 轴
eager apply 通过
JIT apply 通过
prefix-only 与 suffix-only 调用都通过
```

### 8.9 验证命令

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/conditional_action_lora_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/overview_action_conditioning_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_split_forward_test.py -q
```

GPU：

```bash
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

### 8.10 通过标准

```text
zero-gate identity 成立
prefix KV 完全不变
非零 q_gate 能改变 suffix 输出
参数树稳定
梯度可达 Q LoRA 和 context modules
compute_loss/sample_actions 均可 JIT
```

### 8.11 常见失败定位

| 现象 | 优先检查 |
|---|---|
| init 有参数、apply 找不到 | 参数创建是否依赖 runtime gate |
| scan 报 axis 错误 | gate 是否转成 `[L,B,N]` |
| remat 参数错误 | `static_argnums` 是否随签名更新 |
| zero gate 不 identity | gate 是否在 delta 加到 Q 前完整乘入 |
| prefix cache 改变 | 是否误对 expert 0 或 prefix K/V 应用分支 |
| 首次所有梯度为零 | A/B 是否也被零初始化 |

### 8.12 停止条件

Q-only 未通过全部 identity、cache、gradient 和 JIT 验证时，不实现 O branch。

### 8.13 建议提交

```text
feat(ocae): condition action expert queries
```

## 9. P5：O-only 与 Q/O Conditional LoRA

### 9.1 目标

在已稳定的 Q-only 调用链上增加 O branch，并支持：

```text
target=q
target=o
target=q_o
```

### 9.2 修改文件

```text
src/openpi/models/gemma.py
src/openpi/models/conditional_action_lora_test.py
src/openpi/models/model_test.py
```

### 9.3 Conditional O 参数

只在 expert index 1 创建：

```text
A_o: [N,H,R]
B_o: [N,R,D_action]
```

### 9.4 O 计算位置

必须在 head 维仍存在时应用 gate：

```text
encoded: [B,T,N,H]
O_low = einsum(encoded,A_o)                     # [B,T,N,R]
O_low *= o_gate[:,None,:,None]
O_delta = einsum(O_low,B_o)                    # [B,T,D_action], sum over N
O = O_base + lora_scale * O_delta
```

禁止先执行 base O projection 得到 `[B,T,D_action]` 后再尝试施加 per-head gate。

### 9.5 参数存在性测试

```text
target=q   -> 只有 conditional_q_lora_1
target=o   -> 只有 conditional_o_lora_1
target=q_o -> 两者都有
```

### 9.6 Identity 与有效性测试

分别对 q/o/q_o 执行：

```text
zero gate identity
固定非零 gate 改变 suffix output
prefix cache 不变
JIT forward/backward
```

O-only 额外断言：

```text
base attention probabilities 不变
encoded attention value aggregation不变
只有 action expert output projection结果变化
```

### 9.7 验证命令

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/conditional_action_lora_test.py -q
```

GPU：

```bash
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

### 9.8 通过标准

```text
q/o/q_o 三种 target 均正确创建参数
三种 target 的 zero-gate identity 都成立
非零 gate 都能改变 suffix 输出
O branch 不改变 attention logits/probs
prefix KV 不变
```

### 9.9 常见失败定位

| 现象 | 优先检查 |
|---|---|
| o_gate 无效果 | 是否在 head sum 前应用 |
| 输出 shape 错误 | O_delta einsum 是否正确求和 N 轴 |
| target=o 仍有 Q 参数 | 参数创建条件是否只看静态 target |
| attention probs 改变 | 是否误把 O branch 放到 logits/encoded 之前 |

### 9.10 建议提交

```text
feat(ocae): condition action expert attention outputs
```

## 10. P6：Checkpoint Loader 与 Freeze Filter

### 10.1 目标

确保 base π0 checkpoint 可以加载到 OCAE 模型，并且阶段 1 只训练新增 OCAE 参数。

### 10.2 修改文件

```text
src/openpi/training/weight_loaders.py
src/openpi/training/weight_loaders_test.py  # 若已有则追加，否则新增
src/openpi/models/pi0_config.py
src/openpi/models/pi0_test.py
```

### 10.3 Checkpoint loader

允许 base checkpoint 缺失：

```text
overview_action_conditioning/...
conditional_q_lora_1/...
conditional_o_lora_1/...
```

保持精确 missing regex，不使用无条件 `.*`。

### 10.4 Loader 测试

使用人工小参数树验证：

1. OCAE 参数缺失时从 reference params 合并。
2. 普通 base 参数缺失时仍然失败。
3. shape 不匹配时仍然失败。
4. dtype 转换行为保持原样。

真实 base checkpoint restore 标记为 manual/GPU 测试，避免普通 CPU 单元测试下载大文件。

### 10.5 Freeze filter

V1 OCAE-only trainable regex：

```text
.*(overview_action_conditioning|conditional_q_lora_1|conditional_o_lora_1).*
```

对应：

```text
freeze_filter = Not(trainable_regex)
```

### 10.6 Freeze 参数树测试

对 `target=q/o/q_o` 分别验证：

```text
所有实际存在的 OCAE 参数可训练
所有非 OCAE 参数冻结
action_out_proj 冻结
state_proj 冻结
VLM expert 冻结
action expert base 参数冻结
```

同时验证 OCAE disabled 时现有 LoRA freeze 行为不变。

### 10.7 参数量报告

测试或工具函数应能输出：

```text
total params
trainable params
overview/state/controller params
conditional Q params
conditional O params
```

参数量将用于后续 matched baseline 报告。

### 10.8 验证命令

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/training/weight_loaders_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_test.py -q
```

### 10.9 通过标准

```text
base 参数树可合并到 OCAE 模型
非允许参数缺失仍报错
只有 OCAE 参数可训练
OCAE disabled 的原有行为不变
参数量可准确统计
```

### 10.10 常见失败定位

| 现象 | 优先检查 |
|---|---|
| restore 缺新参数 | 实际路径是否匹配 missing regex |
| 普通参数也被放过 | regex 是否过宽 |
| OCAE 参数被冻结 | trainable regex 是否覆盖 scan 内路径 |
| base 参数意外可训练 | freeze filter 的 Not/All 语义是否写反 |

### 10.11 建议提交

```text
feat(ocae): support base checkpoints and ocae-only freezing
```

## 11. P7：端到端回归与训练配置

### 11.1 目标

把已经独立通过的模块组合成完整 JAX π0 OCAE 配置，验证训练、推理、保存和恢复链路闭合。

### 11.2 修改文件

```text
src/openpi/training/config.py
src/openpi/models/model_test.py
src/openpi/models/overview_action_conditioning_test.py
必要时新增 scripts/inspect_ocae_model.py
```

### 11.3 新增训练配置

建议增加独立配置，例如：

```text
pi0_ocaev1_debug
pi0_ocaev1_<dataset>
```

配置要求：

```text
pi05=False
paligemma_variant 不带 _lora
action_expert_variant 不带 _lora
overview_action_conditioning.enabled=True
target=q_o
rank=16
lora_alpha=16
freeze_filter=OCAE-only
base π0 checkpoint loader
```

### 11.4 端到端测试

1. 创建模型。
2. 加载 base checkpoint。
3. 执行 `compute_loss()`。
4. 执行一次 gradient update。
5. 执行 `sample_actions()`。
6. 保存 checkpoint。
7. 恢复 checkpoint。
8. 固定 noise 再次采样并比较恢复前后输出。

### 11.5 回归测试范围

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/overview_action_conditioning_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/conditional_action_lora_test.py -q
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/pi0_split_forward_test.py -q
```

GPU：

```bash
uv run pytest src/openpi/models/model_test.py::test_pi0_model -q
```

并执行现有非 OCAE π0 测试，确认 disabled 模式没有回归。

### 11.6 性能测量

固定：

```text
batch size
action horizon
num flow steps
image view 数
dtype
设备
```

比较：

```text
base π0 sample latency
OCAE zero-gate sample latency
OCAE trained/nonzero-gate sample latency
peak memory
compile time
```

Overview encoder 必须只在 denoising loop 外执行一次。

### 11.7 通过标准

```text
完整 train/inference JIT 通过
base checkpoint 能加载
checkpoint 保存/恢复闭合
恢复前后固定 noise 输出一致
disabled 模式无回归
额外延迟和显存已测量
```

### 11.8 建议提交

```text
test(ocae): add end-to-end jax pi0 validation
```

## 12. P8：Smoke Training

### 12.1 目标

用最小训练预算验证模块实际能从 zero gate 开始学习，不追求任务性能结论。

### 12.2 建议规模

```text
小数据子集
几十到几百 step
固定 1 个 seed
较小 batch
频繁记录 gate/delta/gradient
```

### 12.3 必须记录

每个日志周期记录：

```text
loss
global grad norm
overview encoder grad norm
state encoder grad norm
controller final projection grad norm
Q LoRA A/B grad norm
O LoRA A/B grad norm
mean/std/max abs of g_q/g_o
||Q_delta|| / ||Q_base||
||O_delta|| / ||O_base||
step time
peak memory
```

### 12.4 预期现象

初始化：

```text
g_q/g_o = 0
Q/O delta 对输出贡献 = 0
模型输出等于 base
```

前几步：

```text
controller final projection 先开始变化
gate 从 0 离开
随后 Q/O A/B、overview/state encoder 获得有效梯度
loss 保持 finite
```

### 12.5 失败条件

出现任一情况即停止：

```text
loss NaN/Inf
gate 始终严格为 0
第一次更新后 Q/O A/B 仍长期无梯度
Q/O delta ratio 快速爆炸
只有非 OCAE 参数发生更新
checkpoint 无法恢复
```

### 12.6 通过标准

```text
连续训练完成预定 step
loss finite
gate 已离开 0
OCAE 所有目标模块最终都有非零梯度
冻结参数未改变
checkpoint 可恢复
性能开销在可接受范围
```

### 12.7 建议提交

Smoke training 结果和配置可以单独提交：

```text
chore(ocae): add smoke training configuration
```

训练 checkpoint 和大日志不提交到源码仓库。

## 13. P9：核心实验顺序

### 13.1 第一批实验

只跑最能判断路线是否成立的设置：

| 顺序 | 实验 | 判断目标 |
|---|---|---|
| 1 | E0 原始 action-expert LoRA | 常用基线 |
| 2 | E1 Static Q/O LoRA | 排除新增低秩容量 |
| 3 | E2 Constant-input OCAE | 排除 controller 参数量 |
| 4 | E5 OCAE Q-only | Q 条件化收益 |
| 5 | E6 OCAE O-only | O 条件化收益 |
| 6 | E7 OCAE Q/O | 主方法 |

### 13.2 第一批继续条件

只有满足以下趋势才继续完整消融：

```text
E7 > E1
E7 > E2
E5/E6 至少一个显示稳定正向贡献
rollout success 与多个 seed 趋势一致
```

### 13.3 第二批消融

```text
no visual
no language
no state
shuffled images/views
shuffled language
shuffled state
image-token gate
action-suffix gated adapter
```

### 13.4 实验停止条件

```text
E7 不优于 E1：收益主要是低秩容量
E7 不优于 E2：conditioning 近似常数
视觉消融无影响：overview 没有使用视觉
只有 validation loss 改善、rollout 无收益
延迟或显存超出部署边界
```

## 14. 阶段验收总表

| 阶段 | 核心产物 | 必须通过 |
|---|---|---|
| P0 | 基线记录 | 当前测试和参数树已记录 |
| P1 | split training forward | joint/split 等价 |
| P2 | Prefix layout | slice/mask/JAX pytree 正确 |
| P3 | Context/controller | zero gate + 非零 controller gradient |
| P4 | Q-only | identity、KV 不变、gradient、JIT |
| P5 | O/QO | head-wise O 正确、三种 target 正确 |
| P6 | loader/freeze | base restore、仅 OCAE 可训练 |
| P7 | 端到端链路 | train/sample/save/restore 闭合 |
| P8 | smoke training | gate 学习、loss finite、冻结参数不变 |
| P9 | 实验 | 优于 matched baselines |

## 15. 推荐提交序列

```text
1. refactor(pi0): split training prefix and suffix forwards
2. refactor(pi0): carry static prefix layout metadata
3. feat(ocae): add contextual overview and zero-gate controller
4. feat(ocae): condition action expert queries
5. feat(ocae): condition action expert attention outputs
6. feat(ocae): support base checkpoints and ocae-only freezing
7. test(ocae): add end-to-end jax pi0 validation
8. chore(ocae): add smoke training configuration
```

每个提交都应满足：

```text
本阶段测试通过
上阶段测试继续通过
ruff check/format check 通过
git diff --check 通过
不包含无关格式化或重构
```

## 16. 最终完成定义

V1 工程实现只有在以下条件全部满足时才算完成：

```text
训练前向拆分与原行为等价
context 来自 contextualized prefix_out
zero-gate 与 base 模型等价
prefix KV cache 完全不受 OCAE 影响
Q/O conditional LoRA 参数按 layer/head 正确创建
controller 与 LoRA 均有有效梯度
base checkpoint 可加载
只有 OCAE 参数可训练
JAX compute_loss/sample_actions 均可 JIT
checkpoint 保存与恢复闭合
smoke training 稳定
性能开销已测量
```

完成工程实现不等于研究结论成立。研究结论还必须满足：

```text
OCAE Q/O 优于 Static Q/O LoRA
OCAE Q/O 优于 Constant-input OCAE
视觉、语言和状态消融符合预期
闭环 rollout success 稳定提升
```
