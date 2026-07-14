# OCAE V1/V2 讲解稿

## 0. 开场：我们到底要解决什么问题

这套方案想解决的不是“让视觉模型再看一遍图像”，也不是普通的 language grounding 或 heatmap 增强。

π0 已经有很强的视觉-语言 backbone，它的问题在于：视觉和语言 prefix 被编码好之后，Action Expert 在生成动作时是用一套固定的 attention 读取这些 prefix 信息。也就是说，同一组视觉-语言 token，在不同任务、不同机器人状态下，Action Expert 的读取方式并没有被显式地动态校准。

我们的目标是借鉴 OverLoCK 的思想：

```text
先纵观全局 -> 形成 context prior -> 反馈给局部计算
```

但在 π0 里，这个“局部计算”不应该优先落在视觉 token 上，而应该落在 Action Expert 上。因为最终决定动作的是 Action Expert，而不是视觉 backbone 本身。

所以核心问题变成：

```text
面对当前图像、语言和机器人状态，Action Expert 应该怎样读取视觉-语言 prefix？
```

这就是 OCAE：Overview-Conditioned Action Expert。

## 1. 代码已有的前提设定

先讲仓库当前代码是怎么工作的。理解这个前提很重要，因为我们的 V1/V2 都是在这个结构上设计的。

### 1.1 π0 由两类 token 组成

当前 `src/openpi/models/pi0.py` 中，π0 的输入大体分成两部分：

```text
prefix: 图像 tokens + 语言 tokens
suffix: state token + noisy action tokens
```

prefix 在 `Pi0.embed_prefix()` 中生成：

```text
obs.images[name] -> self.PaliGemma.img(...) -> image_tokens
obs.tokenized_prompt -> self.PaliGemma.llm(..., method="embed") -> language_tokens
```

suffix 在 `Pi0.embed_suffix()` 中生成：

```text
obs.state -> state_proj -> state token
noisy_actions + timestep -> action expert tokens
```

这里要注意：当前代码中的 pi0 state token 在 suffix 中，而不是 prefix cache 中。这个细节会影响 V2 设计。

### 1.2 推理时 prefix 会被缓存

当前 `sample_actions()` 的核心流程是：

```text
1. 先算 prefix_tokens
2. 用 prefix_tokens 跑一次 LLM，得到 kv_cache
3. 进入 10 步 flow denoising
4. 每一步只重新计算 suffix/action tokens
5. suffix 通过 kv_cache 读取 prefix 信息
```

直观理解：

```text
图像和语言先读一遍，形成缓存记忆；
动作生成的每一步，只带着 noisy action 去读这份记忆。
```

这就是为什么我们不能随便改 prefix。只要 prefix 内容在每个 denoising step 中变化，就会破坏 KV cache，推理成本会明显增加。

### 1.3 Gemma attention 支持两个 expert

当前 `src/openpi/models/gemma.py` 中，`Module` 接收一个 expert token 列表：

```text
[expert_0_tokens, expert_1_tokens]
```

其中：

```text
expert 0: PaliGemma / VLM backbone
expert 1: Action Expert
```

在 attention 里，每个 expert 有自己的 Q/K/V projection。然后所有 token 的 K/V 会拼接起来，action tokens 可以通过 attention 读取 prefix tokens。

这给我们的设计留下了一个关键入口：

```text
不改 VLM expert；
只改 Action Expert 侧的 Q/O 或后续 modulation；
让 action token 用不同方式读取同一份 prefix cache。
```

## 2. 方案总览：为什么分 V1 和 V2

PDF 方案里建议把 context prior token 插入序列：

```text
[V, L, C, q, A_tau]
```

这个设计在论文逻辑上是合理的，但对当前代码来说改动比较大。因为当前 prefix cache 只缓存 `[V, L]`，而 PDF 方案希望缓存 `[V, L, C, q]`。这意味着我们要重构 prefix/suffix 边界、expert routing 和 mask。

所以这里分成两个阶段：

```text
V1: 不插入 context token，只把 context prior 作为 Action Expert 的 conditioning 信号。
V2: 按 PDF 思路插入 context token，并让 [V, L, C, q] 一起进入 prefix cache。
```

分阶段的原因是：

1. V1 改动小，能先验证核心假设：overview context 是否真的能改善 action expert。
2. V2 更完整，更接近 PDF 和 OverLoCK 的结构叙事，但工程风险更高。
3. 如果 V1 都不能超过普通 gate/adapter，V2 没有必要急着做。

## 3. V1：Overview context 作为 Action Expert conditioning

### 3.1 V1 的一句话

V1 不改变 token 序列，不新增 prefix token，只额外生成一个 `context prior`，用它动态调制 Action Expert 的 Q/O 或 action-side adapter。

```text
[V, L] -> Overview-Net -> c
q -> State Context -> s
[c, s] -> Conditioning Controller
Action Expert Q/O -> conditional low-rank modulation
```

### 3.2 V1 为什么这样设置

V1 的设计原则是：保留 π0 最稳定的部分，只改最该改的地方。

不改的部分：

```text
不改 SigLIP image encoder
不改 PaliGemma language backbone
不改变 prefix token 数量
不改变 prefix KV cache
不改变 blockwise attention 的基本逻辑
```

改的部分：

```text
Action Expert 读取 prefix 的方式
```

为什么这样改？

因为动作生成真正依赖的是 Action Expert。视觉 backbone 已经提供了图像和语言的语义记忆，但不同任务下，Action Expert 应该用不同 query 去读这份记忆。

这就像同一本说明书，不同工人会带着不同问题去查不同段落。我们不是重写说明书，而是改变工人提问和整理答案的方式。

### 3.3 V1 的结构

V1 分成三条支路。

第一条是原始 prefix 支路：

```text
images -> PaliGemma image encoder -> image tokens
language -> Gemma embedding -> language tokens
[image tokens, language tokens] -> prefix KV cache
```

第二条是 overview 支路：

```text
image tokens + language tokens -> Overview-Net -> scene-task context c
```

这里的 `c` 主要描述任务和场景，例如“目标是什么、场景里有哪些关系、语言重点是什么”。

第三条是 state 支路：

```text
obs.state -> State Context Encoder -> state context s
```

这里的 `s` 描述机器人当前状态，例如末端执行器位置、夹爪状态、动作阶段。

最后合并：

```text
[c, s] -> Conditioning Controller -> q_scale / o_scale
```

这些 scale 用于调制 Action Expert。

### 3.4 为什么 Overview-Net 不直接混入 state

PDF 中有一个合理提醒：Overview-Net 初始最好只用 `[V, L]`，不要太早混入 `q`。原因是如果 state 太强，context prior 可能学成机器人姿态 shortcut，而不是场景和任务先验。

但完全不用 state 也不合理，因为同一张图、同一句话，在不同机器人姿态下下一步动作不同。

所以 V1 拆成：

```text
c = OverviewNet(V, L)
s = StateEncoder(q)
condition = Controller(c, s)
```

这样做的作用是：

1. `c` 保持为场景-任务 prior。
2. `s` 单独表达机器人状态。
3. 最后由 controller 决定 action expert 如何被调制。

这比直接把 `[V, L, q]` 全部扔进一个 MLP 更可解释，也更容易做消融。

### 3.5 V1 具体怎么调制 Action Expert

首选调制 Q 和 O。

attention 可以这样理解：

```text
Q: action token 在问什么
K/V: prefix memory 里有什么
O: 读完之后怎么写回 action 表征
```

V1 不改 prefix K/V，只改 action 侧 Q/O：

```text
Q_base = x W_q
Q_delta = (x A_q) B_q
Q = Q_base + alpha_q * scale_q(c, s) * Q_delta
```

输出侧：

```text
O_base = attn_out W_o
O_delta = (attn_out A_o) B_o
O = O_base + alpha_o * scale_o(c, s) * O_delta
```

这里的 `A_q B_q` 和 `A_o B_o` 是低秩分支，类似 LoRA，但它不是固定 LoRA，而是由 overview context 和 state context 动态控制强弱。

### 3.6 为什么不直接动态生成完整 QKV

不建议：

```text
W_q' = HyperNet(c)
W_k' = HyperNet(c)
W_v' = HyperNet(c)
```

原因：

1. 参数量和计算量太大。
2. 训练不稳定。
3. 很难保持 checkpoint 初始行为。
4. 容易破坏已有 attention 分布。

所以 V1 用低秩、残差、alpha=0 初始化：

```text
启用模块时，初始输出 == 原始 π0
训练后，模型逐步学会使用 overview conditioning
```

这对研究和工程都很重要，因为 identity 验证过不了，后面所有提升都没法归因。

### 3.7 V1 的创新性

V1 的创新不在于“多了一个 adapter”，而在于：

```text
adapter 的作用位置和条件来源不同。
```

普通方案：

```text
language -> heatmap -> enhance image tokens
```

V1：

```text
image/language overview + robot state -> condition action expert query/output
```

这表达的是：

```text
不同任务和状态下，action expert 应该用不同方式读取同一份视觉-语言 prefix。
```

这比 image-token gate 更贴近 π0 的结构特点，因为 π0 本来就有一个专门的 Action Expert。

### 3.8 V1 的实现流程

训练时：

```text
1. preprocess observation
2. image/lang -> prefix tokens
3. prefix tokens -> overview context c
4. state -> state context s
5. [c, s] -> q/o conditioning scale
6. noisy actions + timestep -> suffix action tokens
7. LLM forward with prefix + suffix
8. Action Expert 内部使用 conditioning 调制 Q/O
9. 输出 vector field
10. flow matching loss
```

推理时：

```text
1. image/lang -> prefix tokens
2. prefix tokens -> context c
3. state -> context s
4. [c, s] -> conditioning scale
5. prefix tokens -> KV cache
6. 10 步 flow denoising
7. 每一步 suffix action tokens 使用同一个 conditioning scale
```

关键点：

```text
Overview-Net 每个控制周期只算一次；
conditioning scale 每个控制周期只算一次；
10 步 denoising 复用同一份 prefix cache 和 conditioning。
```

### 3.9 V1 的风险

工程风险：

1. 需要给 `gemma.Module -> Block -> Attention` 传递 conditioning。
2. `gemma.Module` 使用 `nn.scan`，每层参数和 conditioning shape 要小心。
3. `embed_prefix()` 返回值变化会影响 `compute_loss()` 和 `sample_actions()`。
4. checkpoint loader 和 freeze filter 要允许新增参数。

研究风险：

1. `c` 可能只学语言，不看视觉。
2. `s` 可能太强，导致 state shortcut。
3. q/o scale 可能学成常数，说明没有真正任务条件化。
4. loss 下降不一定带来 rollout 成功率提升。

因此 V1 必须做诊断：

```text
same image + different language -> scale should change
same image + same language + different state -> scale should change
shuffled language/state -> performance should drop
```

### 3.10 V1 的作用

V1 的主要作用是快速验证核心假设：

```text
overview context 是否能帮助 action expert 更好地生成动作？
```

如果 V1 成立，就说明 OverLoCK 的 local-global 思想迁移到 π0 是有价值的，而且价值不只是视觉增强，而是动作专家的条件化读取。

## 4. V2：Context token 进入 prefix cache

### 4.1 V2 的一句话

V2 更接近 PDF 的完整设想：把 Overview-Net 生成的 `C_t` 作为真正的 context tokens 插入序列，并让它们与 state 一起进入 prefix cache。

目标序列：

```text
B1 = [V, L]
B2 = [C]
B3 = [q]
B4 = [A_tau]
```

推理时缓存：

```text
cached prefix = [V, L, C, q]
```

每个 flow step 只重算：

```text
suffix = [A_tau]
```

### 4.2 V2 为什么这样设置

V1 中，`C_t` 是一个外部 conditioning 信号。它不进入 Transformer token 序列。

V2 中，`C_t` 成为 action expert 可以直接 attend 的 token。这样更接近 OverLoCK：

```text
Overview-Net 产生 context prior；
Focus/action path 在网络内部反复读取这个 prior。
```

这也更接近 PDF 中的 local-global relation：

```text
action token <-> context token
```

V2 的好处是结构叙事更完整：

1. `C_t` 是显式 token，可以被 attention 读取。
2. Action token 可以直接与 context token 建立关系。
3. 可以可视化 action token 对 context token 的 attention。
4. `C_t` 在控制周期内固定，可以进入 KV cache。

### 4.3 V2 和当前代码的冲突点

当前代码缓存的是：

```text
llm([prefix_tokens, None])
```

其中 `prefix_tokens` 是 VLM expert 的 `[V,L]`。

V2 希望缓存：

```text
llm([vlm_prefix_tokens, action_prefix_tokens])
```

其中：

```text
vlm_prefix_tokens = [V, L]
action_prefix_tokens = [C, q]
```

这意味着要重构当前接口：

```text
embed_prefix_vlm() -> [V, L]
embed_prefix_action() -> [C, q]
embed_suffix_action() -> [A_tau]
```

这就是 V2 不适合作为第一版的原因。它不是理论上不可行，而是比 V1 更深地改了 token 边界和 cache 逻辑。

### 4.4 V2 的 attention mask

V2 需要四块 mask：

```text
B1 = [V, L]
B2 = [C]
B3 = [q]
B4 = [A_tau]
```

规则：

```text
B1 只能看 B1
B2 可以看 B1, B2
B3 可以看 B1, B2, B3
B4 可以看 B1, B2, B3, B4
```

为什么这样设置？

1. 保持 PaliGemma 预训练输入 `[V,L]` 不看未来新 token，减少分布偏移。
2. `C` 来自 `[V,L]`，可以看 `[V,L]` 和自己。
3. `q` 是机器人当前状态，可以看感知和 context。
4. `A_tau` 是动作生成 token，可以看全部条件。

用当前 `make_attn_mask()` 可以表达这种 blockwise causal mask。关键是正确设置 `ar_mask` 的 block 起点。

### 4.5 V2 中 `C_t` 应该 route 到哪里

建议：

```text
C_t route 到 action expert
```

原因：

1. `C_t` 是新增机器人控制侧 token，不是 PaliGemma 预训练中的图像或语言 token。
2. π0 的设计本来就是：图像/语言走 VLM expert，机器人 state/action 走 Action Expert。
3. 如果把 `C_t` 放进 VLM expert，会增加破坏预训练分布的风险。

但这也带来工程要求：

```text
prefix pass 中必须同时有 expert 0 tokens 和 expert 1 tokens。
```

### 4.6 V2 的实现流程

训练时：

```text
1. image/lang -> VLM prefix tokens [V,L]
2. [V,L] -> Overview-Net -> C
3. state -> q token
4. noisy actions + timestep -> A_tau tokens
5. 构造 expert inputs:
   expert 0: [V,L]
   expert 1: [C,q,A_tau]
6. 构造四块 blockwise mask
7. LLM forward
8. action outputs -> flow matching loss
```

推理时：

```text
1. image/lang -> [V,L]
2. [V,L] -> C
3. state -> q
4. prefix pass:
   expert 0: [V,L]
   expert 1: [C,q]
   -> kv_cache
5. denoising loop:
   expert 0: None
   expert 1: [A_tau]
   use kv_cache
6. output action chunk
```

### 4.7 V2 的创新性

V2 的创新性比 V1 更显式：

```text
context prior 不只是外部条件向量，
而是变成 action expert 内部可读取、可缓存、可解释的 context tokens。
```

它更容易讲成 OverLoCK 迁移：

```text
Overview-Net 生成 context prior tokens；
Action Expert 作为 Focus-Net；
action tokens 与 context tokens 建立局部-全局关系；
flow matching head 输出动作场。
```

同时，V2 不是普通 grounding，因为 `C_t` 的作用不是输出区域 mask，而是作为 Action Expert 的内部计算条件。

### 4.8 V2 的风险

工程风险：

1. 需要重构 prefix/suffix 划分。
2. 需要让 prefix cache 同时包含 expert 0 和 expert 1 token。
3. 需要重新处理 positions 和 attention mask。
4. state token 从 suffix 移到 prefix 后，要确认训练和采样行为一致。
5. 如果 context token 数过多，会增加 prefix forward 和 attention 成本。

研究风险：

1. `C_t` 可能被 action tokens 忽略。
2. `C_t` 可能只复制语言 embedding，没有形成视觉-语言 context。
3. context token 插入可能带来收益，但收益来自 token 数增加而非机制本身。
4. 如果没有严格对照，很难证明 V2 优于普通 extra tokens。

所以 V2 必须做对照：

```text
真实 C_t vs random C_t
真实 C_t vs learned constant tokens
真实 C_t vs repeated global pooled tokens
真实 C_t + modulation vs C_t only
K=1 vs K=4/8
```

### 4.9 V2 的作用

V2 的主要作用是把 V1 的外部 conditioning 推进成 Transformer 内部结构：

```text
V1 证明 overview context 有用；
V2 证明显式 context tokens 能进一步改善 Action Expert 的动作生成。
```

如果 V2 成立，论文叙事会更强，因为它能展示：

```text
context prior token 被 action token 动态读取，
并且这种读取方式与任务、状态和动作阶段相关。
```

## 5. V1 和 V2 的关系

可以这样讲：

```text
V1 是低风险验证版。
V2 是结构完整增强版。
```

V1 不插 token，主要验证：

```text
overview context 能否作为 conditioning 改善 Action Expert？
```

V2 插入 token，主要验证：

```text
context prior 作为显式 prefix token 后，Action Expert 是否能更有效地进行 local-global reasoning？
```

不要把 V1 说成“简化版而已”。V1 的价值是明确的：它以最小工程改动验证最核心的机制。

也不要一开始就做 V2。因为如果 V2 失败，我们很难判断是机制无效，还是 token/mask/cache 改动引入了训练难度。V1 是必要的归因步骤。

## 6. 推荐讲法总结

可以用下面这段作为汇报总结：

```text
我们的方案不是给图像 token 做普通的语言热力图增强，而是把 OverLoCK 的 overview-to-focus 思想迁移到 π0 的 Action Expert。

当前 openpi 中，图像和语言 prefix 会被缓存，动作生成阶段只反复计算 state/action suffix。因此最稳的第一版 V1 不改变 token 序列，而是从视觉-语言 prefix 中生成 scene-task context，再结合机器人 state 产生 conditioning 信号，用于动态调制 Action Expert 的 Q/O 低秩分支。这样可以保持 prefix KV cache 不变，同时让 action token 在不同任务和状态下以不同方式读取同一份视觉-语言记忆。

在 V1 证明有效后，V2 再进一步引入 PDF 中的 context prior tokens，把序列扩展为 [V,L,C,q,A]，并让 [V,L,C,q] 进入 prefix cache。这样 context prior 不只是外部调制向量，而成为 Action Expert 内部可 attend、可解释、可缓存的 token。V2 更接近 OverLoCK 的结构叙事，但需要重构 prefix/suffix 边界、expert routing 和 blockwise mask，因此应作为第二阶段。
```

## 7. 最小实验闭环

最后一定要强调实验闭环。没有这些对照，方案讲得再好也很难证明创新有效。

V1 必做：

```text
E0: baseline π0
E1: OCAE alpha=0, 验证 identity
E2: image-token gate
E3: action-token gated adapter
E4: Q-only
E5: O-only
E6: Q/O
E7: no-language
E8: no-state
E9: shuffled-language
E10: shuffled-state
```

V2 必做：

```text
E0: V1 best
E1: add C_t tokens but no modulation
E2: random C_t tokens
E3: learned constant tokens
E4: repeated global pooled tokens
E5: real C_t tokens
E6: real C_t + local-global modulation
```

判断标准：

```text
V1 要证明 conditioning 本身有效；
V2 要证明显式 context tokens 比普通 extra tokens 有效；
两者都要证明收益不是来自参数量或 token 数增加。
```
