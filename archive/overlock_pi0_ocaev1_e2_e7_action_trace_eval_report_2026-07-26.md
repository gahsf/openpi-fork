# E2/E7 LIBERO 单次全任务动作轨迹测评报告

## 1. 测评协议

执行时间：2026-07-25 UTC

模型：

- E2 constant Q/O：`pi0_libero_e2_constant_qo`，checkpoint `seed42_30k_resume_from3000_20260724/29999`
- E7 sample-conditioned Q/O：`pi0_libero_e7_ocae_qo`，checkpoint `seed42_30k_20260723/29999`

公共协议：

- suites：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`
- 每 suite 10 个 task，每个 task 1 次，共每模型 40 episodes
- seed：7
- resize：224
- replan steps：5
- 视频：开启
- 每步记录 policy 输入的 8D state、最终送入 `env.step` 的 7D action，以及重规划时的 5x7 planned actions

结果目录：

```text
/data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725
```

日志目录：

```text
/data1/gqy/logs/ocaev1/action_trace_smoke_seed42_20260725
```

## 2. 最终成功率

| 模型 | Spatial | Object | Goal | LIBERO-10 | 总计 |
|---|---:|---:|---:|---:|---:|
| E2 constant Q/O | 0/10 | 0/10 | 0/10 | 0/10 | 0/40（0%） |
| E7 sample Q/O | 0/10 | 1/10 | 2/10 | 0/10 | 3/40（7.5%） |

E7 成功任务：

- Object task 8：`pick up the chocolate pudding and place it in the basket`，162 steps
- Goal task 7：`turn on the stove`，226 steps
- Goal task 8：`put the bowl on the plate`，91 steps

全部 80 episodes 的 `error` 均为 `null`。因此这是策略任务执行失败，不是 simulator、WebSocket、EGL 或 OOM 失败。

## 3. 产物完整性

- 结果：8 个 `results.jsonl`，每个均含 10 条 episode 和 1 条 summary
- 视频：80 个 MP4，每个 episode 一个
- trace：8 个 `action_trace.jsonl`，均含 10 个 `episode_end`
- E2 action steps：13,200；全部 40 个 episode 都运行到最大步数
- E7 action steps：12,772；3 个成功 episode 提前结束

## 4. 与训练动作的对比

训练数据：1,693 个 parquet，273,465 个 7D action。

完整统计：

```text
/data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725/action_comparison.json
```

最明显的异常在旋转三维（action 3-5）：

| 数据 | rotation std 3 | rotation std 4 | rotation std 5 |
|---|---:|---:|---:|
| 训练数据 | 0.039 | 0.063 | 0.078 |
| E2 rollout | 1.883 | 1.499 | 1.657 |
| E7 rollout | 1.395 | 1.082 | 1.180 |

相对训练数据，E2 的旋转标准差约放大 48x、24x、21x；E7 约放大 36x、17x、15x。

越过训练数据逐维 q01-q99 区间的比例：

- E2 rotation：52.0%、58.3%、42.0%
- E7 rotation：36.8%、40.7%、28.6%
- 任意动作维越界：E2 81.4%，E7 73.0%

平移前三维的越界比例明显较低，大约 1%-5%。异常主要集中在旋转和 gripper，而不是所有维度统一爆炸。

E7 的 3 个成功 episode 也仍有 59.7% 的步骤至少一维越界，但明显好于 E7 失败 episode 的 73.5%。

## 5. 结论

E7 比 E2 略好，但两者真实闭环能力都很弱。本次动作记录给出了比成功率更直接的证据：模型送入环境的旋转动作严重偏离 demonstration 分布。

这与当前动作表示风险一致：LIBERO 原始 action 已经是 delta action，但训练配置启用了 `extra_delta_transform=True`，训练目标会再减去当前绝对 state，推理再加回 state。数学上可逆，但要求窄 Q/O PEFT 极精确地学习姿态抵消；小的 transformed-space 误差会在还原后形成很大的 raw rotation action。

因此下一步优先级应是：

1. 使用 `extra_delta_transform=False` 重新计算 norm stats，做短程 A/B。
2. 同时训练官方 full fine-tuning 或标准 action-expert LoRA，作为阳性对照。
3. 不优先继续调 gate；当前主要矛盾是动作表示和可训练容量。

## 6. 复现动作统计

```bash
uv run python scripts/compare_libero_rollout_actions.py \
  --train-glob '/data1/gqy/data/huggingface/lerobot/physical-intelligence/libero/data/**/*.parquet' \
  --eval-root /data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725 \
  --output /data1/gqy/evals/ocaev1/action_trace_smoke_seed42_20260725/action_comparison.json
```
