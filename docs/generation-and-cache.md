# 生成、缓存与混合架构：补齐第一次学习的关键环节

本文用 Qwen3.5 贯穿实验。先说明普通因果注意力的通用原理，再标出 Gated DeltaNet 的差异，避免把混合模型硬套成纯 Transformer。

## 整模型和一个 Block 分开看

整模型：Token IDs → Embedding → 多个 Decoder Block → 最终 Norm → LM Head → logits → 采样。Embedding 与 LM Head 并不出现在每一个 Block 里。

常见 Pre-Norm Block 有两条残差：`x = x + Attention(Norm(x))`，随后 `x = x + FFN(Norm(x))`。这是教学骨架；原始 2017 Transformer 的 Post-Norm 顺序、Qwen 的 gated attention 与 GDN 内部细节有所不同。RoPE 在注意力的 Q/K 上引入位置旋转，不能笼统画成“在所有模型 Embedding 上加位置向量”。

## 从字符串到 logits

Token 是词表项，不保证是一个汉字或完整单词。Tokenizer 给出 token id；chat template 还会插入角色、消息边界、生成起始标记与思考控制符。相同自然语言内容在不同模板下可能形成不同前缀。

对 batch B、输入长 S、hidden dimension D、词表大小 V：

| 数据 | 普通常规批次形状 |
| --- | --- |
| input_ids | `[B, S]` |
| Embedding / hidden states | `[B, S, D]` |
| Q | `[B, Hq, S, d]` |
| K、V（普通 GQA） | `[B, Hkv, S, d]` |
| logits | `[B, S, V]`（实现可只计算需要的位置） |

每个位置的 logits 预测其后一个 token。生成时取最后位置的分数做采样，然后把选出的 id 追加到输入历史。LM Head 输出分数；选 token 还包括温度、top-k/top-p 等规则。Packed/ragged batch 的引擎布局可以不同，不要拿上表强行匹配所有内部张量。

## 首 token 从哪里来

假设已知输入为 `[x1,x2,x3,x4]`，普通自回归执行如下。采样不等于额外一次模型前向：

| 次数 | 本次新输入 | 本次新算的缓存位置 | 本次采样输出 |
| --- | --- | --- | --- |
| 第一次（prefill） | x1,x2,x3,x4 | x1…x4 | y1 |
| 第二次（decode） | y1 | y1 | y2 |
| 第三次（decode） | y2 | y2 | y3 |

因此 y1 刚输出时，它自身的 KV 尚未计算；需要下一次把它送入模型。最后输出的 EOS 或达到生成上限的最后 token，可能根本不再做一次前向来补自己的 KV。引擎可能预留槽位，但预留槽位不是缓存已计算。

Prefill 可以并行处理多个输入位置；因果 mask 只限制“看谁”，不要求 prompt 按 token 串行计算。Chunked prefill 将长输入切成多个 chunk，通常只在最后一个 chunk 完成时采样首个新 token。动画的等长步骤仅表示执行先后，不代表相同耗时。

## 为什么缓存 K/V，而不是 Q

新位置的查询 `q_new` 要与已知历史的 K/V 交互，过去的 Q 不参与这个新位置的普通注意力计算。因果性保证早先位置不依赖后来才出现的 token，因此它们在各层已计算的 K/V 可以复用。

不带 cache 的朴素生成循环会重算全部历史的投影、注意力与 FFN；使用 cache 后，仅对新输入位置做新的层变换，但仍须读取相关历史 KV。缓存不会把每一步注意力都变成与历史长度无关的常数操作。

## Qwen3.5-9B 的两种缓存不能混为一谈

模型有 32 层，布局为 8 组“3 层 GDN + 1 层全注意力”。全注意力层有 4 个 KV 头、头维 256。

普通全注意力 K/V 的理论字节数（无共享、相同长度、未计分页对齐）为：

`2 × 全注意力层数 × KV头数 × 头维 × 总序列长 × 活跃序列数 × 每元素字节数`

在 FP16、8192 总 token、单序列下，9B 模型这部分为 `2×8×4×256×8192×1×2 = 256 MiB`。

另外 24 层 GDN 使用卷积与递推状态，并非每个 token 都保存完整 K/V 历史；**这不意味着它们没有缓存或不占显存**。单个活跃序列的滚动状态通常不随完整历史线性增长，但多序列、状态精度、引擎状态检查点与 prefix caching 会改变实际内存分配。

整卡预算至少包括：模型实际权重 + 全注意力 KV 池 + GDN 状态/检查点 + 临时激活/工作区 + CUDA Graph/框架等。nvidia-smi 占用、PyTorch allocated/reserved 与逻辑 KV 使用量是不同口径。

## PagedAttention、Radix 与 FlashAttention 各优化什么

| 机制 | 主要问题 | 不能据此推断的结论 |
| --- | --- | --- |
| 分页 KV / PagedAttention | 按逻辑块映射物理块，按需增长、共享，减少预留浪费 | 不会消除模型计算，也仍有尾页内部碎片 |
| APC / Radix 前缀缓存 | 复用相同 token 前缀的既有计算 | 不是语义相似缓存；不能任意拼接相同中间句子 |
| FlashAttention 类内核 | 减少注意力计算的中间矩阵物化与 HBM 读写 | 不等于分页，也不等于跨请求复用；硬件与版本要求独立 |

混合模型的前缀恢复必须同时正确恢复普通 KV 和递推状态。能缓存全注意力层，不自动意味着全部混合状态都可在任意 token 边界恢复。前缀缓存通常减少 prefill，不能直接消除 decode 读取历史的成本。

## 从后台服务经验理解性能

Prefill 的多位置矩阵乘可复用权重，长 prompt 常偏计算受限；小 batch decode 每步的计算量少，但需反复读取权重和历史 KV，常偏带宽受限。实际瓶颈取决于 batch、序列长度、硬件、量化与架构；不能只记“decode 只读 KV”或“prefill 永远 compute-bound”。

连续批处理在调度步边界补入新请求，完成即离开，不是在某个运行中的 GPU kernel 中途插入请求。`max-num-seqs` 是调度容量上限，不是客户端连接/QPS 限流。

对单个输出 N>1 token 的请求，可近似理解为 `E2E ≈ TTFT + (N-1)×TPOT`，还需考虑测量边界；分位数不能这样相加。SSE 一个文本事件可包含多个 token，网络 chunk 间隔不能直接冒充 token ITL。思考模型要同时观察 reasoning 首片段与首个答案的时间。

## 来源

- [Transformer 原论文](https://arxiv.org/abs/1706.03762)。
- [Hugging Face 缓存与手写生成循环](https://huggingface.co/docs/transformers/cache_explanation)。
- [Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B) 与 [Transformers 模型文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。
- [vLLM 执行流程](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) 与 [混合缓存管理](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)。
- [LMSYS RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)。
- [Transformers 连续批处理与服务能力](https://huggingface.co/docs/transformers/continuous_batching)：朴素 generate 封装不等于该库今天的全部能力。
