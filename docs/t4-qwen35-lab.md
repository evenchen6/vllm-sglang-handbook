# 单张 T4 上学习 Qwen3.5 推理

主线是 **Qwen3.5-9B W4A16 文本服务**。先体验模型，再做原理观测与系统实验；不要求换成旧模型。本文的配置已对照官方模型卡与指定版本源码，**尚未在本项目维护环境的 T4 上实测**。不要将下列命令或理论显存当成成功运行与性能保证。

## 1. 固定本次实验的条件

| 项目 | 起步配置 | 原因 |
| --- | --- | --- |
| GPU | 单张 T4，SM75 | 按用户现有设备设计 |
| 服务模型 | `RedHatAI/Qwen3.5-9B-quantized.w4a16` | 同代 9B；部分线性权重 INT4 |
| 模型 revision | `a398088c4228b0ae0c8c78df88fd1e4bf445f068` | 固定本次检查到的权重与配置 |
| 引擎镜像 | `vllm/vllm-openai:v0.18.1` | 量化模型卡列出的服务评估版本。实验命令用这一版；块大小、状态模式和 dtype 的看法见第 6 节 |
| 激活 / 未量化权重 | 显式 FP16（`--dtype half`） | 避免把 BF16 配置直接套到 T4 |
| KV | `auto`，跟随 FP16 | 暂不引入独立的 KV 量化变量 |
| 上下文 | 4096（输入 + 输出） | 先跑短请求，后按实测增加 |
| `max-num-seqs` | 1 | 先观察单请求，再提高调度上限 |
| 普通注意力 | `TRITON_ATTN` | 对照 v0.18.1 的后端支持表选择保守路径，不代表 T4 最快路径 |
| CUDA Graph / 前缀缓存 | 首次关闭 | 逐项加回，避免同时调多个变量 |
| 思考模式 | 客户端请求关闭 | 先看可见答案，再单独测 thinking |

W4A16 不意味着所有参数恰好 4bit。这份 checkpoint 的量化 `ignore` 包含 GDN 部分线性层、Embedding、LM Head 和视觉模块等。仅用 `9B × 0.5 字节` 会低估权重占用。`--language-model-only` 限制多模态路径，但不能据此凭空推断具体显存节省值。

按固定 revision 各分片 safetensors 头逐个求和（只读文件头，不是 GPU 实测）：

| 部分 | 字节 | 存储 |
| --- | --- | --- |
| 24 层 GDN 线性层 | 3.24 GB | 16 位，未量化 |
| 32 层 FFN | 2.49 GB | INT4，group 128 |
| LM Head | 2.03 GB | 16 位，未量化 |
| Embedding | 2.03 GB | 16 位，未量化 |
| 8 层全注意力 | 0.24 GB | INT4 |
| **语言模型合计** | **10.04 GB（9.35 GiB）** | |
| 视觉塔 / MTP | 0.91 / 0.49 GB | 文本服务通常不加载 |

`--gpu-memory-utilization 0.85` 在约 15 GiB 可见容量上的预算是 12.75 GiB。扣掉 9.35 GiB 权重，留给 KV、GDN 状态、激活与工作区的只有约 3.4 GiB。这也是起步上下文与并发都设得很保守的原因。

**三个检查分别完成：** 权重能放下；量化加载/矩阵乘内核支持 SM75；GDN 与普通注意力内核支持当前 dtype 和 GPU。普通注意力后端开关不会自动修复 GDN 的内核问题。若 FP16 下出现 NaN 或异常输出，也不能把它当成正常模型质量。

**同日开源的 DeepInfer 不在这张卡上跑。** DeepSeek 2026-10-20 开源的推理框架 [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md) 只支持 NVIDIA SM100 和华为昇腾 950（构建要 CUDA 13 或 CANN 9.2），T4 是 SM75；它服务的 DeepSeek-V4.1-Flash 在 NVIDIA 上的回归测试也要 4 张卡起步。本指南只在讲原理时引用它和配套的 deepseek-recipe，实验命令仍是 vLLM 0.18.1。

## 2. 先确认容器能用 GPU

需要 Linux、Docker、NVIDIA 驱动与 NVIDIA Container Toolkit。宿主机先执行：

```bash
nvidia-smi
docker --version
```

记录 GPU 名、驱动、总显存、其他进程占用。T4 的“16GB”是产品容量标称，`nvidia-smi` 可见容量常约 15360 MiB（15 GiB），以实际读数为准。输出顶部的 CUDA Version 是驱动可支持的 CUDA 版本信息，不等于当前容器 PyTorch 的 CUDA 构建版本。

使用与服务相同的镜像验证 PyTorch。这里不加载模型：

```bash
docker run --rm --gpus device=0 --entrypoint python \
  vllm/vllm-openai:v0.18.1 -c \
  'import torch, vllm; print("vllm", vllm.__version__); print("torch", torch.__version__, "CUDA", torch.version.cuda); print("available", torch.cuda.is_available()); print("GPU", torch.cuda.get_device_name(0)); print("capability", torch.cuda.get_device_capability(0)); x=torch.ones((32,32), device="cuda", dtype=torch.float16); print("FP16 matmul", (x@x).shape)'
```

应识别 T4、capability `(7, 5)` 并完成 FP16 矩阵乘。这只证明基础 CUDA 路径可用，**不能证明量化与 GDN 内核可用**。如果这里失败，先处理驱动/容器运行时；不要改模型上下文去解决环境错误。镜像自带 CUDA/PyTorch 依赖，不要在里面随意升级单个 PyTorch 包。

记录实际镜像 digest（标签也可能被重新发布）：

```bash
docker image inspect vllm/vllm-openai:v0.18.1 --format '{{json .RepoDigests}}'
```

## 3. 启动 9B，先做一次真实问答

在仓库根目录执行：

```bash
DRY_RUN=1 bash scripts/serve-vllm-t4.sh
bash scripts/serve-vllm-t4.sh
```

首次需要下载镜像和权重，给磁盘、内存与下载时间留出空间。默认缓存目录为仓库的 `.cache/huggingface`；可设置绝对路径 `HF_CACHE_DIR` 复用已有缓存。脚本前台启动容器，保留终端查看日志；停止用 Ctrl+C。脚本不自动换模型、不自动升 nightly，也不自动重试掩盖错误。

另开一个终端，在**同一台 T4 机器**上执行：

```bash
curl http://127.0.0.1:8000/v1/models
python3 examples/stream-bench.py --requests 1 --warmup 0 --show-output \
  --prompt '请用中文解释 Go 的 goroutine 泄漏，给一个定位思路。'
```

`/v1/models` 返回后再发请求；模型下载、编译与首次初始化不能混入稳态测试。记下从容器启动到这次 `/v1/models` 成功的墙钟时间。这是读权重和初始化，每条请求的 TTFT 从这之后才开始算。以后比较另一台引擎，比的也是这两段，而不是把加载时间写进 tok/s。服务别名为 `qwen3.5-9b`，不是权重仓库 ID。脚本只用 Python 标准库，不需要在宿主机安装模型依赖。端口默认绑定到 T4 主机的回环地址；从个人电脑访问远端 T4 时可建立 SSH 隧道：

```bash
ssh -L 8000:127.0.0.1:8000 YOUR_T4_HOST
```

在个人电脑运行客户端即可；记录是否经过隧道，它会影响客户端延迟。

启动日志里还有一行要抄：`Asynchronous scheduling is enabled.`（或 `disabled.`）。0.18.1 默认打开异步调度：CPU 排下一步、处理上一步输出的同时，GPU 接着跑当前这一步。按这一版[源码](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/vllm.py)，单卡、不开推测解码的这套配置预期是 enabled。它为什么值得看，第 5 节再说。

检查答案是否完整、是否重复或乱码、`finish_reason` 是否为 `length`。256 输出 token 是冒烟长度，可提高 `--max-tokens`，但它加上实际输入不能超过 4096。短窗口不是对模型能力的直接降级；需要更长上下文/推理时被截断才会影响结果完整性。

## 4. 认识脚本的测量口径

```bash
python3 examples/stream-bench.py --requests 12 --concurrency 1 \
  --label 9b-c1 --output results/9b-c1.json
```

默认先做 1 条不计入结果的预热，再用线程池做闭环请求：最多 C 个请求同时进行，一个结束再补一个。这不是指定到达率的开放流量，也不是线上过载能力证明。每个请求从线程开始实际发起 HTTP 计时，线程池等待不计入其延迟；吞吐的总时间覆盖整个测量批次。

| 字段 | 含义与限制 |
| --- | --- |
| `ttft_text_ms` | 首个非空 content 或 reasoning 文本片段到达时间；忽略 role-only、空片段和 usage 事件。是客户端近似 TTFT，非 GPU 纯计算时间 |
| `first_answer_ms` | 首个非空 `content` 片段到达时间；若启用 thinking，可明显晚于 reasoning 首包；答案被截断没出现时为 null |
| `e2e_ms` | 从发起 HTTP 到流正常结束，含网络、队列、预处理与回传 |
| `estimated_tpot_ms` | `(末文本片段时间 - 首文本片段时间) / (completion_tokens - 1)`；依赖服务端 usage，仅是客户端估计 |
| `text_chunk_gap_ms` | 相邻非空 SSE 文本事件的到达间隔；**不是逐 token ITL**，一个事件可能含多个 token |
| `output_tok_per_s` | 成功请求的服务端 completion token 总数 / 测量总时长；有成功请求缺 usage 时为 null，不拿 chunk 数冒充 token 数 |
| `finish_length` | 输出达到上限的请求数；启用思考后尤其要检查 |

文本事件和 token 对不上，还有反方向的原因：中文多是多字节 UTF-8，一个 token 可能拼不出完整的字，流式解码器通常会先攒着，等后面的 token 补齐再发。DeepSeek 开源的 deepseek-recipe 就这样处理，并且只把 token 形式的 `</think>` 当成思考结束（[源码](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/decoder.rs)）。所以脚本估 TPOT 用服务端 usage 的 token 数，不用事件数。

失败请求不混入成功延迟，但失败数明确保留；不完整 SSE（没有 `[DONE]`）会报错。测试脚本不自动重试。p95 用最近秩法；12 条只够检查方向，不能发布为稳定 p95/p99。需要准确引擎 ITL、缓存命中与调度耗时，应结合对应版本的 `/metrics` 或官方 benchmark 工具。不要把各指标的 p99 直接相加推算 E2E p99。

默认请求会关闭思考。单独体验 thinking：

```bash
python3 examples/stream-bench.py --requests 1 --warmup 0 --thinking \
  --max-tokens 1024 --show-output --output results/9b-thinking.json
```

思考与答案共用 completion token 预算。若 1024 token 内只有 reasoning，`first_answer_ms` 为 null，不代表服务没有生成。reasoning parser 解析字段，`enable_thinking` 控制模板生成模式，两者职责不同。

## 5. 连续批处理：区分客户端并发与调度上限

先停止旧服务，再启用最多 4 条序列：

```bash
MAX_SEQS=4 bash scripts/serve-vllm-t4.sh
```

分别运行（每条命令结束后再执行下一条）：

```bash
python3 examples/stream-bench.py --requests 12 --concurrency 1 --label c1 --output results/c1.json
python3 examples/stream-bench.py --requests 12 --concurrency 2 --label c2 --output results/c2.json
python3 examples/stream-bench.py --requests 12 --concurrency 4 --label c4 --output results/c4.json
```

记录 TTFT、E2E、输出 token/s、实际输出长度、失败数、preemption 与显存。输出长度随生成不同而变化，比较时必须一起看。若服务仍为 `MAX_SEQS=1`，客户端 C=4 主要看到排队，不是四条 decode 真正同时推进。无论哪种配置，网关可接受的连接数都不是 `max-num-seqs`。

同时看 `ttft_text_ms` 和 `estimated_tpot_ms`。并发从 1 到 4 时，若总输出 token/s 上升、单条间隔几乎不变，符合「带宽受限时加并发，把读权重的成本摊薄」；若 TTFT 明显变差，多半是排队，或某一步里的 prefill 挡住了正在吐字的请求。这就是单卡上的 prefill / decode 冲突。把它拆到两台机器上，是后面学 PD 分离时要认出的同一件事。

还有一段时间屋顶线没算：CPU 的调度和后处理。按顺序跑时，CPU 忙的那段 GPU 是空着的。异步调度把两边错开一步，代价是 CPU 看到的长度落后一步，要靠占位和 GPU 侧修正补上（见[原理补充](generation-and-cache.md)）。9B 在 T4 上一步至少约 25 ms，CPU 那点开销占比很小，所以预期这个开关对 token 间隔影响不大。这个预测可以用数据检验：第 7 节加一组 `ASYNC_SCHED=0`，其他条件不变。

依赖启动、把 all-reduce 叠进 GEMM 收尾、把一层收成一两次常驻 kernel，都是小 batch 延迟内核。它们要高端 GPU 和专用插件，这一步不跑。T4 上用上面两项时间看清「延迟形态」和「吞吐形态」的差别即可。

## 6. 先在 vLLM 上做混合模型前缀对照

保持同一模型 revision、精度、token 预算、并发等条件。每组都重启服务，再预热；不把冷启动混进测量。先顺序请求，让前一条的缓存有机会可用：

```bash
# 第一组：关闭前缀缓存，服务前台启动后在另一终端测量
PREFIX_CACHE=0 bash scripts/serve-vllm-t4.sh
python3 examples/stream-bench.py --shared-prefix --requests 12 --concurrency 1 \
  --label prefix-off --output results/prefix-off.json
```

停止第一组服务后：

```bash
PREFIX_CACHE=1 bash scripts/serve-vllm-t4.sh
python3 examples/stream-bench.py --shared-prefix --requests 12 --concurrency 1 \
  --label prefix-on --output results/prefix-on.json
```

脚本把资料放在固定 system 消息里，只在 user 后缀改变编号。`--prefix-lines` 调整资料行数，不是精确 token 数；根据 usage 的 `prompt_tokens` 控制输入 + 输出长度。不能把随机编号放在 system 最前面，再期待后面相同片段仍命中。

**先查块大小。** vLLM 的 APC 只缓存写满的块。Qwen3.5 是混合模型，一页要同时放下若干 token 的全注意力 KV，和这一层 GDN 的状态检查点（config 指定递推状态为 FP32，约 2.05 MiB）。检查点不随 token 变小，所以块会被调大：按 v0.18.1 源码，不开前缀缓存时是 16 × ⌈2.05 MiB ÷ (16 × 4 KiB)⌉ = 528 token，开了前缀缓存还可能再调。以启动日志的 `Setting attention block size to N tokens` 为准，再让共享 system 段明显长于 N。命中长度会是 N 的整数倍，user 后缀基本不会命中。

**状态模式决定 GDN 能不能跟着 KV 一起复用。** 在 v0.18.1 上，关掉前缀缓存时是 `none`，不为复用保存这份状态。开着前缀缓存且模型支持时，默认常常是 `all`：在每个 `i * block_size` 都存状态。这是 [这一版 cache 配置](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/cache.py)写明的默认，日志里出现 `all` 是预期，把原文抄进记录。v0.32 起去掉了 `all`。要带到更新引擎的模型是：开着 APC 用 `align`（只在调度步刚好结束在块边界时，保存这一步最后一个 token 的 GDN 状态），关掉用 `none`。APC 只命中写满的块，所以检查点和 KV 页落在同一条边界上。无论这次日志是 `all` 还是以后的 `align`，复用是否成立都看命中长度是不是 N 的整数倍。

这条判断在别的混合结构里也成立：命中长度是所有状态能在同一条边界一起拿回来的最长整块前缀。DeepSeek-V4.1-Flash 的全局 KV 按 128 token 一块存；滑动窗口（SWA）的状态不在最后那块里，就退一块，把最后 128 个 token 重新前向，近似补回（SWA Bounded Replay）。SWA 能这样补，是因为它每层只看最近一个窗口；GDN 的状态累积整段历史，只能存检查点。细节见[原理补充](generation-and-cache.md)。

两组都做完后，对着 [v0.18.1 的 APC 说明](https://docs.vllm.ai/en/v0.18.1/features/automatic_prefix_caching/) 填实验记录：

- 启动日志里的块大小 N，以及 Mamba/GDN 模式原文。
- 激活 dtype、KV dtype、普通注意力后端。本实验预期是 FP16、`auto` 跟随 FP16、`TRITON_ATTN`。这三项在更新的服务里仍然要分开看：低精度 KV 压的是每 token 字节，不改变 W4 权重。
- `/metrics` 里 prefix cache 的 queries 与 hits，用页面上的原字段名。命中率用这两个数算。字段没出现、命中为 0 或请求短于 N，就记成这次没有复用，不能解释成缓存算法无效。

APC/Radix 主要节省重复的 **prefill**；不会消除 decode 对历史上下文的读取。高并发同时发来的第一波请求可能都赶不上复用，先顺序验证命中，再增加并发。

## 7. 每次只增加一个变量

依次尝试：`MAX_LEN=8192`、`MAX_SEQS=2/4`、`EAGER=0` 开 CUDA Graph、调整 `BATCHED_TOKENS`、前缀缓存、思考模式、`ASYNC_SCHED=0` 关掉异步调度。不要一次全开。更大上下文既受 KV 容量也受实际输入/输出需求影响，模型卡的 128K 思考建议不作为这张卡的首次验收条件。

调 `BATCHED_TOKENS` 时记住它是 token 预算，不是时间预算：同样 1024 个新 token，接在长前缀后面时注意力要和更长的历史算，这一块更贵。DeepSeek 的 prefill 调度给每一步定一个时间预算，用拟合的代价模型（新 token 数、已算前缀长度）反解这一块能切多大（[Q&A](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2%20Q%26A.pdf)）。在 T4 上比较 token 预算时，把输入长度一起记下。`ASYNC_SCHED=0` 那组同理，只改这一项，c1 和 c4 都测，比 `estimated_tpot_ms`。

若使用另一个引擎镜像，可设置 `VLLM_IMAGE`，但这是新环境，需要保存 digest、help 和全部结果。版本升级后不要将它与原来的数据混为一组。换镜像之后仍然记第 6 节那三项：块大小、状态模式、queries/hits，外加就绪时间和同一条 shared-prefix 的 TTFT。

## 8. 出错时按阶段定位

| 阶段 / 现象 | 先看什么 | 下一步 |
| --- | --- | --- |
| 容器识别不到 GPU | 驱动、Container Toolkit、同镜像 PyTorch 检查 | 先修基础环境 |
| 下载失败 / 磁盘不足 | 网络、缓存盘容量、模型 revision | 处理下载；不改变模型参数掩盖问题 |
| 加载权重 OOM | 实际 GPU 空闲容量、其他进程、未量化模块 | 清理本次实验资源或选择同代 4B；缩 KV 窗口不保证解决权重 OOM |
| `no kernel image` / unsupported capability | SM75、量化内核、GDN / 普通注意力的具体栈 | 查对应版本支持；降低上下文通常无效 |
| BF16 不支持 | 命令是否显式 `--dtype half`、实际日志 dtype | 检查未量化路径，不把 W4 等同 FP16 |
| GDN / Triton 编译失败或长时间无输出 | 具体 kernel、dtype、是否仍在编译 | 保存日志与版本；不要把普通注意力开关当成 GDN 开关 |
| KV 校验 / profiling / Graph OOM | 输入上限、调度序列数、Graph、池分配 | 从 2048、1 条、EAGER=1 起步；逐项复测 |
| NaN、重复、乱码 | 数值、量化兼容、模板、finish_reason | 先确认正确性，再测速度 |
| reasoning 很长、答案为空 | thinking、max_tokens、上下文、reasoning/content 字段 | 先关 thinking 打通；再扩大输出预算并检查截断 |
| prefix 无收益 / 启动拒绝 | 实际启用状态、GDN 状态模式、命中计数 | 记录为未成立的实验条件 |

可手动做同代备用实验，脚本不会自动切换：

```bash
PROFILE=4b-fp16 bash scripts/serve-vllm-t4.sh
python3 examples/stream-bench.py --model qwen3.5-4b --requests 1 --warmup 0 --show-output
```

4B FP16 只减少权重容量/量化变量，不保证绕过同样的 GDN 内核问题。它不是 9B 的性能/质量代表，不把二者曲线放在一起当引擎比较。

## 9. 可选的逐次前向观测

这是 **Qwen3.5-4B FP16** 的独立五步实验；先停止 9B 服务。使用 full multimodal class 的文本输入路径，不把 9B compressed checkpoint 当成普通全精度 HF 权重直接加载。

```bash
mkdir -p "$PWD/.cache/huggingface"
docker run --rm --gpus device=0 --ipc=host \
  -v "$PWD:/work" -w /work \
  -v "$PWD/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint python vllm/vllm-openai:v0.18.1 \
  examples/trace-generation.py --steps 5
```

第二次追加 `--no-cache`，观察每次输入长度。代码只打印实际 cache 对象公开的张量；没有公开形状不等于没有状态。HF 普通注意力 `eager` 也不保证 GDN 不使用额外内核；这项同样未做真实 T4 验证。不要用这个逐步打印实验比较引擎吞吐。

第三次追加 `--sync-debug`：脚本在生成循环里打开 PyTorch 的 `torch.cuda.set_sync_debug_mode("warn")`，遇到 CPU-GPU 同步就打印警告（同一行只报一次；这是实验性功能，不保证覆盖所有同步）。`argmax(...).item()` 和把 token 解码成文字，都要 CPU 停下来等 GPU 算完，朴素循环每一步都这样等。异步调度要消掉的正是这段等待；DeepInfer 在前向和采样的发射路径里把这个模式设成 `error`，出现同步就直接报错（[源码](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/utils.py)）。

## 10. 主线做完之后，怎样读 SGLang

本仓库没有声称跑通的 SGLang + Qwen3.5-9B + W4A16 + T4 配方。官方安装环境、SM75 内核、Qwen3.5 加载与量化要一起核对；“某版本支持 T4”不等于这份 checkpoint 能加载。下面是对照你已经跑过的 vLLM 实验该看什么，不是另一套启动命令。

先把 vLLM 记录抄到一边：容器启动到 `/v1/models` 的时间、块大小 N、Mamba/GDN 模式、prefix cache 的 queries/hits、shared-prefix 开关两组的 TTFT、激活 / KV dtype、注意力后端。

再读 [SGLang 安装文档](https://docs.sglang.io/docs/get-started/install) 和 [Qwen3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)，只回答四个是否：加载器能否读这份 W4A16、能否强制 FP16、GDN 在 SM75 上有没有内核、上下文能不能先收到 4096。有一项不成立就停，把缺口写进记录。不要照抄 BF16、262144 上下文的命令。

若某一版确实用同一 checkpoint 起服务，复用同一客户端，并只比下面三段：

- **就绪时间。** 权重从磁盘进 GPU 的等待，和请求的 TTFT 不是一回事。SGLang 的 InstantTensor 加载器针对前一段。项目报告称加载约快 10 倍；同页的测量是 DeepSeek-R1、8×H200、50 GB/s 存储，从 160 s 降到 15.3 s。不是 9B，也不是 T4。你能比的是同一块盘、同一份权重上的就绪时间。
- **同一条 shared-prefix 的 TTFT 和缓存命中。** 协议与第 6 节相同：顺序、预热、只改 user 后缀。两引擎都能跑通之前，不做吞吐胜负。
- **prefill 与 decode 是否还挤在同一步。** 单张 T4 拆不开。PD 分离把长 prompt 的 prefill 和持续的 decode 放到不同实例，用搬走 KV 换掉「重算 prompt」和「decode 被 prefill 挡住」。SGLang 在长 prompt 负载上报告过吞吐最多增加 27%、p95 TTFT 降到约一半。那是他们的测量。你在 T4 上已经看到的，是第 5 节里 TTFT 与 token 间隔怎样一起变。

注意力的数据并行是另一件事：多张卡上把不同请求的注意力分开算，份数不等于模型副本数。阅读服务参数时，找「注意力 DP 的大小」（新文档里的 `--attn-dp-size`），不要把旧的布尔开关 `--enable-dp-attention` 当成还要手写的步骤。单卡实验用不到。

低延迟插件（PDL、GEMM 收尾里的 all-reduce、AMD monokernel）要 GB300 / MI355X 这一档的 GPU。不要在 T4 上打开。

只有两引擎都用同一 checkpoint/revision、精度、长度、采样与前缀条件跑通之后，才把吞吐放在一起比。若服务已经起来，客户端是：

```bash
python3 examples/stream-bench.py --base-url http://127.0.0.1:30000/v1 \
  --model YOUR_SERVED_MODEL_ID --requests 1 --warmup 0 --show-output
```

## 官方依据（核对日期 2026-10-20）

实验命令对照 v0.18.1。正文里的外部测量分别来自 SGLang 对加载与 PD 的报告，以及 LMSYS 的低延迟内核文章，都不是这张 T4 的结果。

- [Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)：混合结构、文本模式、思考控制与采样参数。
- [RedHatAI W4A16 模型卡](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16)：量化排除模块与评估环境（vLLM 0.18.1）。
- [该 revision 的量化配置](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16/blob/a398088c4228b0ae0c8c78df88fd1e4bf445f068/config.json)。
- [vLLM v0.18.1 GPU 安装](https://docs.vllm.ai/en/v0.18.1/getting_started/installation/gpu/) 与 [注意力后端表](https://docs.vllm.ai/en/v0.18.1/design/attention_backends/)：版本化的硬件/后端依据。
- [v0.18.1 CompressedTensors WNA16 源码](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)：最低 capability 75 只是这条量化 scheme 的门槛，不是整模型验证。
- [v0.18.1 cache 配置](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/cache.py)：`none` / `all` / `align` 的定义。开着前缀缓存且模型支持时，这一版默认常常是 `all`。[v0.32.0 说明](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)去掉 `all`，开 APC 用 `align`、关掉用 `none`。实验把 0.18.1 启动日志里的模式原文抄下。
- [v0.18.1 APC 说明](https://docs.vllm.ai/en/v0.18.1/features/automatic_prefix_caching/) 与 [v0.18.1 APC 设计](https://docs.vllm.ai/en/v0.18.1/design/prefix_caching.html)、[v0.18.1 benchmark CLI](https://docs.vllm.ai/en/v0.18.1/cli/bench/serve/)。
- [SGLang 对 InstantTensor 与 PD 的报告](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)：第 10 节里的加载时间和长 prompt PD 数字。
- [LMSYS：低延迟内核](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode)：PDL、GEMM 收尾 all-reduce、monokernel，以及 GB300 / MI355X 上的 AgentX 数字。
- [SGLang 安装指南](https://docs.sglang.io/docs/get-started/install)。安装指南不代表已有 T4 + Qwen3.5-9B W4A16 配方。
- [vLLM v0.18.1 异步调度配置](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/vllm.py)：默认自动开启，遇到不兼容的推测解码方式或执行器才关，并打印 `Asynchronous scheduling is …`；`ASYNC_SCHED=0` 对应 `--no-async-scheduling`。
- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md) 与[推理部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2%20Q%26A.pdf)：硬件只支持 SM100 / 昇腾 950；第 6、7 节的 SWA 命中规则和 prefill 时间预算来自 Q&A，是 DeepSeek 生产栈的做法，不是 T4 结果。[deepseek-recipe 的流式解码器](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/decoder.rs)：第 4 节的 UTF-8 缓冲。
