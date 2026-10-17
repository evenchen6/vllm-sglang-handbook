# 单张 T4 上学习 Qwen3.5 推理

主线是 **Qwen3.5-9B W4A16 文本服务**。先体验模型，再做原理观测与系统实验；不要求换成旧模型。本文的配置已对照官方模型卡与指定版本源码，**尚未在本项目维护环境的 T4 上实测**。不要将下列命令或理论显存当成成功运行与性能保证。

## 1. 固定本次实验的条件

| 项目 | 起步配置 | 原因 |
| --- | --- | --- |
| GPU | 单张 T4，SM75 | 按用户现有设备设计 |
| 服务模型 | `RedHatAI/Qwen3.5-9B-quantized.w4a16` | 同代 9B；部分线性权重 INT4 |
| 模型 revision | `a398088c4228b0ae0c8c78df88fd1e4bf445f068` | 固定本次检查到的权重与配置 |
| 引擎镜像 | `vllm/vllm-openai:v0.18.1` | 量化模型卡列出的服务评估版本；并非最新版本的宣称 |
| 激活 / 未量化权重 | 显式 FP16（`--dtype half`） | 避免把 BF16 配置直接套到 T4 |
| KV | `auto`，跟随 FP16 | 暂不引入独立的 KV 量化变量 |
| 上下文 | 4096（输入 + 输出） | 先跑短请求，后按实测增加 |
| `max-num-seqs` | 1 | 先观察单请求，再提高调度上限 |
| 普通注意力 | `TRITON_ATTN` | 对照 v0.18.1 的后端支持表选择保守路径，不代表 T4 最快路径 |
| CUDA Graph / 前缀缓存 | 首次关闭 | 逐项加回，避免同时调多个变量 |
| 思考模式 | 客户端请求关闭 | 先看可见答案，再单独测 thinking |

W4A16 不意味着所有参数恰好 4bit。这份 checkpoint 的量化 `ignore` 包含 GDN 部分线性层、Embedding、LM Head 和视觉模块等。仅用 `9B × 0.5 字节` 会低估权重占用。`--language-model-only` 限制多模态路径，但不能据此凭空推断具体显存节省值。

**三个检查分别完成：** 权重能放下；量化加载/矩阵乘内核支持 SM75；GDN 与普通注意力内核支持当前 dtype 和 GPU。普通注意力后端开关不会自动修复 GDN 的内核问题。若 FP16 下出现 NaN 或异常输出，也不能把它当成正常模型质量。

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

`/v1/models` 返回后再发请求；模型下载、编译与首次初始化不能混入稳态测试。服务别名为 `qwen3.5-9b`，不是权重仓库 ID。脚本只用 Python 标准库，不需要在宿主机安装模型依赖。端口默认绑定到 T4 主机的回环地址；从个人电脑访问远端 T4 时可建立 SSH 隧道：

```bash
ssh -L 8000:127.0.0.1:8000 YOUR_T4_HOST
```

在个人电脑运行客户端即可；记录是否经过隧道，它会影响客户端延迟。

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

**Qwen3.5 特有的检查：** vLLM 不只要缓存 8 层全注意力 KV，还需可复用的 GDN 状态。v0.18.1 的 Mamba 缓存模式包含 `none/all/align`；自动选择和模型支持约束要看实际启动日志，而不是只看到命令里有 `--enable-prefix-caching` 就认定全部生效。对照 `/metrics` 的 prefix cache 查询/命中指标；若功能被禁用、没有命中或请求过短，记录限制，不能强行解释为缓存算法无效。

APC/Radix 主要节省重复的 **prefill**；不会消除 decode 对历史上下文的读取。高并发同时发来的第一波请求可能都赶不上复用，先顺序验证命中，再增加并发。

## 7. 每次只增加一个变量

依次尝试：`MAX_LEN=8192`、`MAX_SEQS=2/4`、`EAGER=0` 开 CUDA Graph、调整 `BATCHED_TOKENS`、前缀缓存、思考模式。不要一次全开。更大上下文既受 KV 容量也受实际输入/输出需求影响，模型卡的 128K 思考建议不作为这张卡的首次验收条件。

若使用另一个引擎镜像，可设置 `VLLM_IMAGE`，但这是新环境，需要保存 digest、help 和全部结果。版本升级后不要将它与原来的数据混为一组。

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

## 10. SGLang 是完成主线后的独立兼容性实验

本仓库暂不提供声称跑通的 SGLang + Qwen3.5-9B + W4A16 + T4 配置。官方安装环境、SM75 内核、Qwen3.5 模型加载与量化支持要一起核对；“SGLang 某版本支持 T4”不等于任意新模型都支持。按当前安装指南/模型卡选择版本并固定 commit 或镜像 digest。成功后复用 HTTP 客户端：

```bash
python3 examples/stream-bench.py --base-url http://127.0.0.1:30000/v1 \
  --model YOUR_SERVED_MODEL_ID --requests 1 --warmup 0 --show-output
```

只有两引擎都能跑同一 checkpoint/revision、精度、长度、采样与前缀条件时，才做引擎性能对照。

## 官方依据（核对日期 2026-10-17）

- [Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)：混合结构、文本模式、思考控制与采样参数。
- [RedHatAI W4A16 模型卡](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16)：量化排除模块与评估环境（vLLM 0.18.1）。
- [该 revision 的量化配置](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16/blob/a398088c4228b0ae0c8c78df88fd1e4bf445f068/config.json)。
- [vLLM v0.18.1 GPU 安装](https://docs.vllm.ai/en/v0.18.1/getting_started/installation/gpu/) 与 [注意力后端表](https://docs.vllm.ai/en/v0.18.1/design/attention_backends/)：版本化的硬件/后端依据。
- [v0.18.1 CompressedTensors WNA16 源码](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)：最低 capability 75 只是这条量化 scheme 的门槛，不是整模型验证。
- [v0.18.1 cache 配置](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/cache.py)：Mamba 状态缓存模式。
- [vLLM APC 限制](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) 与 [官方 benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)。stable 页面会变化，使用时要匹配实际引擎版本。
- [SGLang 当前安装指南](https://docs.sglang.io/docs/get-started/install)。
