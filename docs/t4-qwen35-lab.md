# T4 / H20 双轨：用 vLLM 0.32 学 Qwen 混合模型推理

默认学习路径跟当前稳定版：**vLLM 0.32.0**（2026-10-18 发布），对照用 **SGLang 0.5.23**。两条轨道共用一份启动脚本、一个客户端和同一份记录模板：

- **T4 受限轨**：单张 T4（SM75，标称 16 GB），`Qwen3.5-9B` 的 W4A16 量化版、FP16 激活、短上下文。学的是放不放得下、哪些内核在 SM75 上能用、前缀缓存和显存的边界。T4 没有 FP8 张量核。
- **H20 主线轨**：单张 H20（SM90，约 96 GiB；主机可能有 8 张卡，Pod 通常只暴露 1 张），`Qwen3.8-27B` 的 FP8 版，BF16 版作对照。学的是吞吐、前缀缓存、并发和 KV 容量，从 0.32 的默认行为出发。

两个模型在 vLLM 0.32 里是同一个类 `Qwen3_5ForConditionalGeneration`：都是「3 层 Gated DeltaNet（GDN）+ 1 层全注意力」的混合结构，块大小、状态模式和前缀命中的规则一样，只是层数和头数不同。T4 上弄懂的，到 H20 上直接用。

命令和预期来自 0.32.0 / 0.5.23 的源码、文档、官方镜像元数据，以及在无 GPU 机器上用 0.32.0 的参数解析与配置代码做的检查（核对日期 2026-10-30）。**本项目维护环境没有 T4 或 H20，这些组合都没有真实 GPU 启动、质量与性能验证**；文中没有实测的显存峰值、tok/s 或日志。写「预期」的地方，以你自己的 0.32 启动日志和 `/metrics` 为准。

## 先选轨道

| 你手上有 | 怎么走 | 入口 |
| --- | --- | --- |
| 只有 T4 | 第 1–9 节按 T4 那一列做；第 10 节只读 | `scripts/serve-vllm-t4.sh` |
| 只有 H20 | 第 1–10 节按 H20 那一列做 | `scripts/serve-vllm-h20.sh` |
| 两台都有 | 两边按同样的步骤各做一遍：同一版 vLLM、同一个客户端、同一份记录模板。想比 GPU 而不是比模型，在 H20 上加跑 `PROFILE=9b-w4a16`，那是 T4 上的同一份 checkpoint | 两个都用 |

两个入口只是把 `MACHINE` 固定成 `t4` / `h20`，再调用 `scripts/serve-vllm.sh`。`RUNTIME`、`PREFIX_CACHE`、`MAX_SEQS` 等开关在两台机器上含义相同，默认值按机器不同；`DRY_RUN=1` 打印完整命令。脚本参数原样追加到引擎参数末尾，同名参数以后出现的为准。

旧版手册钉的 vLLM 0.18.1 只留作历史参照（T4 那份 W4A16 模型卡的评估版本是 0.18.1），不再是学习默认值。0.18.1 时的几条结论在 0.32 都变了：`all` 状态模式、`Asynchronous scheduling is …` 日志、Python 3.10、glibc 2.31 门槛。下文按 0.32 逐条重写。

## 1. 固定本次实验的条件

| 项目 | T4 受限轨 | H20 主线轨 | 原因 |
| --- | --- | --- | --- |
| GPU | 单张 T4，SM75 | 单张 H20，SM90 | 按用户现有设备 |
| 引擎 | vLLM 0.32.0，镜像 `vllm/vllm-openai:v0.32.0` | 同左 | 两台机器同一版本，记录才能放在一起看 |
| 对照引擎 | 不跑（没有核对过的 SM75 配方） | SGLang 0.5.23，镜像 `lmsysorg/sglang:v0.5.23` | 第 10 节 |
| 服务模型 | `RedHatAI/Qwen3.5-9B-quantized.w4a16` | `Qwen/Qwen3.8-27B-FP8`；对照 `Qwen/Qwen3.8-27B` | Qwen3.8 只有 27B 这一档稠密模型，最小的 W4A16 版也有 19.5 GB，T4 放不下；两者同一个 vLLM 类 |
| 模型 revision | `a398088c4228b0ae0c8c78df88fd1e4bf445f068` | FP8 `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`，BF16 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 固定本次检查到的权重与配置 |
| 权重量化 | W4A16（INT4，group 128） | FP8 块量化（128×128，激活动态量化） | T4 没有 FP8 张量核；SM90 有 |
| 激活 / 未量化权重 | FP16（`--dtype half`） | BF16（`--dtype bfloat16`） | T4 没有 BF16 张量核 |
| KV | `auto`，跟随 FP16 | `auto`，跟随 BF16；`KV_DTYPE=fp8` 是单独一组 | 先不引入 KV 量化变量 |
| 上下文 | 4096 | 32768（模型原生 262144） | 先跑短请求，后按实测增加 |
| `max-num-seqs` / batched tokens | 1 / 1024 | 64 / 8192 | 8192 是 0.32 给 ≥70 GiB 卡的服务默认；64 够做 1–64 的并发曲线 |
| 普通注意力 | `TRITON_ATTN` | 交给 vLLM 选（SM90 优先 `FLASH_ATTN`） | 0.32 在 SM75 上 FlashAttention 不可用、FlashInfer 被暂时禁用，自动选择也会落到 Triton |
| CUDA Graph / 前缀缓存 | 首次都关 | 都开（0.32 默认） | T4 先缩小排障范围；H20 先看主线默认，对照时再关 |
| 思考模式 | 客户端请求关闭 | 同左 | 先看可见答案，再单独测 thinking |

**T4：9B W4A16 的权重账。** W4A16 不意味着所有参数恰好 4bit。这份 checkpoint 的量化 `ignore` 包含 GDN 部分线性层、Embedding、LM Head 和视觉模块等，仅用 `9B × 0.5 字节` 会低估权重。按固定 revision 各分片 safetensors 头逐个求和（只读文件头，不是 GPU 实测）：

| 部分 | 字节 | 存储 |
| --- | --- | --- |
| 24 层 GDN 线性层 | 3.24 GB | 16 位，未量化 |
| 32 层 FFN | 2.49 GB | INT4，group 128 |
| LM Head | 2.03 GB | 16 位，未量化 |
| Embedding | 2.03 GB | 16 位，未量化 |
| 8 层全注意力 | 0.24 GB | INT4 |
| **语言模型合计** | **10.04 GB（9.35 GiB）** | |
| 视觉塔 / MTP | 0.91 / 0.49 GB | `--language-model-only` 不加载视觉塔 |

`--gpu-memory-utilization 0.85` 在约 15 GiB 可见容量上的预算是 12.75 GiB，扣掉 9.35 GiB 权重，留给 KV、GDN 状态、激活与工作区的只有约 3.4 GiB。这是起步上下文和并发都设得很保守的原因。

**H20：27B 的权重账。** 同样按固定 revision 的文件头求和：

| 部分 | FP8 checkpoint | BF16 checkpoint |
| --- | --- | --- |
| 64 层 FFN | 17.12 GB（FP8） | 34.23 GB |
| 48 层 GDN 线性层 | 5.59 GB（FP8，几个小张量 BF16） | 11.12 GB |
| 16 层全注意力 | 1.68 GB（FP8） | 3.36 GB |
| Embedding / LM Head | 2.54 / 2.54 GB（BF16，未量化） | 2.54 / 2.54 GB |
| **语言模型合计** | **29.47 GB（27.44 GiB）** | **53.79 GB（50.10 GiB）** |
| 视觉塔 / MTP | 0.92 / 0.48 GB | 0.92 / 0.85 GB |

按约 96 GiB 可见容量（以你的 `nvidia-smi` 为准）、`MEM_UTIL=0.90` 算，预算约 86 GiB：FP8 扣掉权重剩约 59 GiB，BF16 剩约 36 GiB，给 KV、GDN 状态、激活、CUDA Graph。账要分两种：全注意力 KV 每 token 64 KiB（16 层 × 4 KiB，BF16），GDN 状态每条序列 146.8 MiB（48 层，递推部分按 config 存 FP32），一条序列在约 2350 token 以内时 GDN 状态比 KV 还大。9B 对应的数是 32 KiB / token、49.1 MiB / 序列、约 1570 token。池子实际多大，看启动日志的 `GPU KV cache size: … tokens, Maximum concurrency for … tokens per request: …x`。

**三个检查分别完成：** 权重能放下；量化内核支持这块 GPU；GDN 与普通注意力内核支持当前 dtype 和 GPU。T4 上 0.32 的 W4A16 按顺序试内核，SM90 起步的 Cutlass W4A8、Machete 会跳过，Marlin 从 SM75 起；启动日志 `Using …LinearKernel for mixed-precision linear` 写的是实际选中的那个，抄下来。普通注意力后端开关不会修复 GDN 的内核问题；FP16 下出现 NaN 或异常输出，也不能当成正常质量。H20 上 FP8 块量化 GEMM 是 SM90 的原生路径，主要风险在驱动和 CUDA 版本（第 2 节）。

**大 MoE 不在这两张卡上。** DeepSeek-V4.1-Flash 按 [HF 页面](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)的 safetensors 统计约 765 GB，单张 96 GB 的 H20 放不下；它的生产引擎 [DeepInfer](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md) 只支持 NVIDIA SM100 和华为昇腾 950，回归测试 4 张卡起步。H20 主机有 192 核、2.2 TiB 内存，以后学 Engram 的 CPU offload 有条件，但前提是先有多卡（TP/EP），那是本指南之后的一步。这里只在讲原理时引用 DeepInfer 和配套的 deepseek-recipe。

## 2. 先确认运行环境能用 GPU

宿主机（或 Pod 里）先执行：

```bash
nvidia-smi
docker --version   # 在 K8s Pod 里通常没有，见本节后半
```

记录 GPU 名、驱动版本、总显存、其他进程占用。T4 的「16GB」是产品标称，`nvidia-smi` 可见容量常约 15360 MiB。输出顶部的 CUDA Version 是驱动支持的最高 CUDA 版本，不等于镜像里 PyTorch 的 CUDA 构建版本。

**驱动决定用哪个 CUDA 构建。** 0.32.0 的默认镜像 `vllm/vllm-openai:v0.32.0` 和 PyPI wheel 都按 CUDA 13.0 构建，正常需要 R580 以上的驱动；官方另发了 CUDA 12.9 的 `v0.32.0-cu129` 镜像和 wheel。

| 驱动 | 做法 | 说明 |
| --- | --- | --- |
| R580 及以上 | 直接用默认镜像 | |
| R535、R570（T4、H20 都是数据中心卡） | 加 `CUDA_COMPAT=1` | 前向兼容：容器里自带 CUDA 13.0 的 compat 用户态驱动，NVIDIA 的兼容矩阵列出 `cuda-compat-13-0` 支持 535、570 分支。PTX JIT 照常可用。SGLang 0.5.23 只有 CUDA 13 镜像，两边对照也是这条路 |
| 其他旧分支（如 R550，已不在兼容矩阵里），或兼容模式报 803/804 | `VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129`，或升级驱动 | CUDA 12.x 小版本兼容只覆盖预编译的 SASS，不能 JIT 更新的 PTX。部分内核（例如 Marlin 的权重重排）在 SM90 上只带 PTX，要驱动现编，旧驱动会报 `the provided PTX was compiled with an unsupported toolchain` |

H20 现在是 535.247.01，按第二行走。T4 的驱动按你的读数选。两台机器尽量用同一个 CUDA 构建（都 CUDA 13 或都 cu129），否则对照时要把它记成一个变量。

`CUDA_COMPAT=1` 做两件事：设 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`（vLLM 文档的开关），并在**进程启动之前**把 `/usr/local/cuda/compat`（官方镜像自带）放到 `LD_LIBRARY_PATH` 最前（NVIDIA 前向兼容文档的做法）。只设前者不够：vLLM 在 Python 里改 `LD_LIBRARY_PATH`，而 glibc 只在进程启动时读这个变量，0.32 的 EngineCore 和 worker 默认是 fork 出来的，也读不到。本项目在本机用 `ctypes` 加载测试库复现了这一点，但没有在 535 驱动的 GPU 上验证整条链路。前向兼容只适用于数据中心和专业卡，消费级卡加载 compat 库会报错 803。

用与服务相同的镜像验证 PyTorch，这里不加载模型。下面的检查片段同时打印实际加载的 `libcuda` 报告的 CUDA 版本：535 系统驱动是 12020，compat 生效后应是 13000。

```bash
CHECK='import ctypes, torch, vllm
v = ctypes.c_int(); ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v))
print("vllm", vllm.__version__, "| torch", torch.__version__, "CUDA build", torch.version.cuda, "| libcuda", v.value)
cap = torch.cuda.get_device_capability(0); dtype = torch.float16 if cap < (8, 0) else torch.bfloat16
print("GPU", torch.cuda.get_device_name(0), "capability", cap)
x = torch.ones((32, 32), device="cuda", dtype=dtype); print("matmul", dtype, (x @ x).shape)'

# 驱动 R580+：
docker run --rm --gpus device=0 --entrypoint python3 vllm/vllm-openai:v0.32.0 -c "$CHECK"
# 驱动 R535/R570（与 CUDA_COMPAT=1 相同的库路径）：
docker run --rm --gpus device=0 --entrypoint python3 \
  -e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  vllm/vllm-openai:v0.32.0 -c "$CHECK"
```

T4 应识别 capability `(7, 5)` 并完成 FP16 矩阵乘，H20 是 `(9, 0)` 和 BF16。这只证明基础 CUDA 路径可用，**不能证明量化与 GDN 内核可用**。这里失败先处理驱动、兼容库或容器运行时，不要改模型上下文去解决环境错误。镜像自带 CUDA/PyTorch 依赖，不要在里面随意升级单个包。

记录实际镜像 digest（标签可能被重新发布）：

```bash
docker image inspect vllm/vllm-openai:v0.32.0 --format '{{json .RepoDigests}}'
```

2026-10-30 在 Docker Hub 上查到的 manifest list digest：`v0.32.0` 是 `sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`，`v0.32.0-cu129` 是 `sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f`。

**已经在 GPU Pod 里、没有 Docker。** K8s Pod 里一般跑不了 Docker（嵌套容器要特权模式），不必硬装。启动脚本的 `RUNTIME=native` 把同一组引擎参数交给本机的 vLLM 进程（第 3 节）。vLLM 0.32.0 从哪来，按推荐顺序：

- **Pod 能换镜像（推荐，T4 和 H20 都适用）：** 把 Pod 镜像设成 `vllm/vllm-openai:v0.32.0`（按上表也可能是 `-cu129`），`command` 改成 `sleep infinity` 再进去操作（镜像默认入口是 `vllm serve`）。镜像里的 `python3` 已装好 vLLM 0.32.0，依赖和 Docker 路径相同；`/usr/local/cuda/compat` 也在，`CUDA_COMPAT=1` 直接可用。digest 用 `kubectl get pod YOUR_POD -o jsonpath='{.status.containerStatuses[0].imageID}'` 查。上面的检查改成在 Pod 里执行 `python3 -c "$CHECK"`，驱动 R535/R570 时前面加 `LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH`。
- **只能用现有系统：** 自建 venv。0.32.0 要 Python 3.11–3.14（系统自带的 3.9 不行，用 uv 下载托管的 3.12），x86_64 wheel 是 `manylinux_2_28`。但依赖 `llguidance` 1.7.x 给 CPython 的 x86_64 wheel 只有 `manylinux_2_31`，glibc 低于 2.31 的系统（el8 系，例如 TencentOS 3.2，是 2.28）会转去编译它的 Rust 源码包，没有 Rust 工具链就失败。这是 2026-10-30 用 uv 按 `manylinux_2_28` 平台解析出来的结论，198 个依赖里只有它没有现成 wheel。el8 上请回到上一条。

glibc ≥ 2.31 的机器上，venv 和权重缓存放在空间够的数据盘上，例如 `/data/vllm-lab`：

```bash
uv venv /data/vllm-lab/.venv --python 3.12 --seed --managed-python
source /data/vllm-lab/.venv/bin/activate
uv pip install 'vllm==0.32.0' --torch-backend=cu130          # CUDA 13 构建，与默认镜像一致
# 或 CUDA 12.9 构建（与 -cu129 镜像一致）：
# uv pip install 'https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' --torch-backend=cu129
vllm download-kernels   # FlashInfer 预编译内核；不装会在 Hopper 上启动时现编几分钟
```

两条安装命令都用 uv 按 `manylinux_2_31` 平台解析过，分别得到 `torch==2.13.0+cu130` 和 `torch==2.13.0+cu129`。venv 里要用前向兼容，得另装 `cuda-compat-13-0`（来自 NVIDIA 的 CUDA 软件源），再设 `CUDA_COMPAT_DIR=/usr/local/cuda-13.0/compat`。然后在这个环境里做同一项检查；依赖清单代替镜像 digest 进记录：

```bash
python3 -c "$CHECK"
mkdir -p results && python3 -m pip freeze > results/pip-freeze.txt
```

## 3. 启动，先做一次真实问答

在仓库根目录执行，先看完整命令再启动：

```bash
# T4（驱动是 R535/R570 时在前面加 CUDA_COMPAT=1）
DRY_RUN=1 bash scripts/serve-vllm-t4.sh
bash scripts/serve-vllm-t4.sh

# H20（驱动 535）
CUDA_COMPAT=1 DRY_RUN=1 bash scripts/serve-vllm-h20.sh
CUDA_COMPAT=1 bash scripts/serve-vllm-h20.sh
```

已在 GPU Pod 里、没有 Docker，按第 2 节备好 vLLM 0.32.0 之后，在前面加 `RUNTIME=native`；venv 没激活时再加 `PYTHON=/data/vllm-lab/.venv/bin/python`：

```bash
RUNTIME=native bash scripts/serve-vllm-t4.sh
CUDA_COMPAT=1 RUNTIME=native bash scripts/serve-vllm-h20.sh
```

两种方式共用同一份引擎参数，只换拉起进程的那一层。镜像入口 `vllm serve` 换成 `python3 -m vllm.entrypoints.cli.main serve`，它就是 `vllm` 命令本身，只是固定用通过了版本检查的那个解释器。服务直接监听 `127.0.0.1:$PORT`。`GPU_DEVICE` 写进 `CUDA_VISIBLE_DEVICES`，`HF_CACHE_DIR` 写进 `HF_HOME`（目录结构与容器里的 `/root/.cache/huggingface` 相同）；这两个脚本变量没设时沿用环境里已有的值，都没有才用 0 和仓库的 `.cache/huggingface`。环境里的 vllm 不是 0.32.0 时，脚本直接退出并打印安装提示。不要改用 `python -m vllm.entrypoints.openai.api_server`：0.32 里它已标为弃用，而且仍然不读位置参数里的模型，会静默起默认的 `Qwen/Qwen3-0.6B`。

后面各节的启动命令，原生方式都在前面加 `RUNTIME=native`（或先 `export RUNTIME=native`），H20 再加 `CUDA_COMPAT=1`，其余变量含义不变。

首次需要下载镜像（原生方式没有这一步）和权重：9B 约 11 GB，27B FP8 约 31 GB，BF16 约 56 GB（都含视觉塔和 MTP 的分片）。默认缓存目录是仓库的 `.cache/huggingface`；可设置绝对路径 `HF_CACHE_DIR` 复用已有缓存，在 Pod 里指到数据盘，例如 `/data/vllm-lab/hf`。脚本前台启动服务，保留终端看日志，Ctrl+C 停止。脚本不自动换模型、不自动升级，也不自动重试掩盖错误。

另开一个终端，在**同一台机器**（或同一个 Pod）上执行：

```bash
curl http://127.0.0.1:8000/v1/models
# T4（服务别名 qwen3.5-9b 是客户端默认值）
python3 examples/stream-bench.py --requests 1 --warmup 0 --show-output \
  --prompt '请用中文解释 Go 的 goroutine 泄漏，给一个定位思路。'
# H20：加上服务别名
python3 examples/stream-bench.py --model qwen3.8-27b-fp8 --requests 1 --warmup 0 --show-output \
  --prompt '请用中文解释 Go 的 goroutine 泄漏，给一个定位思路。'
```

`/v1/models` 返回后再发请求；下载、编译与首次初始化不能混入稳态测试。记下从服务启动（容器或本机进程）到 `/v1/models` 成功的墙钟时间：这是读权重和初始化，请求的 TTFT 从这之后才开始算。服务别名不是权重仓库 ID：T4 默认 `qwen3.5-9b`，H20 默认 `qwen3.8-27b-fp8`，BF16 版 `qwen3.8-27b`。客户端只用 Python 标准库，Pod 里系统自带的 Python 3.9 就够。端口默认绑在本机回环地址；从个人电脑访问远端机器时建 SSH 隧道 `ssh -L 8000:127.0.0.1:8000 YOUR_HOST`，K8s 里用 `kubectl port-forward pod/YOUR_POD 8000:8000`，并记录是否经过隧道。

**启动日志里抄这几行**（引号里是 0.32 源码的日志原文，`N`、`…` 处是你的值）：

- `Using … attention backend out of potential backends: …`：实际选中的普通注意力后端。T4 预期 `TRITON_ATTN`，H20 预期 `FLASH_ATTN`。
- `Setting attention block size to N tokens to ensure that attention page size is >= mamba page size.`：混合模型的块大小，第 6 节要用。
- `GPU KV cache size: … tokens, Maximum concurrency for … tokens per request: …x`：KV 池能装多少、按 `MAX_LEN` 算能并发几条。
- `Model loading took … GiB memory and … seconds`；T4 还有 `Using …LinearKernel for mixed-precision linear`。
- `non-default args: {…}`：脚本显式传了哪些参数。

0.18.1 的 `Asynchronous scheduling is enabled.` 这一行在 0.32 已经没有了。按 0.32 的[配置源码](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/vllm.py)，不传 `--async-scheduling` 时自动开启，只有池化模型、不兼容的推测解码方式或不支持的执行器才关；单卡、不开推测解码的这两套配置预期是开启。想在日志里留下证据，就显式传 `ASYNC_SCHED=1` 或 `0`，`non-default args` 里会出现 `async_scheduling`。为什么值得看，见第 5 节。

检查答案是否完整、是否重复或乱码、`finish_reason` 是否为 `length`。256 输出 token 是冒烟长度，可提高 `--max-tokens`。0.32 起 `max_tokens` 只是输出上限，不再从上下文里预留：输入加 `max_tokens` 超过 `MAX_LEN` 不会被拒，输出写满上下文就以 `length` 结束；只有输入本身超过 `MAX_LEN` 才报错。

## 4. 认识脚本的测量口径

```bash
python3 examples/stream-bench.py --requests 12 --concurrency 1 \
  --label t4-9b-c1 --output results/t4-9b-c1.json
# H20 加 --model qwen3.8-27b-fp8，标签写 h20-27b-fp8-c1
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
| `cached_tokens` | 每条请求的输入里有多少 token 来自前缀缓存，取自 usage 的 `prompt_tokens_details.cached_tokens`。启动脚本传了 `--enable-prompt-tokens-details`，vLLM 才会返回；SGLang 对应 `--enable-cache-report`。汇总里是分布和 `cache_hit_requests`（命中数大于 0 的请求数），服务端没报告时为 null |

文本事件和 token 对不上，还有反方向的原因：中文多是多字节 UTF-8，一个 token 可能拼不出完整的字，流式解码器通常会先攒着，等后面的 token 补齐再发。DeepSeek 开源的 deepseek-recipe 就这样处理，并且只把 token 形式的 `</think>` 当成思考结束（[源码](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/decoder.rs)）。所以脚本估 TPOT 用服务端 usage 的 token 数，不用事件数。

失败请求不混入成功延迟，但失败数明确保留；不完整 SSE（没有 `[DONE]`）会报错。测试脚本不自动重试。p95 用最近秩法；12 条只够检查方向，不能发布为稳定 p95/p99。需要准确引擎 ITL、缓存命中与调度耗时，应结合 `/metrics` 或官方 benchmark 工具。不要把各指标的 p99 直接相加推算 E2E p99。

默认请求会关闭思考（采样参数取自模型卡的非思考建议：T = 0.7、top-p 0.8、top-k 20、presence penalty 1.5，Qwen3.5 和 Qwen3.8 相同）。单独体验 thinking：

```bash
python3 examples/stream-bench.py --requests 1 --warmup 0 --thinking \
  --max-tokens 1024 --show-output --output results/t4-9b-thinking.json
```

思考与答案共用 completion token 预算。若 1024 token 内只有 reasoning，`first_answer_ms` 为 null，不代表服务没有生成。reasoning parser 解析字段，`enable_thinking` 控制模板生成模式，两者职责不同。Qwen3.8 模型卡另有 `reasoning_effort` 和默认开启的 `preserve_thinking`，多轮对话会保留历史思考，影响前缀长度；本指南的单轮实验不涉及。

## 5. 连续批处理：区分客户端并发与调度上限

**T4：** 先停止旧服务，再启用最多 4 条序列，分别运行 C = 1、2、4（每条命令结束后再执行下一条）：

```bash
MAX_SEQS=4 bash scripts/serve-vllm-t4.sh
python3 examples/stream-bench.py --requests 12 --concurrency 1 --label t4-c1 --output results/t4-c1.json
python3 examples/stream-bench.py --requests 12 --concurrency 2 --label t4-c2 --output results/t4-c2.json
python3 examples/stream-bench.py --requests 12 --concurrency 4 --label t4-c4 --output results/t4-c4.json
```

**H20：** 默认 `MAX_SEQS=64`，直接测 C = 1、4、16、64；请求数至少是并发的两倍，否则测到的多是爬坡：

```bash
CUDA_COMPAT=1 bash scripts/serve-vllm-h20.sh
for c in 1 4 16 64; do
  python3 examples/stream-bench.py --model qwen3.8-27b-fp8 --requests $((c < 16 ? 32 : 4 * c)) \
    --concurrency "$c" --label "h20-fp8-c$c" --output "results/h20-fp8-c$c.json"
done
```

记录 TTFT、E2E、输出 token/s、实际输出长度、失败数、preemption 与显存。输出长度随生成不同而变化，比较时必须一起看。若服务的 `MAX_SEQS` 小于客户端并发，多出来的请求主要在排队，不是真正同时推进。无论哪种配置，网关可接受的连接数都不是 `max-num-seqs`。

同时看 `ttft_text_ms` 和 `estimated_tpot_ms`。并发升高时，若总输出 token/s 上升、单条间隔几乎不变，符合「带宽受限时加并发，把读权重的成本摊薄」；若 TTFT 明显变差，多半是排队，或某一步里的 prefill 挡住了正在吐字的请求。这就是单卡上的 prefill / decode 冲突。把它拆到两台机器上，是后面学 PD 分离时要认出的同一件事。

**用屋顶线先做预测，再拿数据对。** T4 的数（[NVIDIA 规格](https://www.nvidia.com/en-us/data-center/tesla-t4/)：FP16 张量核 65 TFLOPS、320 GB/s，拐点约 203 FLOP/byte）：9B 每个 decode 步至少读约 8.0 GB 权重，单条请求一步至少约 25 ms。H20 没有 NVIDIA 官网产品页，常被引用的规格（服务器厂商与媒体）是 4.0 TB/s、BF16 148 TFLOPS、FP8 296 TFLOPS（稠密），只拿来做估算：27B FP8 每步至少读约 26.9 GB（语言模型权重减去 Embedding），单条请求一步至少约 6.7 ms；BF16 约 51.3 GB、12.8 ms。只算权重，强度约为 2B（9B）、1.9B（27B FP8）、1.0B（27B BF16）FLOP/byte，H20 的拐点只有 74（FP8）/ 37（BF16）。但每条序列每步还要读、写各一次 GDN 状态（9B 约 0.10 GB，27B 约 0.31 GB），它和 KV 一样随 B 增长：按[手册第 8 节](https://evenchen6.github.io/vllm-sglang-handbook/#prefill)计算器的口径（上下文 1024），H20 上 FP8 要到 B ≈ 84 才碰到拐点（BF16 约 50），T4 上 9B 的强度趋近约 117，始终低于 203。H20 的带宽是 T4 的 12.5 倍，算力只有 2.3 倍（BF16）/ 4.6 倍（FP8）：1024 token 的 prefill 在 T4 上 9B 至少约 0.22 s，在 H20 上 27B FP8 至少约 0.17 s。这些都是理论上下界，没算反量化、kernel 效率与调度，也没在这两张卡上实测。

还有一段时间屋顶线没算：CPU 的调度和后处理。按顺序跑时，CPU 忙的那段 GPU 是空着的。异步调度把两边错开一步，代价是 CPU 看到的长度落后一步，要靠占位和 GPU 侧修正补上（见[原理补充](generation-and-cache.md)）。T4 上一步至少约 25 ms，CPU 那点开销占比很小，预期这个开关对 token 间隔影响不大；H20 上 FP8 一步可短到约 7 ms，同样的 CPU 开销占比大得多，`ASYNC_SCHED=0` 的对照在 H20 上更值得做（第 7 节）。

依赖启动、把 all-reduce 叠进 GEMM 收尾、把一层收成一两次常驻 kernel，都是小 batch 延迟内核，要更高端的 GPU 和专用插件，这两条轨道都不跑。单卡上用上面两项时间看清「延迟形态」和「吞吐形态」的差别即可。

## 6. 先在 vLLM 上做混合模型前缀对照

保持同一模型 revision、精度、token 预算、并发等条件，两组也用同一种运行方式（都用 Docker 或都用原生）。每组都重启服务，再预热；不把冷启动混进测量。先顺序请求，让前一条的缓存有机会可用：

```bash
# 第一组：关闭前缀缓存，服务前台启动后在另一终端测量
PREFIX_CACHE=0 bash scripts/serve-vllm-t4.sh          # H20：CUDA_COMPAT=1 PREFIX_CACHE=0 bash scripts/serve-vllm-h20.sh
python3 examples/stream-bench.py --shared-prefix --requests 12 --concurrency 1 \
  --label prefix-off --output results/prefix-off.json   # H20 加 --model qwen3.8-27b-fp8
```

停止第一组服务后：

```bash
PREFIX_CACHE=1 bash scripts/serve-vllm-t4.sh          # H20：CUDA_COMPAT=1 PREFIX_CACHE=1 bash scripts/serve-vllm-h20.sh
python3 examples/stream-bench.py --shared-prefix --requests 12 --concurrency 1 \
  --label prefix-on --output results/prefix-on.json
python3 -c 'import json,sys; print([r.get("cached_tokens") for r in json.load(open(sys.argv[1]))["results"]])' results/prefix-on.json
```

脚本把资料放在固定 system 消息里，只在 user 后缀改变编号。`--prefix-lines` 调整资料行数，不是精确 token 数。用两个模型的 tokenizer 和 chat template（关闭思考）实算过：默认 32 行时整个输入 988 token，两条请求的公共前缀 966 token；64 行是 1916 / 1894。不能把随机编号放在 system 最前面，再期待后面相同片段仍命中。

**先查块大小 N。** vLLM 的 KV 池只有一种页大小，混合模型的一页要同时放下 N 个 token 的全注意力 KV 和一层 GDN 的状态（递推部分 FP32，不随 token 变小），所以块会被调大。0.32 的[对齐代码](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/platforms/interface.py)：N = 对齐 × ⌈每层状态字节 ÷ (对齐 × 每 token 每层 KV 字节)⌉，对齐取注意力内核支持的最小块，`TRITON_ATTN` 和 SM90 上的 `FLASH_ATTN` 都是 16。用 0.32 自己的 spec 类算出来：

| 配置 | 每层 GDN 状态 | 每 token 每层 KV | 预期 N |
| --- | --- | --- | --- |
| T4，9B，FP16 KV | 2.05 MiB | 4 KiB | 528 |
| H20，27B，BF16 KV | 3.06 MiB | 4 KiB | 784 |
| H20，27B，`KV_DTYPE=fp8` | 3.06 MiB | 2 KiB | 1568 |

FP8 KV 让每 token 的 KV 减半，GDN 状态不变，N 就翻倍，命中粒度也跟着变粗。这一步在 0.32 里对所有混合模型都做，和开不开前缀缓存无关，两组日志里的 N 应当相同。以日志的 `Setting attention block size to N tokens …` 为准。

**状态模式：0.32 只剩 `align` 和 `none`。** PREFIX_CACHE=1 时，启动日志有一行 INFO：`Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled`。这句原文是在无 GPU 机器上跑 0.32.0 的配置代码得到的，9B W4A16、27B FP8、27B BF16 三份 checkpoint 都一样。PREFIX_CACHE=0 时模式保持 `none`，不打印这一行，GDN 状态不为复用保存。0.18.1 里给部分模型用的 `all` 已删除，手动传 `--mamba-cache-mode all` 会在参数解析时报 `invalid choice: 'all' (choose from 'align', 'none')`。脚本不传这个参数。

**`align` 把 GDN 状态存在哪。** 按 0.32 的[调度器](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py)和 [cache 配置](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/cache.py)：状态在调度步结束时写，所以 prefill 的分块要停在块边界上；每条 prompt 会在最后一个整块边界 ⌊(L−1)/N⌋·N 处停一下，保留那里的状态（至少留 1 个 token 重算）。如果后来的请求在全注意力 KV 上命中得比 GDN 状态更长，调度器还会在这个「共享前缀交汇点」停一步补存状态（Marconi 式），下一条请求就能整段命中。默认只保留这些点上的检查点（`prefix_cache_retention_interval=0`）。命中要全部状态在同一条边界上都拿得回来，所以命中长度是 N 的整数倍。

**据此对默认 32 行的预期**（源码推断，不是实测）：第一条（预热）不命中；之后各条的最后整块边界在 T4 上是 528、在 H20 上是 784，都落在 966 的公共前缀里，所以从第二条起 `cached_tokens` 应当是 528 / 784。`KV_DTYPE=fp8` 时 N = 1568 比整个输入还长，32 行一个块都凑不满，改用 `--prefix-lines 64`，预期命中 1568。如果交汇点落在后缀里，命中会推迟到第三条。PREFIX_CACHE=0 那组应全为 0。

两组都做完后，对着 [0.32 的 APC 说明](https://docs.vllm.ai/en/v0.32.0/features/automatic_prefix_caching/)填实验记录：

- 日志里的块大小 N、Mamba 模式那一行原文（或「没有这一行」）。
- 每条请求的 `cached_tokens` 序列和 `cache_hit_requests`。和上面的预期对不上时，把序列原样记下：说明 0.32 的实际行为和这里的源码阅读不一致，以日志和计数为准。
- `/metrics` 里的 `vllm:prefix_cache_queries_total` 与 `vllm:prefix_cache_hits_total`（单位都是 token），命中率用这两个数算；服务日志也会周期性打印 `Prefix cache hit rate: …%`。
- 激活 dtype、KV dtype、普通注意力后端。低精度 KV 压的是每 token 字节，不改变权重。

APC/Radix 主要节省重复的 **prefill**；不会消除 decode 对历史上下文的读取。高并发同时发来的第一波请求可能都赶不上复用，先顺序验证命中，再增加并发。

「命中长度 = 所有种类的状态都能在同一条边界拿回来的最长整块前缀」在别的混合结构里也成立。DeepSeek-V4.1-Flash 的全局 KV 按 128 token 一块存；滑动窗口（SWA）的状态不在最后那块里，就退一块，把最后 128 个 token 重新前向，近似补回（SWA Bounded Replay，0.32 的 DeepSeek-V4 实现里也有）。SWA 能这样补，是因为它每层只看最近一个窗口；GDN 的状态累积整段历史，只能存检查点。细节见[原理补充](generation-and-cache.md)。

## 7. 每次只增加一个变量

**T4** 依次尝试：`MAX_LEN=8192`、`MAX_SEQS=2/4`、`EAGER=0` 开 CUDA Graph、调整 `BATCHED_TOKENS`、前缀缓存、思考模式、`ASYNC_SCHED=0` 关掉异步调度。不要一次全开。更大上下文既受 KV 容量也受实际输入/输出需求影响，模型卡的长上下文建议不作为这张卡的首次验收条件。

**H20** 从主线默认出发，每次改一项：`KV_DTYPE=fp8`（KV 每 token 减半，N 变成约 1568，看 `GPU KV cache size` 和最大并发怎么变，同时验收质量）；`PROFILE=27b-bf16`（同一模型，权重字节翻倍，看 decode 间隔）；`PROFILE=9b-w4a16`（T4 的同一份 checkpoint，只换 GPU）；`ASYNC_SCHED=0`（c1 和 c16 都测，比 `estimated_tpot_ms`）；`EAGER=1`；`MAX_LEN=131072` 或 `262144`；`MAX_SEQS`、`BATCHED_TOKENS`。需要引擎参数而脚本没有开关的，直接写在命令后面，例如用模型自带的 MTP 头做推测解码（[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) 的写法）：

```bash
CUDA_COMPAT=1 bash scripts/serve-vllm-h20.sh --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

推测解码改的是每步验证的行数，接受率要从 `/metrics` 的 `vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_draft_tokens_total` 读，吞吐本身分不出草稿有没有起作用。GDN 状态改存 BF16（`--mamba-ssm-cache-dtype bfloat16`）能把每条序列的状态减半，vLLM 会警告与 config 不符，质量要自己验收。

调 `BATCHED_TOKENS` 时记住它是 token 预算，不是时间预算：同样 1024 个新 token，接在长前缀后面时注意力要和更长的历史算，这一块更贵。DeepSeek 的 prefill 调度给每一步定一个时间预算，用拟合的代价模型（新 token 数、已算前缀长度）反解这一块能切多大（[Q&A](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2%20Q%26A.pdf)）。比较 token 预算时，把输入长度一起记下。

换镜像就是换环境：`VLLM_IMAGE` 换成 `-cu129`、或者从 Docker 换到原生，都要另存 digest（或 `pip freeze`）、`--help` 和全部结果，不和原来的数据混为一组。原生方式只认 vllm 0.32.0；想试别的版本就另建 venv，照 `DRY_RUN=1` 打印的参数手动起服务。换环境之后仍然记第 6 节那几项：块大小、状态模式、`cached_tokens`、queries/hits，外加就绪时间和同一条 shared-prefix 的 TTFT。

多卡是这之后的一步：Pod 能拿到多张 H20 时，再学张量并行（`--tensor-parallel-size`）、专家并行和 PD 分离，以及 DeepSeek-V4.1-Flash 这类大 MoE。现在的脚本固定 `--tensor-parallel-size 1`。

## 8. 出错时按阶段定位

| 阶段 / 现象 | 先看什么 | 下一步 |
| --- | --- | --- |
| 容器 / Pod 识别不到 GPU | 驱动、Container Toolkit 或 Pod 的 GPU 资源、第 2 节的检查 | 先修基础环境 |
| `CUDA driver version is insufficient` / `The NVIDIA driver on your system is too old` | `nvidia-smi` 的驱动版本；检查片段里 `libcuda` 报的版本 | 驱动低于 R580：数据中心卡在 R535/R570 上加 `CUDA_COMPAT=1`，否则用 `-cu129` 镜像或升级驱动 |
| 加了 `CUDA_COMPAT=1` 仍失败，或报 803 / 804 | GPU 是否数据中心卡、驱动分支是否在 NVIDIA 兼容矩阵里、`/usr/local/cuda/compat` 是否存在 | 改用 `-cu129` 镜像；原生 venv 要装 `cuda-compat-13-0` 并设 `CUDA_COMPAT_DIR` |
| `the provided PTX was compiled with an unsupported toolchain` | 是否用了 `-cu129` 配旧驱动（小版本兼容不能 JIT 新 PTX） | 换回默认镜像加 `CUDA_COMPAT=1`，或升级驱动 |
| 原生：pip/uv 找不到 0.32.0 | `python --version`（要 3.11–3.14） | 用 uv 托管的 Python 3.12，或改用镜像作 Pod 镜像 |
| 原生：开始编译 llguidance、要求 Rust | `ldd --version`（低于 2.31，如 el8 的 2.28） | 用 `vllm/vllm-openai:v0.32.0` 作 Pod 镜像 |
| 原生：进程直接被杀，没有 Python 报错 | Pod 的内存 limit，`dmesg` 或 `kubectl describe pod` 里的 OOM 记录 | 调大 Pod 内存；这是主机内存，与显存 OOM 分开处理 |
| 下载失败 / 磁盘不足 | 网络、缓存盘容量、模型 revision | 处理下载；不改变模型参数掩盖问题 |
| 加载权重 OOM | 实际 GPU 空闲容量、其他进程、未量化模块 | 清理本次实验资源；T4 可选同代 4B；缩 KV 窗口不保证解决权重 OOM |
| `no kernel image` / unsupported capability | SM75/SM90、量化内核、GDN / 普通注意力的具体栈 | 查 0.32 的支持情况；降低上下文通常无效 |
| T4：BF16 不支持 | 命令是否带 `--dtype half`、日志里的 dtype | 检查未量化路径，不把 W4 等同 FP16 |
| GDN / Triton 编译失败或长时间无输出 | 具体 kernel、dtype、是否仍在编译 | 保存日志与版本；不要把普通注意力开关当成 GDN 开关 |
| KV 校验 / profiling / Graph OOM | 输入上限、调度序列数、Graph、池分配 | T4 从 2048、1 条、`EAGER=1` 起步；H20 先降 `MAX_SEQS` 或 `MAX_LEN`；逐项复测 |
| `invalid choice: 'all'` | 是否手动传了 `--mamba-cache-mode all` | 0.32 已删除 `all`，去掉这个参数 |
| NaN、重复、乱码 | 数值、量化兼容、模板、`finish_reason` | 先确认正确性，再测速度 |
| reasoning 很长、答案为空 | thinking、`max_tokens`、上下文、reasoning/content 字段 | 先关 thinking 打通；再扩大输出预算并检查截断 |
| prefix 无收益 | 日志里的 N 和模式行、`cached_tokens`、共享段是否长于 N | 记录为未成立的实验条件 |

T4 可手动做同代备用实验，脚本不会自动切换：

```bash
PROFILE=4b-fp16 bash scripts/serve-vllm-t4.sh
python3 examples/stream-bench.py --model qwen3.5-4b --requests 1 --warmup 0 --show-output
```

4B FP16 只减少权重容量和量化变量，不保证绕过同样的 GDN 内核问题。它不是 9B 的性能/质量代表，不把二者曲线放在一起当引擎比较。

## 9. 可选的逐次前向观测

这是 **Qwen3.5-4B FP16** 的独立五步实验，两台机器都能跑；先停止服务。使用 full multimodal class 的文本输入路径，不把 9B compressed checkpoint 当成普通全精度 HF 权重直接加载。脚本要 transformers ≥ 5.2（`Qwen3_5ForConditionalGeneration` 从这一版起才有）；0.18.1 要求 transformers < 5，在它的镜像里会 ImportError，0.32 的镜像带 5.16–5.19，可以直接用。

```bash
mkdir -p "$PWD/.cache/huggingface"
docker run --rm --gpus device=0 --ipc=host \
  -v "$PWD:/work" -w /work \
  -v "$PWD/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint python3 vllm/vllm-openai:v0.32.0 \
  examples/trace-generation.py --steps 5
# 驱动 R535/R570：在镜像名前加
#   -e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
# Pod 里（镜像即 vllm/vllm-openai）：python3 examples/trace-generation.py --steps 5
```

transformers 5.x 的缓存里，全注意力层是 `DynamicLayer`（`keys` / `values` 随 token 增长），GDN 层是 `LinearAttentionLayer`（`conv_states` / `recurrent_states` 按状态序号存，形状不随 token 变）。脚本两种都打印；在 transformers 5.19 上用随机初始化的小模型验证过输出（不是 4B 的真实数）。注意 HF 的卷积状态保留整个卷积核宽度（4），vLLM 只存 3，两边的形状不必相同。

第二次追加 `--no-cache`，观察每次输入长度。代码只打印实际 cache 对象公开的张量；没有公开形状不等于没有状态。HF 普通注意力 `eager` 也不保证 GDN 不使用额外内核（没装 flash-linear-attention 时会退回 PyTorch 参考实现，很慢）；这项同样未做真实 GPU 验证。不要用这个逐步打印实验比较引擎吞吐。

第三次追加 `--sync-debug`：脚本在生成循环里打开 PyTorch 的 `torch.cuda.set_sync_debug_mode("warn")`，遇到 CPU-GPU 同步就打印警告（同一行只报一次；这是实验性功能，不保证覆盖所有同步）。`argmax(...).item()` 和把 token 解码成文字，都要 CPU 停下来等 GPU 算完，朴素循环每一步都这样等。异步调度要消掉的正是这段等待；DeepInfer 在前向和采样的发射路径里把这个模式设成 `error`，出现同步就直接报错（[源码](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/utils.py)）。

## 10. 用 SGLang 0.5.23 做对照

**版本事实（2026-10-30 核对）。** SGLang 0.5.23 的 git tag（`297e54f`）和 PyPI wheel 当天已发布，GitHub 上还没有这一版的 Release 说明页。官方镜像 `lmsysorg/sglang:v0.5.23` 是 CUDA 13.0.3，入口是 NVIDIA 的 `nvidia_entrypoint.sh`，后面直接跟 `sglang serve …`。[安装文档](https://docs.sglang.io/docs/get-started/install)写明 SGLang 只支持 CUDA 13，`v0.5.19-cu129` 是最后一个 CUDA 12 镜像。wheel 要 Python ≥ 3.11、glibc ≥ 2.34（`manylinux_2_34`），el8 装不上，Pod 里同样用镜像。依赖 torch 2.14.1、transformers 5.19.0，和 vLLM 0.32 的 torch 2.13.0 不同，**两个引擎不要装进同一个环境**。

**T4：只读不跑。** 本仓库没有 SGLang + Qwen3.5-9B + W4A16 + T4 的核对过的配方。官方文档说 FlashInfer 在 SM75 上出问题时改用 `--attention-backend triton --sampling-backend pytorch`，但 W4A16 加载、FP16、GDN 在 SM75 上的内核要一起成立，加上 CUDA 13 对驱动的要求，任何一项不成立就停，把缺口写进记录。不要照抄 BF16、262144 上下文的命令。

**H20：与 vLLM 对齐条件。** 同一 checkpoint 与 revision、BF16 激活、`auto` KV（两边对 Qwen3.8-27B-FP8 都解析成 BF16）、32768 上下文、同一个客户端。[SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) 的 H200 配方钉了 FP8 KV，对照时把 FP8 KV 当成两边各自的第二组，不混进第一组。驱动 535 需要前向兼容；SGLang 没有 vLLM 那样的开关，按 NVIDIA 文档把 compat 目录放到 `LD_LIBRARY_PATH` 最前（未验证，先确认镜像里有这个目录）：

```bash
docker run --rm --entrypoint ls lmsysorg/sglang:v0.5.23 /usr/local/cuda/compat
docker run --rm --gpus device=0 --ipc=host --shm-size 32g -p 127.0.0.1:30000:30000 \
  -v "$PWD/.cache/huggingface:/root/.cache/huggingface" \
  -e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  lmsysorg/sglang:v0.5.23 \
  sglang serve --model-path Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
    --served-model-name qwen3.8-27b-fp8 --host 0.0.0.0 --port 30000 \
    --language-model-only --reasoning-parser qwen3 --context-length 32768 \
    --max-running-requests 64 --kv-cache-dtype auto --enable-cache-report --enable-metrics
```

Pod 里（Pod 镜像设成 `lmsysorg/sglang:v0.5.23`）去掉 `docker run …` 那一层，直接执行 `LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH sglang serve …`。启动日志里抄 `max_total_num_tokens=…, max_running_requests=…, context_len=…` 那一行。SGLang 把权重之后的显存分成 GDN 状态池和 KV 池，比例由 `--mamba-full-memory-ratio`（默认 0.9）决定，默认值可能悄悄把并发压到低于 64；cookbook 给了按平均请求长度算这个比例的公式，也可以用 `--max-mamba-cache-size` 直接钉住状态槽数。每条运行中的请求占几个状态槽由 `--mamba-radix-cache-strategy` 决定（默认 `extra_buffer` 是 5 个），一个槽是 153.9 MB（FP32），和上面 vLLM 的 146.8 MiB 是同一个数。两个引擎给 GDN 状态留显存的办法不同，这本身就是要记下来的对照点。

客户端相同，只换地址：

```bash
python3 examples/stream-bench.py --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b-fp8 \
  --requests 1 --warmup 0 --show-output
```

先把 vLLM 记录抄到一边：就绪时间（注明 Docker 还是原生）、块大小 N、Mamba 模式那一行、`cached_tokens` 序列、shared-prefix 开关两组的 TTFT、激活 / KV dtype、注意力后端。然后只比下面三段：

- **就绪时间。** 权重从磁盘进 GPU 的等待，和请求的 TTFT 不是一回事。SGLang 的 InstantTensor 加载器针对前一段，项目报告称加载约快 10 倍；同页的测量是 DeepSeek-R1、8×H200、50 GB/s 存储，从 160 s 降到 15.3 s，不是 27B，也不是 H20。0.32 的 vLLM 也把 `instanttensor` 列为依赖。你能比的是同一块盘、同一份权重上的就绪时间。
- **同一条 shared-prefix 的 TTFT 和缓存命中。** 协议与第 6 节相同：顺序、预热、只改 user 后缀。SGLang 的 KV 按 token 进 Radix 树，GDN 状态另按 `--mamba-radix-cache-strategy` 和 `--mamba-track-interval`（默认 256）存，命中粒度和 vLLM 的 N 不是一回事；两边都以 usage 里的 `cached_tokens` 和各自的 `/metrics`（SGLang 是 `sglang:cache_hit_rate`、`sglang:cached_tokens_total`）为准。两引擎都能跑通之前，不做吞吐胜负。
- **prefill 与 decode 是否还挤在同一步。** 单卡拆不开。PD 分离把长 prompt 的 prefill 和持续的 decode 放到不同实例，用搬走 KV 换掉「重算 prompt」和「decode 被 prefill 挡住」。SGLang 在长 prompt 负载上报告过吞吐最多增加 27%、p95 TTFT 降到约一半，那是他们的测量。单卡上能看到的，是第 5 节里 TTFT 与 token 间隔怎样一起变。

注意力的数据并行是另一件事：多张卡上把不同请求的注意力分开算，份数不等于模型副本数。阅读服务参数时，找「注意力 DP 的大小」（新文档里的 `--attn-dp-size`），不要把旧的布尔开关 `--enable-dp-attention` 当成还要手写的步骤。单卡实验用不到。

低延迟插件（PDL、GEMM 收尾里的 all-reduce、AMD monokernel）要 GB300 / MI355X 这一档的 GPU，两条轨道都不打开。只有两引擎都用同一 checkpoint/revision、精度、长度、采样与前缀条件跑通之后，才把吞吐放在一起比。

## 官方依据（核对日期 2026-10-30）

实验命令对照 vLLM v0.32.0 与 SGLang v0.5.23。正文里的外部测量分别来自 SGLang 对加载与 PD 的报告，以及 LMSYS 的低延迟内核文章，都不是 T4 或 H20 上的结果。

- [vLLM v0.32.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)：默认镜像与 PyPI wheel 是 CUDA 13.0，另有 `-cu129`；Python 3.11–3.14；`transformers>=5.16.1,<5.20.0`；删除 `mamba_cache_mode="all"`；`max_tokens` 不再从上下文预留。[PyPI 上的 0.32.0](https://pypi.org/project/vllm/0.32.0/#files)：x86_64 wheel 为 `manylinux_2_28`。镜像元数据（Docker Hub 注册表）：两个标签的入口都是 `vllm serve`，`TORCH_CUDA_ARCH_LIST` 含 7.5 与 9.0，CUDA 13.0.2 / 12.9.1。
- [0.32 安装文档里的旧驱动说明](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/getting_started/installation/gpu.cuda.inc.md)：CUDA 13 镜像正常需要 R580，兼容模式支持 R535 与 R570，只限专业和数据中心卡；[NVIDIA 前向兼容](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)：`cuda-compat-13-0` 与 535+、570+ 兼容，错误码 803 / 804，PTX JIT 不受小版本兼容那条限制；[vLLM 的兼容开关实现](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/env_override.py)与[默认 fork 的进程方式](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/system_utils.py)。[PR #36579](https://github.com/vllm-project/vllm/pull/36579) 描述了 Marlin 在 SM90 上只带 PTX、cu129 配旧驱动时报错的情形。
- 0.32 源码：[`vllm serve` 把位置参数写回 `--model`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/cli/serve.py)，[`api_server` 模块已弃用](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/openai/api_server.py)；[cache 配置](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/cache.py)（`align` / `none`、`prefix_cache_retention_interval`）；[Mamba 模式与 Qwen3.5 状态 dtype](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/config.py)；[混合块大小对齐](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/platforms/interface.py)；[调度器的检查点停靠点](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py)；[注意力后端优先级](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/platforms/cuda.py)与 [FlashInfer 的 SM75 门槛](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py)；[W4A16 内核顺序](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/__init__.py)；[服务默认的批大小](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/engine/arg_utils.py)；[异步调度的默认与兼容条件](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/vllm.py)；[prefix cache 指标](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/metrics/loggers.py)。
- [0.32 APC 说明](https://docs.vllm.ai/en/v0.32.0/features/automatic_prefix_caching/)与 [0.32 benchmark CLI](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)。
- 模型：[Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[RedHatAI W4A16 模型卡](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16)（评估环境 vLLM 0.18.1）与[该 revision 的量化配置](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16/blob/a398088c4228b0ae0c8c78df88fd1e4bf445f068/config.json)；[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与 [FP8 版](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)的模型卡和[固定 revision 的 config](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json)；[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)（`--reasoning-parser qwen3`、MTP 写法、单张 Hopper 用 TP1）。
- SGLang：[安装指南](https://docs.sglang.io/docs/get-started/install)（只支持 CUDA 13、SM75 的后备参数）、[Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)（H200 配方、GDN 状态池与 `--mamba-full-memory-ratio`）、[v0.5.23 tag](https://github.com/sgl-project/sglang/tree/v0.5.23)、[PyPI 上的 0.5.23](https://pypi.org/project/sglang/0.5.23/#files)；[SGLang 对 InstantTensor 与 PD 的报告](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)：第 10 节里的加载时间和长 prompt PD 数字。
- 硬件：[NVIDIA T4 规格](https://www.nvidia.com/en-us/data-center/tesla-t4/)。H20 没有 NVIDIA 官网产品页，屋顶线用的 4.0 TB/s、148 / 296 TFLOPS 来自服务器厂商与媒体流传的规格表（例如 [Tom's Hardware 转述的规格](https://www.tomshardware.com/tech-industry/nvidia-readies-new-ai-and-hpc-gpus-for-china-market-report)），只作估算。
- [LMSYS：低延迟内核](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode)：PDL、GEMM 收尾 all-reduce、monokernel，以及 GB300 / MI355X 上的 AgentX 数字。
- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md) 与[推理部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/doc/DeepSeek%20V4.1%20Flash%20%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2%20Q%26A.pdf)：硬件只支持 SM100 / 昇腾 950；第 6、7 节的 SWA 命中规则和 prefill 时间预算来自 Q&A，是 DeepSeek 生产栈的做法。[deepseek-recipe 的流式解码器](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/decoder.rs)：第 4 节的 UTF-8 缓冲。
- 历史参照：[vLLM v0.18.1 的 cache 配置](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/cache.py)（当时还有 `all`）与 [v0.18.1 的异步调度日志](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/config/vllm.py)。
