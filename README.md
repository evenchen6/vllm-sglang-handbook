# 从 Transformer 到 vLLM / SGLang

面向有深度学习与后台开发基础、刚开始接触 LLM 推理的工程师。学习路径跟当前稳定版：**vLLM 0.32.0**，对照用 **SGLang 0.5.23**。按手上的机器分两条轨道，模型都是 Gated DeltaNet 混合结构，在 vLLM 里是同一个类：

- **T4 受限轨**：单张 T4（SM75，标称 16 GB），`Qwen3.5-9B` W4A16、FP16 激活、短上下文，看显存与内核的边界。
- **H20 主线轨**：单张 H20（SM90，约 96 GiB），`Qwen3.8-27B` FP8（BF16 作对照），看吞吐、前缀缓存、并发与 KV 容量。

内容：

- [图解手册](https://evenchen6.github.io/vllm-sglang-handbook/)：模型结构、生成时序、KV、分页、调度与选型。
- [T4 / H20 双轨实验指南](docs/t4-qwen35-lab.md)：先选轨道，再做环境检查（含驱动与 CUDA 13）、启动、流式体验、并发与前缀缓存对照、SGLang 对照。
- [原理补充](docs/generation-and-cache.md)：张量形状、首 token、控制 token、混合缓存与块大小、调度与性能瓶颈（T4 与 H20 两套屋顶线），并对照 DeepSeek 开源的 DeepInfer / deepseek-recipe 的做法（DeepInfer 只支持 SM100 / 昇腾 950，不在这两张卡上跑）。
- [实验记录模板](docs/experiment-template.md)：记录实际环境与结果。

## 开始

需要 Linux、可用的 NVIDIA 驱动、Docker 与 NVIDIA Container Toolkit；在 K8s GPU Pod 里没有 Docker 时见下文。先阅读实验指南第 2 节：默认镜像 `vllm/vllm-openai:v0.32.0` 是 CUDA 13 构建，驱动低于 R580 时，数据中心卡（T4、H20）在 R535/R570 上要加 `CUDA_COMPAT=1`，或改用 `-cu129` 镜像。

```bash
git clone https://github.com/evenchen6/vllm-sglang-handbook.git
cd vllm-sglang-handbook

# T4
bash scripts/serve-vllm-t4.sh
# H20（驱动 535.x）
CUDA_COMPAT=1 bash scripts/serve-vllm-h20.sh
```

已经在 GPU Pod 里、没有 Docker 时，把 Pod 镜像设成 `vllm/vllm-openai:v0.32.0`（镜像里的 `python3` 已装好 vLLM 0.32.0），再用同一组参数启动；el8 系统（glibc 2.28）上直接 pip 安装会卡在 llguidance 的源码编译，见指南第 2 节：

```bash
RUNTIME=native bash scripts/serve-vllm-t4.sh
CUDA_COMPAT=1 RUNTIME=native bash scripts/serve-vllm-h20.sh
```

服务就绪后，在同一机器的另一个终端运行（客户端只用 Python 标准库，Python 3.9 起即可）：

```bash
python3 examples/stream-bench.py --requests 1 --warmup 0 --show-output                          # T4
python3 examples/stream-bench.py --model qwen3.8-27b-fp8 --requests 1 --warmup 0 --show-output  # H20
```

T4 默认 `RedHatAI/Qwen3.5-9B-quantized.w4a16`、FP16 激活、4096 上下文、调度上限 1，首次关闭前缀缓存和 CUDA Graph 以缩小排障范围。H20 默认 `Qwen/Qwen3.8-27B-FP8`、BF16 激活、32768 上下文、调度上限 64，前缀缓存和 CUDA Graph 按 0.32 默认打开。客户端默认关闭思考。脚本固定模型 revision 与引擎版本（Docker 用 `v0.32.0` 镜像，原生方式检查 `vllm==0.32.0`），并让服务在 usage 里返回每条请求的 `cached_tokens`；`DRY_RUN=1` 先查看完整命令，脚本参数原样追加到引擎参数末尾。旧版手册的 vLLM 0.18.1 只作历史参照。

**验证状态：** 文档对照 0.32.0 / 0.5.23 的源码、文档、官方镜像元数据与模型卡；脚本拼出的各组引擎参数用 0.32.0 自己的参数解析器校验过，三份固定 checkpoint 在无 GPU 机器上跑过 0.32.0 的配置代码（架构、量化方式、状态模式日志）。网页交互、脚本语法和 HTTP/SSE 客户端在本地验证。本项目维护环境没有 T4 或 H20，尚未完成任何一条轨道的真实 GPU 启动、质量与性能验证（Docker 与原生都是）；不提供虚构的显存峰值或 tok/s。前向兼容（`CUDA_COMPAT=1`）在 535 驱动上的整条链路也没有实测。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `index.html` | 无构建步骤的交互手册 |
| `scripts/serve-vllm-t4.sh` / `scripts/serve-vllm-h20.sh` | 两条轨道的入口，固定 `MACHINE` 后调用 `scripts/serve-vllm.sh` |
| `scripts/serve-vllm.sh` | 共享的启动逻辑：版本、模型 revision、各机器默认值与开关；`RUNTIME=docker`（默认）或 `RUNTIME=native` |
| `examples/stream-bench.py` | 流式问答、客户端延迟、并发闭环实验与每条请求的前缀命中 |
| `examples/trace-generation.py` | 可选：Qwen3.5-4B 的逐次前向、混合缓存与 CPU-GPU 同步观测（transformers ≥ 5.2） |
| `docs/` | 实验步骤、原理说明、结果模板 |

Qwen 的混合结构贯穿主线。先用普通因果注意力解释通用原理，再标出 Gated DeltaNet 的状态差异；不要求先学习旧模型。

## 本地检查

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile examples/stream-bench.py examples/trace-generation.py
bash -n scripts/serve-vllm.sh scripts/serve-vllm-t4.sh scripts/serve-vllm-h20.sh
shellcheck scripts/*.sh
git diff --check
```

协议测试使用本机模拟 HTTP/SSE 服务；启动脚本测试用假的 `docker` / `python` / `nvidia-smi` 检查两条轨道、两种运行方式拼出的命令；缓存形状测试用替身对象。都不需要 GPU；通过这些检查不代表 GPU 内核验证通过。
