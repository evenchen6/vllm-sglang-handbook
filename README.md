# 从 Transformer 到 vLLM / SGLang

面向有深度学习与后台开发基础、刚开始接触 LLM 推理的工程师。主实验设备为 **单张 NVIDIA T4（16GB 标称容量）**，主模型为 **Qwen3.5-9B W4A16**。

- [图解手册](https://evenchen6.github.io/vllm-sglang-handbook/)：模型结构、生成时序、KV、分页、调度与选型。
- [T4 + Qwen3.5 实验指南](docs/t4-qwen35-lab.md)：环境检查、启动、流式体验、并发与前缀缓存对照。
- [原理补充](docs/generation-and-cache.md)：张量形状、首 token、控制 token、混合缓存、调度与性能瓶颈，并对照 DeepSeek 开源的 DeepInfer / deepseek-recipe 的做法（DeepInfer 只支持 SM100 / 昇腾 950，不在 T4 上跑）。
- [实验记录模板](docs/experiment-template.md)：记录实际环境与结果。

## 在 T4 上开始

需要 Linux、可用的 NVIDIA 驱动、Docker 与 NVIDIA Container Toolkit。先阅读实验指南中的环境检查，再执行：

```bash
git clone https://github.com/evenchen6/vllm-sglang-handbook.git
cd vllm-sglang-handbook
bash scripts/serve-vllm-t4.sh
```

服务就绪后，在同一机器的另一个终端运行（客户端只用 Python 标准库）：

```bash
python3 examples/stream-bench.py --requests 1 --warmup 0 --show-output
```

默认尝试 `RedHatAI/Qwen3.5-9B-quantized.w4a16`，FP16 激活，4096 上下文，并发调度上限 1。配套客户端默认关闭思考；服务初始关闭前缀缓存和 CUDA Graph，以缩小首次启动的排障范围。脚本固定模型 revision 与引擎镜像版本；`DRY_RUN=1 bash scripts/serve-vllm-t4.sh` 可先查看完整命令。

**验证状态：** 文档对照官方模型卡与对应版本源码；网页交互、脚本语法和 HTTP/SSE 客户端在本地验证。本项目维护环境没有 T4，尚未完成该组合的真实 GPU 启动、质量与性能验证；不提供虚构的显存峰值或 tok/s。若量化或 GDN 内核失败，按指南定位；同代 Qwen3.5-4B 是可选的独立实验，不会自动替换 9B。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `index.html` | 无构建步骤的交互手册 |
| `scripts/serve-vllm-t4.sh` | 固定版本的 T4 启动命令与可调参数 |
| `examples/stream-bench.py` | 流式问答、客户端延迟和并发闭环实验 |
| `examples/trace-generation.py` | 可选：Qwen3.5-4B 的逐次前向、混合缓存与 CPU-GPU 同步观测 |
| `docs/` | 实验步骤、原理说明、结果模板 |

Qwen3.5 的混合结构贯穿主线。先用普通因果注意力解释通用原理，再标出 Gated DeltaNet 的状态差异；不要求先学习旧模型。

## 本地检查

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile examples/stream-bench.py examples/trace-generation.py
bash -n scripts/serve-vllm-t4.sh
git diff --check
```

协议测试使用本机模拟 HTTP/SSE 服务，不需要 GPU；通过这些检查不代表 T4 GPU 内核验证通过。
