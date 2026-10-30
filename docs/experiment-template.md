# T4 / H20 实验记录

状态：待运行 / 启动失败 / 正确性通过 / 性能实验完成（按实际填写）。不要将模板中的配置建议写成实测结果；「预期」一栏抄指南里的源码推断，「实际」一栏只写日志和计数里看到的。

## 环境与实验条件

| 项目 | 实际值 |
| --- | --- |
| 日期、轨道（T4 / H20）、机器 / GPU | |
| nvidia-smi 容量、驱动版本、空闲显存 | |
| CUDA 构建（默认 CUDA 13 / `-cu129`）、`CUDA_COMPAT` 取值、检查片段里 `libcuda` 报的版本 | |
| 运行方式（Docker / 原生）；镜像标签与实际 digest，或原生的 Python 路径与 `pip freeze` | |
| 引擎版本（vLLM 0.32.0 / SGLang 0.5.23）、Python、PyTorch、CUDA、Transformers 版本 | |
| checkpoint 与 revision、`PROFILE` | |
| 量化、激活 / KV / 递推状态 dtype | |
| 普通注意力后端（`Using … attention backend` 原文）/ T4 的 W4A16 内核（`Using …LinearKernel` 原文）/ GDN 内核 | |
| max-model-len / max-num-seqs / batched tokens / gpu-memory-utilization | |
| CUDA Graph / prefix cache / 追加的引擎参数 | |
| 服务启动（容器或原生进程）到 `/v1/models` 的墙钟时间 | |
| 注意力块大小 N（`Setting attention block size to N tokens` 原文）与指南里的预期 N | |
| Mamba/GDN 模式（0.32 开前缀缓存时应有 `Mamba cache mode is set to 'align' …` 一行；关掉时没有这一行，模式为 `none`） | |
| KV 池（`GPU KV cache size: … tokens, Maximum concurrency for … tokens per request: …x` 原文） | |
| 异步调度：`ASYNC_SCHED` 取值、`non-default args` 里有没有 `async_scheduling`（0.32 不再打印 `Asynchronous scheduling is …`） | |
| thinking、采样、输入长度与输出上限 | |
| 客户端并发 / 是否本机、SSH 隧道或 port-forward | |
| 完整启动命令（`DRY_RUN=1` 输出）与客户端命令 | |

## 正确性与结果

| 项目 | 实际值 |
| --- | --- |
| 服务就绪与短问答结果 | |
| 成功/失败/length 截断数 | |
| usage 输入、输出 token 数 | |
| 客户端 TTFT、首答案时间、E2E | |
| TPOT 口径 / chunk 间隔口径 | |
| 输出 token/s 与请求/s | |
| 显存峰值 / 服务器 preemption | |
| 每条请求的 `cached_tokens` 序列、`cache_hit_requests`；与预期（T4 528、H20 784 或 FP8 KV 时 1568 的整数倍）是否一致 | |
| prefix cache queries 与 hits（`/metrics` 上的原字段名，例如 `vllm:prefix_cache_hits_total`） | |
| SGLang 对照：`max_total_num_tokens=… max_running_requests=…` 那一行、`--mamba-full-memory-ratio` / `--max-mamba-cache-size` 取值 | |
| 原始 JSON、日志位置 | |

## 对照与结论

- 本次只改变了哪个变量：
- 冷启动与预热如何处理：
- 测量窗口里有没有开 profiler 或逐步调试日志（两者都会拖慢稳态读数，和测量分开跑）：
- 并发曲线：每个并发点记每用户速度（tok/s，约等于 1000 ÷ TPOT 的毫秒数）和总输出 tok/s，连成一条线；一条线只变一个维度，固定条件写进标签：
- prefill 对照：按真正算了的 token 计，命中前缀的部分不算进 prefill 吞吐：
- 跨机器对照（两台都有时）：是不是同一 checkpoint（H20 上的 `PROFILE=9b-w4a16`）、同一 CUDA 构建；不是的话，结论只能写成「两条轨道各自的趋势」：
- （可选）一致性：temperature 0 的同一请求单发与并发是否逐字相同。不同不直接判为 bug，批的组成会改变浮点累加顺序；记下来：
- 数据支持的结论：
- 尚不能判断的部分：
- 若失败，阶段、完整错误与下一步：
