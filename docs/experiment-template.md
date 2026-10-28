# T4 / Qwen3.5 实验记录

状态：待运行 / 启动失败 / 正确性通过 / 性能实验完成（按实际填写）。不要将模板中的配置建议写成实测结果。

## 环境与实验条件

| 项目 | 实际值 |
| --- | --- |
| 日期、机器 / GPU | |
| nvidia-smi 容量、驱动、空闲显存 | |
| 镜像标签与实际 digest | |
| 引擎、Python、PyTorch、CUDA、Transformers 版本 | |
| checkpoint 与 revision | |
| 量化、激活/KV/递推状态 dtype | |
| 普通注意力 / GDN 实际内核 | |
| max-model-len / max-num-seqs / batched tokens | |
| 显存预算 / CUDA Graph / prefix cache / 状态模式 | |
| 容器启动到 `/v1/models` 的墙钟时间 | |
| 注意力块大小 N（启动日志原文） | |
| Mamba/GDN 模式原文（`none` / `align` / `all`；Qwen3.5 @ v0.18.1 开前缀缓存应为 `align`） | |
| 异步调度日志原文（`Asynchronous scheduling is …`）、`ASYNC_SCHED` 取值 | |
| 激活 dtype、KV dtype、注意力后端 | |
| thinking、采样、输入长度与输出上限 | |
| 客户端并发 / 是否本机、SSH 隧道 | |
| 完整启动命令与客户端命令 | |

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
| prefix cache queries 与 hits（用 `/metrics` 上的原字段名） | |
| 原始 JSON、日志位置 | |

## 对照与结论

- 本次只改变了哪个变量：
- 冷启动与预热如何处理：
- 测量窗口里有没有开 profiler 或逐步调试日志（两者都会拖慢稳态读数，和测量分开跑）：
- 并发曲线：每个并发点记每用户速度（tok/s，约等于 1000 ÷ TPOT 的毫秒数）和总输出 tok/s，连成一条线；一条线只变一个维度，固定条件写进标签：
- prefill 对照：按真正算了的 token 计，命中前缀的部分不算进 prefill 吞吐：
- （可选）一致性：temperature 0 的同一请求单发与并发是否逐字相同。不同不直接判为 bug，批的组成会改变浮点累加顺序；记下来：
- 数据支持的结论：
- 尚不能判断的部分：
- 若失败，阶段、完整错误与下一步：
