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
| 显存峰值 / 服务器 preemption / 缓存查询与命中 | |
| 原始 JSON、日志位置 | |

## 对照与结论

- 本次只改变了哪个变量：
- 冷启动与预热如何处理：
- 数据支持的结论：
- 尚不能判断的部分：
- 若失败，阶段、完整错误与下一步：
