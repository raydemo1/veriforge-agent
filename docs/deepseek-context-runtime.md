# 上下文版本与运行时状态

对话使用不可变消息和显式 Context Epoch。`ContextHistory` 冻结已提交内容，
`SessionJournal` 保存原始消息及 Epoch 基础视图。读取消息得到独立快照；提交日志成功后才更新活动视图。
Agent 重建、请求契约、压缩和恢复改变视图时建立新 Epoch，普通用户消息和工具结果持续追加。
子 Agent 使用各自的日志，避免其 Epoch 覆盖主 Agent 的恢复视图。

DeepSeek 的历史 `reasoning_content` 在流式、非流式和跨轮次请求中保留。
其他 provider 在出站副本中处理不支持的字段。上下文估算计入 reasoning 和工具定义，
并在请求前检查输出空间。估算不是 DeepSeek 服务端的精确 Token 计数。

## 状态与压缩

`get_runtime_status(section)` 从已有状态读取 `task`、`budget`、`verification`、
`environment` 或 `agents`。它不运行测试，不创建新的状态缓存，并计入普通工具预算。
验证结果关联文件指纹；后续文件变化或工作区修改使证据过期。

TODO 不再被临时拼到每次请求。预算到达配置的预警比例时，仅追加一次可信 system 事件。
待执行工具全部返回结果后才允许追加事件或切换 Epoch；硬预算仍由运行时执行。

压缩可以覆盖本轮较早的完整交互，保留最近完整工具组及其原始 reasoning。
摘要分为历史工作和当前运行时事实。失败、工具配对不完整或没有节省空间时不提交。
旧消息、旧预算事件仍保存在原始日志中。

## 工具披露评测

生产路径保留原生工具 Schema 追加。固定工具入口候选及其专项评测脚本已移除，历史评测结果保留备查。

2026-10-23 的有界评测使用配置中的 `normal` 模型强度，请求模型名为
`deepseek-v4-flash-vision-exp`，thinking 开启。共 43 次模型请求，没有扩展到 72 次上限。

| 指标 | 原生 Schema | 固定入口候选 |
| --- | ---: | ---: |
| 成功任务 | 6/6 | 6/6 |
| 参数错误 | 0 | 0 |
| 模型请求 | 21 | 22 |
| 缓存命中 Token | 140,544 | 156,160 |
| 缓存未命中 Token | 13,369 | 7,549 |
| 整体命中率 | 91.31% | 95.39% |
| 排除各任务首请求后的命中率 | 94.83% | 95.09% |
| 排除各任务首请求后的未命中 Token | 5,879 | 6,047 |
| 总 Token | 155,809 | 165,670 |
| 端到端耗时 | 27.457 秒 | 27.139 秒 |

最初的“总 Token 不能增长”门槛已撤回。缓存收益应比较命中率、绝对未命中量和
`命中量 × 命中单价 + 未命中量 × 未命中单价 + 输出量 × 输出单价`。
单价必须使用调用时适用的模型费率，总 Token 只作为容量指标。

这轮公共前缀未按实验分组隔离：原生路径的第一条请求完全未命中，后续候选能够复用公共前缀。
因此整体命中率优势不能直接归因于固定入口。排除首请求也无法完全消除跨组预热影响。
当前结论是缓存收益证据不足。按用户决定保留原生工具披露方式，移除固定入口候选，不追加实验。

原始观测位于 `eval/results/deepseek-disclosure-20261023-063950/`。
其中 `summary.json` 保留当时门槛的结果，`cache_assessment.json` 是修正后的缓存评估。
这次专项评测脚本已随候选实现一并移除。

## 请求诊断

传输边界记录实际 provider/model、请求尝试、Epoch、工具顺序指纹、消息字段指纹、长度、
最长完整公共消息前缀和首个差异路径，以及返回的用量和延迟。不会记录完整 Prompt、reasoning 或附件。
对象键排序用于规范化，数组顺序保留。没有返回缓存用量时记录未知，不能当作零命中。

本地指纹只能诊断请求变化，不能证明服务端一定命中缓存。

官方依据：

- [DeepSeek Context Caching](https://api-docs.deepseek.com/guides/kv_cache/)
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Claude 工具搜索与缓存](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- [OpenAI Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
