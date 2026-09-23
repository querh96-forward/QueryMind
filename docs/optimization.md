# Agent执行控制与评测方法

## 设计范围

本文说明LangGraph运行控制、SQL工具执行和结果评测机制。系统使用FastAPI、PostgreSQL、Redis与Northwind业务数据；语义检索设计见 [semantic-retrieval.md](semantic-retrieval.md)。

## 执行行为

| 触发条件 | 当前行为 | 主要代码 |
|---|---|---|
| 字段/表结构或语法错误 | 分类并反馈修复建议，允许模型在限额内重查结构并修正 | `app/tool_errors.py`、`app/tools.py` |
| 工具参数不符合声明 | 执行前按 JSON Schema 拒绝参数，返回可修复错误 | `app/tools.py` |
| 权限拒绝、写操作、多语句 | 终止执行，不让模型继续绕过限制 | `app/northwind.py`、`app/graph.py` |
| 重复相同工具与参数 | 第三次请求跳过并终止 | `app/graph.py` |
| 超出工具次数、时间、Token | 明确停止原因；未执行的批量调用也补全工具响应 | `app/graph.py` |
| 第一次 SQL 失败、后来成功 | 首次结果保持失败，同时保留最终成功结果 | `app/graph.py` |
| SQL失败后仅查看结构 | 查看结构的成功不会掩盖未解决的SQL错误 | `app/graph.py` |
| SQL成功后查看结构或指标 | 保留最近一次SQL结果用于表格展示 | `app/graph.py` |
| 返回空结果 | 按正常执行处理，不自动放宽用户条件 | 提示词、图校验 |
| 显式 LIMIT 大于200 | 实际查询也限制到200行 | `app/northwind.py` |

恢复策略通过工具结果中的 `error_type`、`retryable`、`recovery_hint` 提供给模型。模型仍可能修复失败；代码不保证每次字段错误都能恢复。

`MAX_SQL_REPAIRS=2` 表示最多容许两次失败后的补救机会；累计第三次失败时停止。该计数是每次运行的累计失败次数。工具调用次数只计入实际分发尝试，被预算拦截的调用保存在审计中并标记 `skipped`。

配置上限通过 `.env` 传入 Docker：`MAX_AGENT_LOOPS`、`MAX_TOOL_CALLS`、`MAX_RUNTIME_SECONDS`、`MAX_SQL_REPAIRS`。修改配置后重新创建容器。

## 测试

```bash
# 完整测试：需要已启动 PostgreSQL 与 Redis
docker compose exec -T api python -m pytest -q tests

# 无数据库、无付费模型的运行控制和评测测试
python -m pytest -q tests/test_runtime_unit.py tests/test_sql_policy_unit.py tests/test_eval_unit.py
```

运行控制测试使用真正的 LangGraph 与 ToolRegistry，配合内存检查点和可控的模型/数据库替身。它们验证首次失败后修复、权限停止、完整工具响应、修复次数、重复调用、预算、取消、空结果及结果保存。测试通过只能说明这些程序行为通过断言。

## 真实模型评测

```bash
docker compose exec -T api python evals/run_eval.py --require-real --output /app/evals/reports/final-real.json
docker compose cp api:/app/evals/reports/final-real.json ./evals/final-real.json
```

`--require-real` 会在未配置真实模型或启用 Mock 时拒绝运行。每题创建独立会话，参考SQL仅在评测器执行，不传入Agent。

新题集 `portfolio_cases.json` 共24题：筛选、聚合、排名、日期条件、多表关联、空结果各4题。每类第4题标为保留题，共6题；其余18题为开发题。题目明确输出列、排序和指标口径，属于小规模受控业务查询，不代表开放式经营分析的通用表现。保留题应在开发结束后使用；若根据保留题结果改代码或提示词，应更换新的保留题。

| 指标 | 定义 |
|---|---|
| `result_accuracy` | 运行正常完成且有SQL证据，结果表与参考结果一致的题数 / 全部题数 |
| `first_sql_success_rate` | 首次SQL执行成功的题数 / 实际尝试SQL的题数；不等于答案正确率 |
| `recovery_result_accuracy` | 首次SQL失败但最终结果正确的题数 / 首次SQL失败的题数；分母为0时返回null |
| `avg_latency_ms` / `p95_latency_ms` | 从提交到评测器观测到终态的墙钟时间，含排队与轮询；超时计实际等待时长 |
| `total_tokens` / `avg_tokens` | 运行记录中的模型用量；供应商未返回用量时网关采用估算 |
| RAG指标 | 12题基础RAG题集上的Recall@5与MRR，独立于SQL正确率 |

有序问题按行顺序比较，无序问题保留重复行进行匹配；默认绝对数值容差0.005。列名需一致，只有题集中事先声明的别名映射生效。空参考结果不会使失败或未完成任务获得正确判定。

报告包含题目、参考/实际结果、实际SQL、模型、代码内容哈希、题集哈希、运行上限、逐题耗时和失败原因；按题写入以保留中断前进度。只有 `complete=true` 才是完整评测。

## 实际边界

- 执行完整性检查不证明SQL语义或自然语言结论正确；自然语言回答未自动评分。
- 时间与Token预算在调用边界检查，单次已发出的请求可能越过预算。模型网关禁用SDK自动重试并设置网络超时；SQL有数据库语句超时。
- 用户取消在后续调用边界生效，不能撤回已经发出的请求。
- 当前展示最近一次SQL结果，尚不支持完整的多查询证据选择与归因。
- 语义向量已在后续阶段接入并独立对照评测；简易上下文压缩和单Worker恢复机制仍需后续独立评估。
- 自建小样本表现不等于线上生产表现；没有本次修改前的同条件对照，报告只陈述最终值。
