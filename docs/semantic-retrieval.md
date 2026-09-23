# 语义检索与版本化索引

QueryMind 的默认向量召回已从 256 维本地 token 哈希改为 OpenAI-compatible `text-embedding-v4` 1024 维语义向量。文档与用户问题使用同一个提供方、模型和维度，与 `pg_trgm` 字符相似度通过等权 RRF（常数 60、每路候选上限 max(20, top_k×4)）融合。关键词相似度为零的文档不参与融合，避免无匹配文档凭 ID 排序获得票数；没有有效关键词候选时，结果自然等同于语义向量检索。

## 配置

`.env` 与 `.env.example` 具有相同配置项；实际密钥仅在被 Git / Docker 忽略的 `.env` 中。聊天使用 `LLM_*`，向量服务使用 `EMBEDDING_*`，两者可使用不同供应商。旧 `DASHSCOPE_*` 名称只作为聊天配置兼容别名。`openai` 表示兼容协议，不要求使用 OpenAI 提供的模型。

默认维度 1024、每批最多 10 条、单次网络超时 30 秒、SDK 最多重试 1 次。请求显式指定 float 编码；返回值检查索引连续性、数量、维度、有限数值和非零向量。依据 [阿里云 Embedding 接口文档](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api/)配置批量上限。任何真实服务错误直接失败，不静默切换到哈希向量。

## 数据与缓存

- `app/embeddings.py` 负责接口调用、分批校验、真实用量和显式哈希基线。
- `app/rag.py` 的索引标识由提供方、模型、维度、端点摘要、知识库内容摘要和索引格式版本生成，不包含 API Key。轮换密钥不强制重建向量。
- 每代使用独立的 `rag_knowledge_<version>` 表，并在 `rag_index_versions` 记录模型元信息、构建 Token 数。先取得完整向量，后在数据库事务内建表、插入、建索引并登记；事务失败不会暴露半成品。
- 同一版本构建受 PostgreSQL advisory lock 保护，重启复用已经完成的索引；并发首次构建仍可能重复调用向量接口，但不会重复落表。
- 原 `rag_knowledge` 表及旧版本表保留；不会删除业务数据。后续大量更新知识库时需要增加旧索引清理策略。
- Redis 键包含索引版本、查询、top-k、知识类型筛选和检索策略。不同模型、语料和策略不会共用结果；缓存命中不会再请求向量服务。
- Context 快照记录索引版本、缓存状态、Embedding Token；运行 Token 与费用估算包含查询向量用量。文档索引构建费用单独记录，不归入单次用户查询。费用系数是可配置估算值，真实账单以供应商为准。

语料变更会重建该代完整索引，目前 48 条知识，不实现增量文档更新和跨实例热切换。修改配置或语料后，重新运行 `docker compose up -d --build` 让 bootstrap 完成索引，并让 API/Worker 一起切换版本。

## 验证与复现

```bash
# 先启动并完成索引初始化
docker compose up -d --build
# 所有自动化测试，使用离线模型替身，不消耗模型额度
docker compose exec -T api python -m pytest -q tests
# 使用真实 Embedding，同题配对比较哈希256 / 语义1024，分别评估向量召回与混合召回
docker compose exec -T api python -m evals.run_retrieval_eval --output /app/evals/reports/semantic-retrieval.json
# 使用真实聊天与向量模型检查 24 道 SQL 的最终结果表
docker compose exec -T api python -m evals.run_eval --require-real --output /app/evals/reports/semantic-real.json
# 报告从容器取回
docker compose cp api:/app/evals/reports/semantic-retrieval.json ./evals/reports/semantic-retrieval.json
docker compose cp api:/app/evals/reports/semantic-real.json ./evals/reports/semantic-real.json
```

首次 12+20 题的 Embedding 替换实验保存在 `semantic-retrieval-embedding-only.json`。这轮暴露了关键词零相似度补位的问题，修正后保留原题作为诊断集，并在最终运行前固定了新增 16 题作为本轮保留题，总计 48 题。未修改知识库、RRF 权重、候选上限或标签。最终报告包含哈希/语义两种向量 × 原融合/修正融合/纯向量三种策略的六组对照，明确区分换模型与修正融合的影响。

所有检索对照关闭结果缓存，每个问题只有一个标注目标，报告 Recall@1、Recall@5、MRR@5、耗时和查询向量 Token。诊断题已经参与错误定位，不作为未见测试；保留题应单独报告。单一目标标签可能不覆盖所有合理知识，且只有 48 题，不能外推线上泛化能力。

`evals/reports/final-real.json` 是上一阶段的历史报告；`semantic-real.json` 是改造后的复测。聊天模型具有随机性，两次 SQL 正确率差异不能单独归因于检索变化。当前端到端指标仅评分结果表，不评分回答自然语言，也不证明工具错误恢复率。
