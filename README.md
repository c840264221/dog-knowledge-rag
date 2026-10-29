# Dog Agent Framework · 养宠知识与任务协作助手

面向养宠资料咨询、用品准备、预算核算和支出管理，构建集**知识问答与任务执行**于一体的助手。项目从犬种知识 RAG（检索增强生成）逐步演进为基于 LangGraph 的多智能体系统，覆盖查询理解、双语检索、工具调用、人工确认、任务恢复与后台执行。

这是个人持续开发与验证的工程项目。以下结果来自本地实际案例和开发评估，不代表商业客户交付、生产吞吐量或线上准确率。

## 业务场景

|场景|用户可以怎样提问|系统如何处理|
|---|---|---|
|犬种与养护问答|金毛日常运动有什么要求？接猫回家要准备什么？|拆分检索子问题，选择集合，中英文召回、融合与重排，按主题组织证据和出处|
|用品准备与预算|准备接猫回家，预算300元，我查到了几项用品价格，帮我选选并算算。|先查养护资料，再筛选价格表中的用品；调用本地预算工具，区分资料支持与推断项|
|自然语言记账|昨天吃饭20、打车15、狗子洗澡80，帮我记账。|Skill 指导分类和日期提取；缺参先澄清，完整账单经确认后通过 MCP 写入教学账本|
|多主题协作|接猫路上怎么运输？家里有狗，第一次接触要注意什么？|规划两个无依赖知识任务，并发调度，隔离各步骤引用，统一汇总|
|长任务|先整理接猫条件，再生成有先后顺序的待办。|支持计划阶段后台执行，以及前台预算耗尽后在步骤边界移交，由 Redis 工作进程继续推进|

预算核算只使用提供的价格，不代表下单或实际消费。知识问答依据资料回答，不替代兽医诊疗。

## 架构与职责

```text
用户请求 / 多轮补充
        ↓
运行时上下文、记忆与任务恢复入口
        ↓
RootAgent（根调度器：粗路由）
        ├── 知识智能体
        │     查询改写 / 子问题拆分 / 集合选择
        │       → 中英文多集合召回 → 排名融合 → 分组重排
        │       → 按主题选择证据 → 带引用回答
        ├── ToolAgent（工具智能体）
        │     技能与工具目录 → 调用提议 → 参数校验
        │       → 补参 / 人工确认 → 本地工具或 MCP → 回执与回答
        ├── 通用问答
        └── 多智能体编排
              Planner（规划器）→ Scheduler（调度器）→ Aggregator（聚合器）
                                      ↓
                            Redis 后台长任务
```

### 多格式资料与 RAG

- 已处理280余份资料，主体为犬种 Markdown，并补充纯文本 PDF、图文混排 PDF 的实际处理案例。
- Markdown 按标题结构切块；PDF 文本按物理页组织并受长度与重叠限制，保留来源、页码、切块策略与向量配置信息。
- 图文 PDF 通过页面转 PNG、视觉模型结构化提取及人工修正，建立文字与页面图片的关联引用。当前为**文字向量检索＋关联图片供模型阅读**，不声称使用图片向量检索。
- 统一查询理解支持有界子问题拆分、查询改写、英文检索文本与集合选择；使用 Chroma、元数据过滤、双语候选排名融合及 BGE 重排。
- 已验证 BM25（关键词检索）对照方案；当前主链路以双语向量检索与重排为主，未把实验方案全部默认启用。
- 提供单文档更新、纯标题块审计、集合复制、向量配置一致性及精确距离排名对照脚本。

### Skill 与统一工具入口

- 文件化 Skill（技能）由 `SKILL.md`、元信息与参考资料组成，经适用性检查后注入模型上下文；技能说明不代替程序授权。
- 已创建自然语言支出记账、接宠用品准备与预算技能，新增技能优先复用通用加载和规划流程。
- 本地预算工具采用十进制金额计算；自建 MCP（模型上下文协议）服务提供单笔、批量记账及账单查询，工具分发器统一连接本地与远端能力。
- 输入参数校验、写入前人工确认、动作幂等键和参数指纹校验共同约束执行，依据持久化回执确认结果。
- 支持普通新一轮补参及检查点确认恢复，已结束会话再次打开只展示结果。自然语言多字段补参仍有扩展空间。

### 多智能体与长任务

- 规划器生成步骤、执行者、工具或能力模式、输入与依赖；结构校验不等于业务计划一定正确。
- 调度器执行无依赖的就绪批次，依赖任务等待前置结果；工具明确的步骤与能力委派步骤采用不同执行约束，避免后续解析任意改写工具选择。
- 聚合器依据实际结果生成回答并保留证据与限制，已运行知识→预算、本地计算→确认记账→回查、双主题知识并发等案例。
- Redis 持久化任务、队列消息与可靠提交回执，后台工作进程领取及推进；支持重试、租约与版本约束、死信和诊断信息。
- 前台时间预算移交发生在步骤边界，不宣称能强制中止任意同步函数或正在执行的外部服务。

### 运行时与服务工程

Container / Provider（容器／服务提供者）统一管理模型、向量库、客户端和图服务生命周期，结合运行时上下文、日志、追踪、时间线与调试报告排障。

包含 FastAPI 接口、SSE（服务端事件流）、Docker Compose、Nginx 和 GitHub Actions 配置。当前聊天 SSE 主要输出请求生命周期事件，不等同于模型逐字流；接口进程数和部署边界以实际配置及校验为准。

## 实际排障与开发评估

### 跨语言漏召回与重排退化

沿“最终回答→实际上下文→重排名次→召回候选”逆向排查。英文资料配中文问题时，既出现候选未召回，也出现相关资料在旧重排模型后跌出前五。固定候选对比问题语言与重排模型，改用 `BAAI/bge-reranker-v2-m3`；再对照中文单路、双语融合及重排，区分漏召回和排序截断。

2026-10-26 开发评估：14个请求，18项预标注证据需求；四组复用同一查询计划，最终最多选择5块。

|方案|前五证据命中|命中率|MRR@5（首个目标排名倒数的平均值）|
|---|---:|---:|---:|
|中文召回|16/18|88.89%|0.4713|
|双语排名融合|16/18|88.89%|0.6130|
|中文＋BGE重排|17/18|94.44%|0.7963|
|双语融合＋BGE重排|18/18|100.00%|0.8426|

这是小规模开发集的**证据命中**，不是独立测试集、最终答案准确率或线上结果；双语候选预算更大，组合收益不能全部归因于翻译。

### 工具与异常恢复

实际 ToolAgent 子图、模型及独立 MCP 账本评估：正常任务12/12、异常处理8/8符合当前检查预期，其中可恢复场景5/5完成。覆盖预算、单笔／批量记账、查询、缺金额／日期、拒绝确认、未来计划、重复键、同键冲突、检查点连接重建及保存后受控响应不可用。

未经过根路由和前端；部分协议场景直接调用 MCP。受控响应不可用不是真实断网，检查点连接重建不是完整进程崩溃测试。评估器也经过默认值和取消语义校准。

详见 [案例与评估口径](docs/project/CASES_AND_EVALUATION.md)。评估输入随仓库分发，原始个人会话、账本及完整本地运行产物不上传。

## 安装与运行

主要本地验证环境为 Windows / Python 3.13。模型下载、外部 API 和部分实验需要额外资源；真实模型入口会产生调用费用。

```bash
git clone https://github.com/c840264221/dog-knowledge-rag.git
cd dog-knowledge-rag
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# Linux / macOS：source .venv/bin/activate
python -m pip install -r requirements-ci.txt
# PDF 页面图片教学流程额外安装：
python -m pip install -r requirements-pdf.txt
```

复制 `.env.example` 为 `.env`，按 `src/settings/` 和实际服务配置密钥、模型与地址。兼容服务使用 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`；模型名以配置为准，可通过 `LLM_MAIN_MODEL` 等配置项覆盖。部分入口还使用本地 Ollama 备用服务，应先核对依赖。不要上传真实 `.env`。

### 本地预算演示（不连接模型）

```bash
python -m tests.learning.plus_supply_budget_demo
```

### MCP 记账与主图

终端一启动本机教学服务：

```bash
python -m tests.learning.plus_expense_mcp_server
```

终端二（PowerShell）：

```powershell
$env:MCP_EXPENSE_ENABLED = "true"
python -m scripts.run_expense_main --question "昨天狗子洗澡80元，帮我记账"
# 补参时保留输出的会话编号：
python -m scripts.run_expense_main --thread-id <会话编号> --reply "80"
# 恢复人工确认等待：
python -m scripts.run_expense_main --thread-id <会话编号>
```

服务默认仅监听本机，固定教学身份，没有生产认证；不要直接对公网开放。记账会真实写入教学账本，确认前应核对金额、用途及日期。

### 知识入库与协作

```bash
python -m scripts.rag.rebuild_dog_rag_index
python -m scripts.rag.index_pet_pdf
python -m scripts.rag.index_pet_pdf --apply
python -m scripts.run_knowledge_query_case
python -m scripts.run_pet_supply_collaboration_case
python -m scripts.run_pet_knowledge_parallel_case
```

新克隆不包含本机向量数据库。犬种入口使用仓库 Markdown 重建；养护 PDF 请按 [入库计划](data/learning/pet_corpus_v1/ingestion_plan.json) 的来源自行获取并核对摘要。默认知识目录指向 `breed_no_heading_v1`、`pet_care_v1`，重建／清理后的实际集合名需与 `src/settings/rag.py` 的目录配置一致。部分对照脚本依赖前序报告，不能跳过数据准备直接运行。

### 评估与后台任务

```bash
python -m scripts.rag.evaluate_unified_knowledge
python -m scripts.evaluation.evaluate_real_tools
python -m scripts.run_long_task_worker
python -m scripts.run_planned_background_case
```

知识评估前需准备相应资料及集合；工具评估启动独立服务并写本轮数据库。后台任务还需启用并配置 Redis，工作进程持续运行。运行产物默认留在本地 `output/`。

## 代码导航

```text
src/agents/root_agent/          根路由
src/agents/dog_knowledge_agent/  统一知识子图与查询理解
src/agents/tool_agent/          工具规划、校验、澄清、确认和执行
src/agents/collaboration/       规划、调度、聚合和执行适配
src/rag/                       加载、切块、入库、召回及上下文管线
src/skills/ + skills/           技能加载与可复用技能文件
src/mcp/                       MCP 接入、分发和记账服务
src/runtime/                   容器、上下文、可观测性和长任务
src/api/                       HTTP 与 SSE 服务
scripts/                       主图、案例、索引维护和评估入口
data/evaluation/               开发评估输入
tests/unit/                    自动化单元检查
tests/learning/                分步骤学习和手动演示入口
```

## 检查与部署

```bash
python -m compileall -q src scripts tests
python -m pytest tests/unit -q
docker compose up -d --build
```

自动化检查与真实模型评估分开，模型效果以当次报告复核，不能用替身单元测试替代。Docker 启动前需准备私有配置与挂载数据；镜像发布和代理入口参见 `compose.release.yaml`、`compose.proxy.yaml` 及 `deployment/`。

## 技术栈与后续方向

Python · LangChain / LangGraph · Pydantic / JSON Schema · Chroma · Hugging Face Embedding · BGE Reranker · MCP · SQLite · Redis · FastAPI · Docker Compose · Nginx · pytest · GitHub Actions。

后续优先扩充独立测试集、完善集合目录及自然语言多字段补参，继续验证权限、并发和部署边界。仓库保留部分历史兼容模块，新增能力优先复用当前主链路。