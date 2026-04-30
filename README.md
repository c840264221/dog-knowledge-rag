# 🐶 Dog Knowledge RAG

一个基于 RAG（Retrieval-Augmented Generation）的狗狗百科智能问答系统，从数据抓取到问答生成实现完整闭环。

---

## 📌 项目简介

本项目从 AKC（American Kennel Club）官网抓取犬种数据，构建结构化 Markdown 知识库，并基于向量数据库与大语言模型实现智能问答系统。

用户可以输入自然语言问题，例如：

```text
Afghan Hound 性格怎么样？
```

系统将自动：

1. 识别犬种名称（支持中英文）
2. 从向量数据库检索相关内容
3. 使用 Reranker 精排结果
4. 结合 LLM 生成结构化回答

---

## 🚀 Features

* 🔎 支持自然语言问答
* 🌏 中英文犬种名称自动识别（Alias）
* 🧠 基于向量数据库的语义检索
* 🎯 结构化过滤（按犬种精准检索）
* 🔍 Reranker 精排优化结果
* 📄 Markdown 语义切块（提升检索质量）
* ⚡ 模型懒加载 + 本地缓存加速
* 🧩 模块化设计（易扩展）

---

## 💡 Highlights（面试亮点）

* 🏗️ **端到端RAG系统**：从数据抓取 → Markdown构建 → 向量检索 → 问答生成完整闭环
* 🧠 **自定义RAG Pipeline（LCEL）**：基于 LangChain 构建可扩展链式结构
* 🎯 **结构化检索增强**：通过 metadata filter 限定犬种范围，避免信息混淆
* 🌏 **Alias语义映射**：支持“金毛 → Golden Retriever”等自然语言映射
* 🔍 **Reranker精排机制**：引入 CrossEncoder 提升 Top-K 检索质量
* 📄 **Markdown结构切块**：基于标题层级进行语义分块，而非简单文本切分
* ⚡ **性能优化**：Embedding / Reranker 懒加载 + 本地缓存
* 🧩 **工程化设计**：模块清晰，接近真实生产系统架构

---

## 🚀 Key Innovations

* 将传统“向量检索”升级为“结构化过滤 + 向量检索”混合模式
* 使用 alias 解决用户输入与知识库字段不一致问题
* 引入 reranker 作为第二阶段排序，提高语义匹配精度
* 基于 Markdown 结构进行语义切块，提升检索质量

---

## 🏗️ 项目结构

```bash
.
├── scripts
│   ├── build_dog_info_db.py   # 构建向量数据库
│   └── run.py                 # 启动问答系统
│
├── src
│   ├── config.py              # 配置
│   ├── qa_chain.py            # 问答链模块
│
│   ├── crawler                # 数据抓取模块
│   │   ├── akc_spider.py
│   │   ├── parser.py
│   │   └── pipeline.py
│
│   ├── ingestion             # 数据处理
│   │   ├── markdown_file_loader.py
│   │   └── splitter.py       # md文件切块
│
│   ├── embedding
│   │   └── embedder.py       
│
│   ├── models
│   │   ├── llm.py
│   │   └── reranker.py
│
│   ├── retrieval
│   │   ├── retriever.py
│   │   ├── alias_loader.py   # 懒加载alias的json文件
│   │   └── extract_dog_name.py
│
│   └── vectorstore
│       └── vector_store.py   # 载入、获取向量数据库
```

---

## ⚙️ 安装与运行

### 1️⃣ 克隆项目

```bash
git clone git@github.com:c840264221/dog-knowledge-rag.git
cd dog-knowledge-rag
```

---

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 3️⃣ 配置环境变量（可选）

```bash
HF_TOKEN=你的token
```

用于加速模型下载（推荐）

---

### 4️⃣ 构建向量数据库

```bash
python scripts/build_dog_info_db.py
```

---

### 5️⃣ 启动问答系统

```bash
python scripts/run.py
```

---

## 🎬 Demo

输入：

```text
Afghan Hound 性格怎么样
```

输出：

```text
1. 性格独立且较为高冷
2. 对主人忠诚但不黏人
3. 具有较强猎犬本能
```

---

## 🧠 技术栈

* Python
* LangChain（LCEL）
* Chroma 向量数据库
* HuggingFace Embedding
* CrossEncoder Reranker
* Selenium（数据抓取）

---

## 📈 后续优化方向

* 🌐 Web UI（Gradio / Streamlit）
* 🤖 多轮对话（Memory）
* 📊 推荐系统（选狗助手）
* 🧠 Agent 化（自动任务流）
* ☁️ 云部署（API 服务）

---

## 📄 License

MIT License
