from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


# 按字符切块
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# 用langchain的md文件切块库
def split_markdown(docs):
    headers = [
        ("#", "title"),
        ("##", "section"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

    all_chunks = []
    for doc in docs:
        dog_name = doc.metadata.get("source", "").split("\\")[-1].replace(".md", "")
        chunks = splitter.split_text(doc.page_content)

        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
            chunk.metadata["dog_name"] = dog_name
        all_chunks.extend(chunks)

    return all_chunks