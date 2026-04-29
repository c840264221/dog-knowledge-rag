from langchain_community.document_loaders import TextLoader
import os


def load_markdown_files(md_dir):
    if not os.path.exists(md_dir):
        print(f"找不到指定路径：{md_dir}")
    docs = []
    for file in os.listdir(md_dir):
        if file.endswith(".md"):
            path = os.path.join(md_dir, file)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    return docs


if __name__ == "__main__":
    load_markdown_files("../../data/dog_markdown")