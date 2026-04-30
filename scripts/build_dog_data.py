import json
from pathlib import Path

from src.service.dog_data_loader import load_dog_data
from src.config import DOG_DATA_JSON_PATH


def build():
    dogs = load_dog_data()

    # 创建目录
    Path(DOG_DATA_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 写入 JSON
    with open(DOG_DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dogs, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 {DOG_DATA_JSON_PATH}，共 {len(dogs)} 条数据")


if __name__ == "__main__":
    import os
    from src.config import DOG_MD_DATA_DIR
    files = os.listdir(DOG_MD_DATA_DIR)
    print(len(files))
    # build()