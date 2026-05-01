import json
import os
from src.service.extract_data import extract_name, extract_all_fields


# 根据md文件内容提取相应的数据 输出为json格式
def load_mddata_to_json():
    from src.config import DOG_MD_DATA_DIR
    dogs = []
    for file in os.listdir(DOG_MD_DATA_DIR):
        if not file.endswith(".md"):
            continue

        path = os.path.join(DOG_MD_DATA_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        dog = parse_md_to_struct(text)
        dogs.append(dog)
    return dogs

def parse_md_to_struct(text):
    result = {"name": extract_name(text)}
    result.update(extract_all_fields(text))
    return result

def load_json_data():
    from src.config import DOG_DATA_JSON_PATH
    with open(DOG_DATA_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    dog_map = {
        d["name"]: d for d in data
    }
    return dog_map


if __name__ == "__main__":
    # dogs = load_mddata_to_json()
    load_json_data()