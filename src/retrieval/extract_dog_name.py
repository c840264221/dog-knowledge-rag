import os
from src.config import DOG_NAME_JSON_PATH, DOG_MD_DATA_DIR, BASE_DIR

def extract_dog_names(md_dir):
    dog_names = []

    for file in os.listdir(md_dir):
        if file.endswith(".md"):
            name = file.replace(".md", "")
            name = name.replace("_", " ")
            dog_names.append(name)

    return dog_names

import json

def save_dog_names(names):
    with open(DOG_NAME_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dog_md_data_dir = os.path.join(BASE_DIR, DOG_MD_DATA_DIR)
    dog_name_list = extract_dog_names(dog_md_data_dir)
    save_dog_names(dog_name_list)