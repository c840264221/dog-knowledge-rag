import os
from src.service.extract_data import extract_name, extract_all_fields


def load_dog_data():
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
    # return {
    #     "name": extract_name(text),
    #     "Trainability Level": extract_all_fields(text),
    #     "Shedding Level": extract_all_fields(text),
    #     "Barking Level": extract_all_fields(text),
    # }
    result = {"name": extract_name(text)}
    result.update(extract_all_fields(text))
    return result

if __name__ == "__main__":
    dogs = load_dog_data()
