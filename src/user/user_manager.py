import json
import uuid
from pathlib import Path
from src.config import USER_FILE



def get_user_id():

    user_file = Path(USER_FILE)
    # 已存在用户
    if user_file.exists():

        with open(user_file, "r", encoding="utf-8") as f:

            data = json.load(f)

            return data["user_id"]

    # 新用户

    user_id = str(uuid.uuid4())

    user_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(user_file, "w", encoding="utf-8") as f:

        json.dump(
            {
                "user_id": user_id
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return user_id