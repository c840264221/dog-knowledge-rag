import re
import os


def extract_name(md_text: str, file_path: str = None) -> str:
    """
    提取狗狗名称（优先级：标题 > 文件名 > 第一行）
    """

    # 1️⃣ 标题提取
    match = re.search(r"^#+\s+(.+)", md_text, re.MULTILINE)
    if match:
        name = match.group(1).strip()
        return clean_name(name)

    # 2️⃣ 文件名 fallback
    if file_path:
        name = os.path.basename(file_path).replace(".md", "")
        return clean_name(name)

    # 3️⃣ 第一行 fallback
    for line in md_text.splitlines():
        line = line.strip()
        if line:
            return clean_name(line)

    return "Unknown"


def clean_name(name: str) -> str:
    """
    清洗名称
    """

    # 去掉奇怪符号
    name = re.sub(r"[^\w\s\-]", "", name)

    # 多空格压缩
    name = re.sub(r"\s+", " ", name)

    return name.strip()

def extract_field(md_text: str, field_name: str) -> str:
    """
    更鲁棒版本：支持不同级别标题 & 模糊匹配
    """
    height = extract_height_avg(md_text, field_name)
    if height:
        return str(height)

    # pattern = rf"-+\s*{re.escape(field_name)}.*?(.*?)(?=\n#+|\Z)"
    pattern = rf"-\s*{re.escape(field_name)}\s*:\s*(.*?)(?=\n-\s|\Z)"

    match = re.search(pattern, md_text, re.IGNORECASE | re.DOTALL)

    if match:
        content = match.group(1).strip()
        return clean_text(content)

    return ""

def clean_text(text: str) -> str:
    """
    清洗 Markdown 内容
    """

    # 去掉多余换行
    text = re.sub(r"\n+", " ", text)

    # 去掉多余空格
    text = re.sub(r"\s+", " ", text)

    return text.strip()

FIELD_MAP = {
    "trainability": "Trainability Level",
    "shedding": "Shedding Level",
    "barking": "Barking Level",
    "height": "身高",
}


def extract_all_fields(md_text: str):
    result = {}

    for key, field_name in FIELD_MAP.items():
        result[key] = extract_field(md_text, field_name)

    return result

def extract_height_avg(text: str, field_name):
    # 1️⃣ 只匹配 “- 身高: xxx” 这一行
    pattern = rf"-\s*{re.escape(field_name)}:\s*(.+)"
    # match = re.search(r"-\s*身高:\s*(.+)", text)
    match = re.search(pattern, text)
    if not match:
        # print("未匹配数据")
        return None

    height_str = match.group(1)

    # 2️⃣ 提取数字 / 区间
    parts = re.findall(r"\d+\.?\d*(?:-\d+\.?\d*)?", height_str)
    # print("parts:", parts)

    values = []
    for part in parts:
        if "-" in part:
            a, b = part.split("-")
            values.append((float(a) + float(b)) / 2)
        else:
            values.append(float(part))

    if not values:
        return None

    return sum(values) / len(values)

if __name__ == "__main__":
    data = "- 身高: 9-11.5 inches\n- 体重: 7-100 pounds\n- 寿命: 12-15 years"
    result = extract_all_fields(data)
    print(result)