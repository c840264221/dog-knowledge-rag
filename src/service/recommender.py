def score_dog(dog, preferences):
    score = 0

    # 新手友好
    if preferences.get("beginner"):
        if float(dog.get("trainability", "0")) > 2:
            score += 2

    # 不掉毛
    if preferences.get("low_shedding"):
        if float(dog.get("shedding", "5")) < 3:
            score += 2

    # 安静
    if preferences.get("quiet"):
        if float(dog.get("barking", "5")) < 3:
            score += 1

    # 小型犬（简单判断）
    if preferences.get("small"):
        if float(dog.get("height", "20")) < 12:
            score += 1

    if preferences.get("big"):
        # if "20 pounds" in str(dog.get("weight", "")):
        if float(dog.get("height", "20")) > 20:
            score += 1

    return score


def recommend(dogs, preferences, top_k=5):
    scored = []

    for dog in dogs:
        s = score_dog(dog, preferences)
        if s > 0:
            scored.append((dog["name"], s))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]

def parse_preferences(text):
    text = text.lower()

    return {
        "beginner": "新手" in text,
        "low_shedding": "不掉毛" in text,
        "quiet": "安静" in text,
        "small": "小型" in text,
        "big": "大型" in text,
    }