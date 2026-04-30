def score_dog(dog, preferences):
    score = 0

    # 新手友好
    if preferences.get("beginner"):
        if "easy" in dog.get("Trainability Level", "").lower():
            score += 2

    # 不掉毛
    if preferences.get("low_shedding"):
        if "low" in dog.get("Shedding Level", "").lower():
            score += 2

    # 安静
    if preferences.get("quiet"):
        if "low" in dog.get("Barking Level", "").lower():
            score += 1

    # 小型犬（简单判断）
    if preferences.get("small"):
        if "20 pounds" in str(dog.get("weight", "")):
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
    }