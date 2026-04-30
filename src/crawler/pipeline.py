def dict_to_markdown(data: dict) -> str:
    md = []

    # 标题
    md.append(f"# {data.get('name', 'Unknown Dog')}\n")

    # 标签
    if "tag" in data:
        md.append("## 🏷️ 标签")
        md.append(f"{data['tag']}\n")

    # 基本信息
    md.append("## 📏 基本信息")
    if "height" in data:
        md.append(f"- 身高: {', '.join(data['height'])}")
    if "weight" in data:
        md.append(f"- 体重: {', '.join(data['weight'])}")
    if "life_span" in data:
        md.append(f"- 寿命: {data['life_span']}")
    md.append("")

    # 性格 / traits
    trait_keys = [
        k for k in data.keys()
        if k not in ["name", "tag", "height", "weight", "life_span",
                     "about_the_breed", "history",
                     "Health", "Grooming", "Exercise", "Training", "Nutrition"]
    ]

    if trait_keys:
        md.append("## 🧬 性格特征")
        for k in trait_keys:
            md.append(f"- {k}: {data[k]}")
        md.append("")

    # About
    if "about_the_breed" in data:
        md.append("## 📖 关于该犬种")
        md.append(data["about_the_breed"] + "\n")

    # History
    if "history" in data:
        md.append("## 🧾 历史")
        md.append(data["history"] + "\n")

    # Care
    md.append("## 🩺 养护指南")

    for key in ["Health", "Grooming", "Exercise", "Training", "Nutrition"]:
        if key in data:
            md.append(f"### {key}")
            md.append(data[key] + "\n")

    return "\n".join(md)

import os

def save_markdown(data: dict, output_dir="../../data/dog_markdown"):
    os.makedirs(output_dir, exist_ok=True)

    name = data.get("name", "unknown").replace(" ", "_")
    file_path = os.path.join(output_dir, f"{name}.md")

    md_content = dict_to_markdown(data)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ 已保存: {file_path}")

import re

def clean_text(text:str) -> str:
    if not text:
        return ""

    blacklist_keywords = [
        "How ",
        "A breed's level",
        "Dogs should always",
        "Some breeds",
        "This can include",
        "consider",
    ]
    for keyword in blacklist_keywords:
        if keyword in text:
            return text

    text = re.sub(r"\n+","\n", text)

    text = text.replace("\xa0", " ")
    return text.strip()

def clean_text2(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 🎯 匹配 ['xxx', 'yyy'] 这种结构
    def replace_func(match):
        inner = match.group(1)  # 'Wiry', 'Double'

        # 去掉引号并按逗号分割
        items = [item.strip().strip("'").strip('"') for item in inner.split(",")]

        return ", ".join(items)

    # 正则替换
    content = re.sub(r"\[(.*?)\]", replace_func, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# 🚀 批量处理目录
def clean_all_md():
    from src.config import DOG_MD_DATA_DIR

    for file in os.listdir(DOG_MD_DATA_DIR):
    # for root, _, files in os.walk(folder):
    #     for file in files:
    #         if file.endswith(".md"):
    #             path = os.path.join(root, file)
    #             print(f"处理: {path}")
        clean_text2(os.path.join(DOG_MD_DATA_DIR,file))


if __name__ == "__main__":
    # mock_data = {'name': 'Airedale Terrier', 'tag': 'friendly / courageous / clever', 'height': ['23 inches'], 'weight': ['50-70 pounds'], 'life_span': '11-14 years', 'Affectionate With Family': 'How affectionate a breed is likely to be with family members, or other people he knows well. Some breeds can be aloof with everyone but their owner, while other breeds treat everyone they know like their best friend.', 'Good With Young Children': "A breed's level of tolerance and patience with childrens' behavior, and overall family-friendly nature. Dogs should always be supervised around young children, or children of any age who have little exposure to dogs.", 'Good With Other Dogs': 'How generally friendly a breed is towards other dogs. Dogs should always be supervised for interactions and introductions with other dogs, but some breeds are innately more likely to get along with other dogs, both at home and in public.', 'Shedding Level': 'How much fur and hair you can expect the breed to leave behind. Breeds with high shedding will need to be brushed more frequently, are more likely to trigger certain types of allergies, and are more likely to require more consistent vacuuming and lint-rolling.', 'Coat Grooming Frequency': 'How frequently a breed requires bathing, brushing, trimming, or other kinds of coat maintenance. Consider how much time, patience, and budget you have for this type of care when looking at the grooming effort needed. All breeds require regular nail trimming.', 'Drooling Level': "How drool-prone a breed tends to be. If you're a neat freak, dogs that can leave ropes of slobber on your arm or big wet spots on your clothes may not be the right choice for you.", 'Coat Type': "Canine coats come in many different types, depending on the breed's purpose. Each coat type comes with different grooming needs, allergen potential, and shedding level. You may also just prefer the look or feel of certain coat types over others when choosing a family pet.", 'Coat Length': "How long the breed's coat is expected to be. Some long-haired breeds can be trimmed short, but this will require additional upkeep to maintain.", 'Openness To Strangers': 'How welcoming a breed is likely to be towards strangers. Some breeds will be reserved or cautious around all strangers, regardless of the location, while other breeds will be happy to meet a new human whenever one is around!', 'Playfulness Level': 'How enthusiastic about play a breed is likely to be, even past the age of puppyhood. Some breeds will continue wanting to play tug-of-war or fetch well into their adult years, while others will be happy to just relax on the couch with you most of the time.', 'Watchdog/Protective Nature': "A breed's tendency to alert you that strangers are around. These breeds are more likely to react to any potential threat, whether it's the mailman or a squirrel outside the window. These breeds are likely to warm to strangers who enter the house and are accepted by their family.", 'Adaptability Level': 'How easily a breed handles change. This can include changes in living conditions, noise, weather, daily schedule, and other variations in day-to-day life.', 'Trainability Level': 'How easy it will be to train your dog, and how willing your dog will be to learn new things. Some breeds just want to make their owner proud, while others prefer to do what they want, when they want to, wherever they want!', 'Energy Level': "The amount of exercise and mental stimulation a breed needs. High energy breeds are ready to go and eager for their next adventure. They'll spend their time running, jumping, and playing throughout the day. Low energy breeds are like couch potatoes - they're happy to simply lay around and snooze.", 'Barking Level': "How often this breed vocalizes, whether it's with barks or howls. While some breeds will bark at every passer-by or bird in the window, others will only bark in particular situations. Some barkless breeds can still be vocal, using other sounds to express themselves.", 'Mental Stimulation Needs': "How much mental stimulation a breed needs to stay happy and healthy. Purpose-bred dogs can have jobs that require decision-making, problem-solving, concentration, or other qualities, and without the brain exercise they need, they'll create their own projects to keep their minds busy -- and they probably won't be the kind of projects you'd like.", 'about_the_breed': "His size, strength, and unflagging spirit have earned the Airedale Terrier the nickname 'The King of Terriers.'\x9d The Airedale stands among the world's most versatile dog breeds and has distinguished himself as hunter, athlete, and companion.The Airedale Terrier is the largest of all terrier breeds. Males stand about 23 inches at the shoulder, females a little less. The dense, wiry coat is tan with black markings. Long, muscular legs give Airedales a regal lift in their bearing, and the long head'¿with its sporty beard and mustache, dark eyes, and neatly folded ears'¿conveys a keen intelligence. Airedales are the very picture of an alert and willing terrier'¿only bigger. And, like his smaller cousins in the terrier family, he can be bold, determined, and stubborn. Airedales are docile and patient with kids but won't back down when protecting hearth and home. Thanks to their famous do-it-all attitude, Airedales excel in all kinds of sports and family activities.", 'Health': "Airedales are generally healthy dogs, and responsible breeders will test for health concerns such as hip dysplasia, a malformation of the hip joint. An Airedale's ears should be checked regularly to remove foreign matter and avoid a buildup of wax, and his teeth should be brushed regularly.", 'Grooming': "The Airedale has a short, wiry coat that needs relatively little maintenance. Weekly brushing keeps the coat looking good and has the additional advantage of removing dead hair that would otherwise be shed around the house. (Some people with dog allergies have found that they can share a living space with a well-brushed Airedale without suffering any symptoms.) If the weekly session turns up any mats, they should be broken up with the fingers and then teased apart with a comb. Full grooming '¿ where the dog is bathed, brushed, and stripped or clipped '¿ should be done three or four times a year, either by the owner or a professional groomer.", 'Exercise': "Terriers are generally known for their high energy levels. Given that the Airedale is the largest of all terriers, that energy must be channeled into safe outlets. Fortunately, Airedales love to play with other family members. A daily play session of moderate length, in addition to walks (or backyard time) several times a day, should be enough to satisfy the Airedale's exercise requirements. Airedales play well with children, but interactions with toddlers and smaller children should be closely supervised. Airedales are rangy but strong; that strength, combined with a boisterous personality, can lead to mishaps.", 'Training': "Because of the Airedale's size (he is a medium-sized dog, but the largest of the terrier breeds), strength, and rambunctiousness, obedience training is highly recommended. At a minimum, an Airedale should learn basic obedience commands such as come, sit, and stay. The breed's intelligence and the fact that they bond closely with family members can make training easy. Owners and trainers should keep in mind that an intelligent dog is an easily bored dog, so varied training sessions will be more successful than repetitive ones. Also, an easily bored dog who is often left alone for long periods of time will tend to develop undesirable behaviors. It often helps to provide the dog with challenging toys that will keep him happily occupied.", 'Nutrition': "The Airedale Terrier should do well on a high-quality dog food, whether commercially manufactured or home-prepared with your veterinarian's supervision and approval. Any diet should be appropriate to the dog's age (puppy, adult, or senior). Some dogs are prone to getting overweight, so watch your dog's calorie consumption and weight level. Treats can be an important aid in training, but giving too many can cause obesity. Learn about which human foods are safe for dogs, and which are not. Check with your vet if you have any concerns about your dog's weight or diet. Clean, fresh water should be available at all times.", 'history': "One of Britain's key manufacturing centers before the turn of the 20th century, the Aire Valley lies in the north of England, less than a hundred miles from the Scottish border. In the mid-1800s, at the height of the Industrial Revolution, the valley's factory hands and mill workers first bred tough, devil-may-care Airedales in their own image. (It's a neat irony that the King of Terriers was developed not by royals but workingmen.) Airedales were created to be large and fearless hunters of ducks and rats, and no doubt did some poaching on the side. Fittingly for a dog from a manufacturing community, this was a 'manufactured' breed: Several existing breeds went into its makeup. These include the Otterhound and such now-extinct breeds as the English Black and Tan Terrier. A practiced eye can also spot traces of the Irish Terrier and Bedlington Terrier in the Airedale. It's quite possible that setters, retrievers, and herding dogs also played a part in the development of the breed. Airedales served in the British Armed Forces with distinction during the First World War as messengers, sentries, and guard dogs. In North America, the breed was known as a three-in-one hunting dog, equally adept on waterfowl, upland gamebirds, and furred prey. This will come as no surprise to anyone who has ever owned an Airedale, as the breed's versatility is legendary. Ratter, duck dog, big-game hunter, herder, guardian, warrior, actor, athlete, K-9 cop, and babysitter are all jobs held at various times by the mighty King of Terriers."}
    # save_markdown(mock_data)
    clean_all_md()