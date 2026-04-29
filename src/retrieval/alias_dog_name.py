import json

from sympy.codegen.ast import continue_


def generate_alias_basic(name):
    alias = set()

    name_lower = name.lower()
    alias.add(name_lower)

    # 拆词
    parts = [x for x in name_lower.split() if x not in ["and", "dog", "the", "a"]]
    alias.update(parts)

    return alias


REMOVE_WORDS = [
    "dog", "hound", "terrier", "spaniel", "retriever"
]

# 去后缀
def remove_suffix(name):
    words = name.lower().split()
    return " ".join([w for w in words if w not in REMOVE_WORDS])

# 首字母缩写
def generate_abbr(name):
    abbr = "".join([w[0] for w in name.lower().split()])
    if len(abbr) > 1:
        return abbr
    else:
        return ''

def generate_alias(name):

    alias = generate_alias_basic(name)

    # 去后缀
    simplified = remove_suffix(name)
    alias.add(simplified)

    # 缩写
    if generate_abbr(name):
        alias.add(generate_abbr(name))

    return {name:list(alias)}

def save_to_json_file(en_alias_dog_name_data):
    from src.config import DOG_NAME_ALIAS_JSON_PATH

    with open(DOG_NAME_ALIAS_JSON_PATH, "r", encoding='utf-8') as f:
        alias_dog_name_data = json.load(f)
        dog_name = list(en_alias_dog_name_data.keys())[0]
        try:
            alias_dog_name_data[dog_name].extend(en_alias_dog_name_data[dog_name])
        except KeyError as e:
            # print(f"{dog_name}原文档没有中文名，已将英文alias存入......")
            # alias_dog_name_data[dog_name] = en_alias_dog_name_data[dog_name]
            return

    with open(DOG_NAME_ALIAS_JSON_PATH, "w", encoding='utf-8') as f:
        json.dump(alias_dog_name_data, f, ensure_ascii=False, indent=2)

def get_all_dog_names():
    from src.config import DOG_NAME_JSON_PATH

    with open(DOG_NAME_JSON_PATH, "r", encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    from src.config import DOG_NAME_ALIAS_JSON_PATH, DOG_MD_DATA_DIR, BASE_DIR
    import json, os
    # names = get_all_dog_names()
    # # print(names)
    # for name in names:
    #     print(f"{name}开始存入......")
    #     en_alias_data = generate_alias(name)
    #     save_to_json_file(en_alias_data)
    #     print(f"🔚  {name}存入成功......")
    new_data = {
  "Hamiltonstovare": ["hamiltonstovare","哈密尔顿猎犬"],
  "Hanoverian Scenthound": ["scenthound","hs","hanoverian scenthound","hanoverian","汉诺威嗅猎犬"],
  "Harrier": ["harrier","猎兔犬"],
  "Hovawart": ["hovawart","霍瓦特犬"],
  "Icelandic Sheepdog": ["icelandic sheepdog","icelandic","sheepdog","is","冰岛牧羊犬"],
  "Irish Red and White Setter": ["white","and","setter","iraws","red","irish red and white setter","irish","爱尔兰红白雪达犬"],
  "Irish Terrier": ["it","irish","irish terrier","terrier","爱尔兰梗"],
  "Irish Water Spaniel": ["water","spaniel","irish water","irish water spaniel","irish","iws","爱尔兰水猎犬"],
  "Jagdterrier": ["jagdterrier","德国猎梗"],
  "Japanese Akitainu": ["ja","akitainu","japanese akitainu","japanese","日本秋田犬"],
  "Japanese Chin": ["jc","chin","japanese","japanese chin","日本狆"],
  "Japanese Terrier": ["japanese terrier","jt","terrier","japanese","日本梗"],
  "Kai Ken": ["kai","ken","kk","kai ken","甲斐犬"],
  "Karelian Bear Dog": ["dog","karelian bear","kbd","bear","karelian bear dog","karelian","卡累利阿熊犬"],
  "Kerry Blue Terrier": ["kbt","blue","kerry blue","kerry blue terrier","kerry","terrier","凯利蓝梗"],
  "Kishu Ken": ["ken","kishu ken","kishu","kk","纪州犬"],
  "Korean Jindo Dog": ["jindo","korean jindo","dog","kjd","korean","korean jindo dog","韩国珍岛犬"],
  "Kromfohrlander": ["kromfohrlander","克龙弗兰德犬"],
  "Kuvasz": ["kuvasz","库瓦兹犬"],
  "Lagotto Romagnolo": ["lr","romagnolo","lagotto","lagotto romagnolo","罗马涅水犬"],
  "Lakeland Terrier": ["lakeland terrier","terrier","lt","lakeland","湖畔梗"],
  "Lancashire Heeler": ["heeler","lh","lancashire","lancashire heeler","兰开夏牧牛犬"],
  "Lapponian Herder": ["herder","lapponian","lh","lapponian herder","拉普牧犬"],
  "Large Munsterlander": ["large","large munsterlander","lm","munsterlander","大型明斯特兰德犬"],
  "Löwchen": ["löwchen","小狮子犬"],
  "Manchester Terrier (Standard)": ["manchester (standard)","(standard)","manchester","mt(","manchester terrier (standard)","terrier","曼彻斯特梗（标准型）"],
  "Manchester Terrier (Toy)": ["manchester (toy)","manchester","mt(","terrier","manchester terrier (toy)","(toy)","曼彻斯特梗（玩具型）"],
  "Miniature American Shepherd": ["miniature","american","miniature american shepherd","mas","shepherd","迷你美国牧羊犬"],
  "Miniature Bull Terrier": ["miniature","bull","miniature bull","mbt","terrier","miniature bull terrier","迷你牛头梗"],
  "Miniature Pinscher": ["miniature pinscher","pinscher","miniature","mp","迷你杜宾犬"],
  "Mountain Cur": ["mc","mountain","mountain cur","cur","山地犬"],
  "Mudi": ["mudi","穆迪犬"],
  "Neapolitan Mastiff": ["nm","mastiff","neapolitan","neapolitan mastiff","那不勒斯獒犬"],
  "Nederlandse Kooikerhondje": ["nk","nederlandse kooikerhondje","kooikerhondje","nederlandse","荷兰诱鸭犬"],
  "Norfolk Terrier": ["norfolk terrier","terrier","norfolk","nt","诺福克梗"],
  "Norrbottenspets": ["norrbottenspets","北博滕尖嘴犬"],
  "Norwegian Buhund": ["buhund","norwegian buhund","nb","norwegian","挪威布哈德犬"],
  "Norwegian Lundehund": ["lundehund","nl","norwegian lundehund","norwegian","挪威猎鹦鹉犬"],
  "Norwich Terrier": ["norwich terrier","norwich","terrier","nt","诺威奇梗"],
  "Nova Scotia Duck Tolling Retriever": ["duck","nova scotia duck tolling","nsdtr","scotia","retriever","tolling","nova scotia duck tolling retriever","nova","新斯科舍诱鸭寻回犬"],
  "Otterhound": ["otterhound","水獭猎犬"],
  "Parson Russell Terrier": ["parson russell terrier","russell","terrier","parson","prt","parson russell","帕森罗素梗"],
  "Peruvian Inca Orchid": ["peruvian inca orchid","inca","pio","orchid","peruvian","秘鲁印加兰花犬"],
  "Petit Basset Griffon Vendéen": ["petit basset griffon vendéen","petit","griffon","pbgv","basset","vendéen","小旺代短腿猎犬"],
  "Pharaoh Hound": ["ph","pharaoh","pharaoh hound","hound","法老王猎犬"],
  "Plott Hound": ["plott","ph","plott hound","hound","普罗特猎犬"],
  "Pointer": ["pointer","指示犬"],
  "Polish Lowland Sheepdog": ["lowland","polish lowland sheepdog","pls","polish","sheepdog","波兰低地牧羊犬"],
  "Pont-Audemer Spaniel": ["pont-audemer","spaniel","pont-audemer spaniel","ps","蓬托德梅猎犬"],
  "Porcelaine": ["porcelaine","瓷器猎犬"],
  "Portuguese Podengo": ["portuguese podengo","pp","portuguese","podengo","葡萄牙波登可犬"],
  "Portuguese Podengo Pequeno": ["podengo","portuguese podengo pequeno","pequeno","portuguese","ppp","葡萄牙小型波登可犬"],
  "Portuguese Pointer": ["portuguese pointer","pp","portuguese","pointer","葡萄牙指示犬"],
  "Portuguese Sheepdog": ["portuguese sheepdog","portuguese","sheepdog","ps","葡萄牙牧羊犬"],
  "Portuguese Water Dog": ["water","dog","pwd","portuguese water","portuguese water dog","portuguese","葡萄牙水犬"],
  "Presa Canario": ["presa canario","pc","canario","presa","加那利犬"],
  "Pudelpointer": ["pudelpointer","贵宾指示犬"],
  "Puli": ["puli","普利犬"],
  "Pumi": ["pumi","普米犬"],
  "Pyrenean Mastiff": ["pm","pyrenean","mastiff","pyrenean mastiff","比利牛斯獒犬"],
  "Pyrenean Shepherd": ["pyrenean shepherd","ps","pyrenean","shepherd","比利牛斯牧羊犬"],
  "Rafeiro do Alentejo": ["rafeiro","do","rda","rafeiro do alentejo","alentejo","阿连特茹牧犬"],
  "Rat Terrier": ["rat","rat terrier","terrier","rt","捕鼠梗"],
  "Redbone Coonhound": ["coonhound","redbone","rc","redbone coonhound","红骨浣熊猎犬"],
  "Rhodesian Ridgeback": ["ridgeback","rr","rhodesian","rhodesian ridgeback","罗得西亚脊背犬"],
  "Romanian Carpathian Shepherd": ["romanian","carpathian","romanian carpathian shepherd","shepherd","rcs","罗马尼亚喀尔巴阡牧羊犬"],
  "Romanian Mioritic Shepherd Dog": ["dog","romanian mioritic shepherd dog","mioritic","romanian","romanian mioritic shepherd","rmsd","shepherd","罗马尼亚米奥里提克牧羊犬"],
  "Russell Terrier": ["russell terrier","russell","terrier","rt","罗素梗"],
  "Russian Toy": ["toy","rt","russian toy","russian","俄罗斯玩具犬"],
  "Russian Tsvetnaya Bolonka": ["bolonka","russian tsvetnaya bolonka","rtb","russian","tsvetnaya","俄罗斯彩色博龙卡犬"],
  "Saluki": ["saluki","萨路基犬"],
  "Schapendoes": ["schapendoes","荷兰牧羊犬（卷毛）"],
  "Schipperke": ["schipperke","史奇派克犬"],
  "Scottish Deerhound": ["sd","deerhound","scottish deerhound","scottish","苏格兰猎鹿犬"],
  "Scottish Terrier": ["st","scottish","scottish terrier","terrier","苏格兰梗"],
  "Sealyham Terrier": ["sealyham","terrier","st","sealyham terrier","锡利哈姆梗"],
  "Segugio Italiano": ["italiano","segugio","si","segugio italiano","意大利猎犬"],
  "Shetland Sheepdog": ["shetland sheepdog","ss","shetland","sheepdog","喜乐蒂牧羊犬"],
  "Shikoku Ken": ["shikoku","ken","shikoku ken","sk","四国犬"],
  "Silky Terrier": ["silky","terrier","st","silky terrier","丝毛梗"],
  "Skye Terrier": ["skye terrier","st","terrier","skye","斯凯梗"],
  "Sloughi": ["sloughi","斯卢吉犬"],
  "Slovakian Wirehaired Pointer": ["slovakian","swp","slovakian wirehaired pointer","wirehaired","pointer","斯洛伐克刚毛指示犬"],
  "Slovensky Cuvac": ["slovensky","sc","cuvac","slovensky cuvac","斯洛伐克库瓦奇犬"],
  "Slovensky Kopov": ["slovensky kopov","slovensky","kopov","sk","斯洛伐克猎犬"],
  "Small Munsterlander": ["munsterlander","sm","small","small munsterlander","小型明斯特兰德犬"],
  "Smooth Fox Terrier": ["fox","smooth fox","sft","terrier","smooth","smooth fox terrier","光滑狐梗"],
  "Soft Coated Wheaten Terrier": ["wheaten","scwt","soft","soft coated wheaten","soft coated wheaten terrier","terrier","coated","软毛麦色梗"],
  "Spanish Mastiff": ["spanish","sm","mastiff","spanish mastiff","西班牙獒犬"],
  "Spanish Water Dog": ["water","dog","spanish water dog","swd","spanish water","spanish","西班牙水犬"],
  "Spinone Italiano": ["si","spinone","italiano","spinone italiano","意大利旋毛猎犬"],
  "Stabyhoun": ["stabyhoun","斯塔比犬"],
  "Staffordshire Bull Terrier": ["bull","staffordshire bull","staffordshire","staffordshire bull terrier","sbt","terrier","斯塔福郡斗牛梗"],
  "Standard Schnauzer": ["ss","standard schnauzer","standard","schnauzer","标准雪纳瑞"],
  "Sussex Spaniel": ["spaniel","sussex spaniel","ss","sussex","苏塞克斯猎犬"],
  "Swedish Lapphund": ["lapphund","swedish","sl","swedish lapphund","瑞典拉普犬"],
  "Swedish Vallhund": ["swedish vallhund","sv","vallhund","swedish","瑞典瓦尔洪德犬"],
  "Taiwan Dog": ["taiwan dog","dog","td","taiwan","台湾犬"],
  "Teddy Roosevelt Terrier": ["trt","teddy roosevelt terrier","teddy roosevelt","teddy","roosevelt","terrier","泰迪罗斯福梗"],
  "Thai Bangkaew": ["thai","bangkaew","thai bangkaew","tb","泰国邦凯犬"],
  "Thai Ridgeback": ["thai ridgeback","ridgeback","thai","tr","泰国脊背犬"],
  "Tibetan Spaniel": ["tibetan","spaniel","tibetan spaniel","ts","西藏猎犬"],
  "Tibetan Terrier": ["tibetan","terrier","tibetan terrier","tt","西藏梗"],
  "Tornjak": ["tornjak","托尔尼亚克犬"],
  "Tosa": ["tosa","土佐犬"],
  "Toy Fox Terrier": ["tft","toy fox terrier","fox","toy fox","toy","terrier","玩具狐梗"],
  "Transylvanian Hound": ["transylvanian","th","transylvanian hound","hound","特兰西瓦尼亚猎犬"],
  "Treeing Tennessee Brindle": ["tennessee","brindle","ttb","treeing tennessee brindle","treeing","田纳西树猎犬"],
  "Treeing Walker Coonhound": ["coonhound","twc","walker","treeing","treeing walker coonhound","沃克浣熊猎犬"],
  "Vizsla": ["vizsla","维兹拉犬"],
  "Volpino Italiano": ["vi","volpino","italiano","volpino italiano","意大利狐狸犬"],
  "Welsh Springer Spaniel": ["spaniel","springer","welsh springer spaniel","welsh","wss","welsh springer","威尔士史宾格犬"],
  "Welsh Terrier": ["welsh","welsh terrier","wt","terrier","威尔士梗"],
  "West Highland White Terrier": ["west","whwt","highland","west highland white","white","west highland white terrier","terrier","西高地白梗"],
  "Wetterhoun": ["wetterhoun","荷兰水猎犬"],
  "Wirehaired Pointing Griffon": ["wpg","pointing","wirehaired pointing griffon","griffon","wirehaired","刚毛指示格里芬犬"],
  "Wirehaired Vizsla": ["wirehaired","wirehaired vizsla","vizsla","wv","刚毛维兹拉犬"],
  "Wire Fox Terrier": ["wft","wire fox","fox","wire fox terrier","terrier","wire","刚毛狐梗"],
  "Working Kelpie": ["working kelpie","kelpie","wk","working","工作型凯尔皮犬"],
  "Xoloitzcuintli": ["xoloitzcuintli","墨西哥无毛犬"],
  "Yakutian Laika": ["yakutian laika","yakutian","yl","laika","雅库特莱卡犬"]
}
    # 1️⃣ 如果文件存在 → 读取
    if os.path.exists(DOG_NAME_ALIAS_JSON_PATH):
        with open(DOG_NAME_ALIAS_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # 2️⃣ 合并（关键）
    for k, v in new_data.items():
        if k in data:
            # 去重追加
            data[k] = list(set(data[k] + v))
        else:
            data[k] = v

    # 3️⃣ 写回（不会丢原数据）
    with open(DOG_NAME_ALIAS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ 追加完成")
