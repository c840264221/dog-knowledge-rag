from bs4 import BeautifulSoup
from src.crawler.pipeline import clean_text


#  解析列表页
def parse_list_page(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    links = []
    cards = soup.find_all("div", class_="breed-type-card")

    for card in cards:
        a_tag = card.find("a", href=True)
        if a_tag:
            url = a_tag["href"]
            if "/dog-breeds/" in url:
                links.append(url)

    return list(set(links))

# 解析出狗狗的标题（名字）
def parse_dog_name(soup):
    return {"name":soup.find("h1").text.strip()}

# 解析出狗狗的标签（特点）
def parse_dog_tag(soup):
    tag = soup.find("p", class_="breed-page__intro__temperment")

    if tag:
        return {"tag":tag.get_text(strip=True)}
    else:
        return ""

def parse_dog_basic_info(soup):
    result = {
        "height": [],
        "weight": [],
        "life_span": ""
    }
    blocks = soup.find_all("div", class_="flex flex-col")
    for block in blocks:
        title_tag = block.find("h3")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)

        values = [p.get_text(strip=True) for p in block.find_all("p")]

        # 👉 根据标题判断
        if "高度" in title or "Height" in title:
            result["height"] = values

        elif "重量" in title or "Weight" in title:
            result["weight"] = values

        elif "寿命" in title or "Life" in title:
            result["life_span"] = values[0] if values else ""

    return result

def parse_dog_all_traits(soup):
    # result = {}
    # CHOICE_TITLES = {"Coat Type", "Coat Length"}
    #
    #
    # total_blocks = soup.find_all("div", class_="breed-trait-group__trait breed-trait-group__padded breed-trait-group__row-wrap")
    # for block in total_blocks:
    #     block_tag = block.find("h4")
    #     description_tag = block.find_all("div",class_="breed-trait-score__score-unit breed-trait-score__score-unit--filled")
    #     description = len(description_tag)
    #     title = ""
    #     if block_tag:
    #         title = block_tag.get_text(strip=True)
    #     if title:
    #         if title in CHOICE_TITLES:
    #             selected = block.find_all(
    #                 "div",
    #                 class_=lambda x: x and "breed-trait-score__choice--selected" in x
    #             )
    #
    #             values = []
    #             for item in selected:
    #                 span = item.find("span")
    #                 if span:
    #                     values.append(span.get_text(strip=True))
    #
    #             traits[title] = values
    #             continue
    #     if title and description > 0:
    #         result[title] = description
    # return result
    traits = {}

    # 🎯 固定选择型字段
    CHOICE_TITLES = {"Coat Type", "Coat Length"}

    blocks = soup.find_all("div", class_="breed-trait-group__trait breed-trait-group__padded breed-trait-group__row-wrap")

    for block in blocks:
        title_tag = block.find("h4")
        # title_tag = block.find("div", class_="breed-trait-group__title")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)

        # =========================
        # 🆕 选择型（只针对两个字段）
        # =========================
        if title in CHOICE_TITLES:
            selected = block.find_all(
                "div",
                class_=lambda x: x and "breed-trait-score__choice--selected" in x
            )

            values = []
            for item in selected:
                span = item.find("span")
                if span:
                    values.append(span.get_text(strip=True))

            traits[title] = values
            continue

        # =========================
        # 普通评分型
        # =========================
        filled_units = block.find_all(
            "div",
            class_=lambda x: x and "breed-trait-score__score-unit--filled" in x
        )

        traits[title] = len(filled_units)
    return traits

def parse_dog_more_info(soup):
    result = {}

    info_blocks = soup.find_all("p", class_="breed-page__about__read-more__text")

    for info_block in info_blocks:
        info = info_block.get_text(strip=True)
        if len(info) > 0:
            current_info = result.get("About the Breed","")
            final_info = "".join([current_info,info])
            result["about_the_breed"] = clean_text(final_info)

    return result

def parse_dog_attention(soup):
    result = {}
    total_blocks = soup.find_all("div", class_="breed-table__wrap breed-page__health__table-wrap")
    for block in total_blocks:
        tittle_tag = block.find("h3")
        title = ""
        info = ""

        if tittle_tag:
            title = tittle_tag.get_text(strip=True)

        info_tag = block.find("p", class_="breed-table__accordion-padding__p")
        if info_tag:
            info = info_tag.get_text(strip=True)

        if title and info:
            result[title] = info
    return result

def parse_dog_history(soup):
    result = {}
    block = soup.find("div", class_="breed-page__history__text-content")
    if block:
        info = block.get_text(strip=True)
        result["history"] = clean_text(info)
    return result

def parse_dog_all_info(soup) -> list:
    return [
        parse_dog_name(soup),
        parse_dog_tag(soup),
        parse_dog_basic_info(soup),
        parse_dog_all_traits(soup),
        parse_dog_more_info(soup),
        parse_dog_attention(soup),
        parse_dog_history(soup),
    ]

def merge_dog_info(data_list:list[dict]) -> dict:
    final_result = {}
    for info in data_list:
        final_result.update(info)
    return final_result

#  解析详情页
def parse_detail_page(soup):
    all_data = parse_dog_all_info(soup)
    return merge_dog_info(all_data)

if __name__ == '__main__':
    from src.crawler.akc_spider import request_fetch, selenium_fetch, init_selenium_driver, click_button,button_xpath, click_read_more_history_button
    import time

    selenium_driver = init_selenium_driver()
    base_url = "https://www.akc.org/dog-breeds/page/2"
    result_text = request_fetch(base_url)
    links = parse_list_page(result_text)
    print(links)

    selenium_fetch(selenium_driver, links[0])

    click_button(selenium_driver, button_xpath['accept_cookies'])
    time.sleep(3)

    click_button(selenium_driver, button_xpath['read_more_about_breed'])
    time.sleep(1)

    click_read_more_history_button(selenium_driver)
    time.sleep(1)

    html = selenium_driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    # print(soup)

    # name = parse_dog_name(soup)
    # tage = parse_dog_tag(soup)
    # basic_info = parse_dog_basic_info(soup)
    all_traits = parse_dog_all_traits(soup)
    # more_info = parse_dog_more_info(soup)
    # attention = parse_dog_attention(soup)
    # history = parse_dog_history(soup)
    # print(name)
    # print(tage)
    # print(basic_info)
    print(all_traits)
    # print(more_info)
    # print(attention)
    # print(history)
    # final_info = parse_detail_page(soup)
    # print(final_info)

    input("按回车关闭浏览器...")

    selenium_driver.quit()
