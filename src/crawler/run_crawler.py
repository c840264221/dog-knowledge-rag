import time
from crawler.akc_spider import (request_fetch, selenium_fetch, click_button,button_xpath,init_selenium_driver,
                                click_read_more_history_button, turn_to_soup)
from crawler.parser import parse_list_page, parse_detail_page
from crawler.pipeline import dict_to_markdown, save_markdown


BASE_URL = "https://www.akc.org/dog-breeds/"


def run():
    # all_links = []
    #
    # # 👉 第一页
    # html = request_fetch(BASE_URL)
    # if html:
    #     all_links.extend(parse_list_page(html))
    # if len(all_links) == 0:
    #     print("未获取到列表页的url....")
    # else:
    #     print(f"成功获取到第一页的详情页url集合......")
    #     print(f"详情页的url集合:{all_links}")
    #
    #     for link in all_links:
    #         selenium_fetch(link)
    #         click_button(selenium_driver, button_xpath['accept_cookies'])
    #         time.sleep(3)
    #
    #         click_button(selenium_driver, button_xpath['read_more_about_breed'])
    #         time.sleep(1)
    #
    #         click_read_more_history_button(selenium_driver)
    #         time.sleep(1)
    #
    #         html_content = selenium_driver.page_source
    #         soup = turn_to_soup(html_content)
    #
    #         final_info = parse_detail_page(soup)
    #         if final_info:
    #             dog_name = link.split('/')[-2]
    #             print(f"{dog_name}狗狗的信息已经采集完毕......")
    #             save_markdown(final_info)
    #             print(f"👉 {dog_name}狗狗的信息已经录入完毕......")
    #
    #     selenium_driver.quit()
    # 👉 分页
    for i in range(1, 26):
        selenium_driver = init_selenium_driver()
        list_links = []
        url = f"{BASE_URL}page/{i}/"
        print(f"📄 列表页: {url}")

        html = request_fetch(url)
        if html:
            list_links.extend(parse_list_page(html))
        if len(list_links) == 0:
            print("未获取到列表页的url....")
        else:
            print(f"成功获取到第{i}页的详情页url集合......")
            print(f"详情页的url集合:{list_links}")
        for link in list_links:
            print(f"开始获取{link}的数据......")
            selenium_fetch(selenium_driver, link)
            click_button(selenium_driver, button_xpath['accept_cookies'])
            time.sleep(3)

            click_button(selenium_driver, button_xpath['read_more_about_breed'])
            time.sleep(1)

            click_read_more_history_button(selenium_driver)
            time.sleep(1)

            html_content = selenium_driver.page_source
            soup = turn_to_soup(html_content)

            final_info = parse_detail_page(soup)
            dog_name = link.split('/')[-2]
            if final_info:
                print(f"{dog_name}狗狗的信息已经采集完毕......")
                save_markdown(final_info)
                print(f"👉 {dog_name}狗狗的信息已经录入完毕......")
            else:
                print(f"❌️{dog_name}狗狗的信息获取失败......")
        print(f'-----------第{i}页狗狗信息已经存储完毕-----------')
        selenium_driver.quit()


if __name__ == "__main__":
    run()