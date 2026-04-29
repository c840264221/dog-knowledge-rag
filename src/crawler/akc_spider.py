import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# 用requests的session来获取数据 绕过部分简单反爬  虽然没什么卵用
session = requests.Session()
session.headers.update(HEADERS)

# 用selenium来进行请求  可以绕过大部分反爬
# selenium_driver = webdriver.Chrome(
#     service=Service(ChromeDriverManager().install())
# )
button_xpath = {
    'accept_cookies': '//*[@id="onetrust-accept-btn-handler"]',
    'read_more_history': '/html/body/div[4]/div[2]/div/div[13]/div/div[1]/div[3]',
    'read_more_about_breed': '/html/body/div[4]/div[2]/div/div[6]/div/div/div/div[3]/span'
}

def init_selenium_driver():
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install())
    )
    return driver

def request_fetch(url):
    try:
        res = session.get(url, timeout=10)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(e)
        return None

def selenium_fetch(selenium_driver, url):
    selenium_driver.get(url)
    # return selenium_driver.page_source

def turn_to_soup(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup

# 有的狗狗的历时信息很长需要点击按钮“read more”来获取完整文档
def click_read_more_history_button(driver):
    try:
        # 点击 history 的 read more
        history_btn = driver.find_element(
            By.XPATH,
            "//h2[contains(text(),'History')]/following::*[contains(., 'Read More')]"
        )

        driver.execute_script("arguments[0].click();", history_btn)
        print("已点击history中的read more按钮")

    except Exception:
        print("⚠️ 没找到按钮")

def click_button(driver, xpath):
    try:
        wait = WebDriverWait(driver, 5)

        button = wait.until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )

        button.click()
        print("🍪 已点击按钮")
        # input("暂停一下，看看浏览器还在不在...")

    except Exception as e:
        # print(e)
        print("⚠️ 没找到 Cookie 按钮（可能已经处理过）")


if __name__ == '__main__':
    base_url = "https://www.akc.org/dog-breeds/akita/"
    selenium_fetch(base_url)
    # # print(result)
    # click_button(selenium_driver, button_xpath['accept_cookies'])
    # import time
    # time.sleep(3)
    # html = selenium_driver.page_source
    # print("拿到HTML长度:", len(html))
    #
    # input("按回车关闭浏览器...")
    #
    # selenium_driver.quit()
