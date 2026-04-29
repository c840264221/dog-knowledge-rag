from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# print("🚀 准备启动浏览器...")
#
# driver = webdriver.Chrome(
#     service=Service(ChromeDriverManager().install())
# )
#
# print("🌐 浏览器已启动")
#
# driver.get("https://www.akc.org/dog-breeds/akita/")
#
# print("✅ 页面已打开")
#
# from crawler.akc_spider import accept_cookies
#
# print("处理cookies中......")
# accept_cookies(driver)
#
# import time
# time.sleep(3)
# html = driver.page_source
# soup = BeautifulSoup(html, "html.parser")
# title = soup.find("h1").text.strip()
# print(f"标题是：{title}")
# tag = soup.find("p", class_="breed-page__intro__temperment")
#
# if tag:
#     print(tag.get_text(strip=True))
#
# input("按回车关闭浏览器...")
# driver.quit()
url = 'https://www.akc.org/dog-breeds/affenpinscher/'
split_list = url.split('/')[-2]
print(split_list)