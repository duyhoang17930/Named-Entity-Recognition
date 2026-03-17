from selenium import webdriver
import time

url = "https://github.com/duyhoang17930"

driver = webdriver.Chrome()   # mở Chrome 1 lần
driver.get(url)               # mở tab đầu tiên

while True:
    time.sleep(2)
    driver.refresh()          # reload lại CÙNG TAB
