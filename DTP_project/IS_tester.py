
# Image scraper for DTP (Designer trend predictor)
# run "source venv/bin/activate" in ML_projects folder

import requests
import csv 
import re
from bs4 import BeautifulSoup

def getdata(url):
    r = requests.get(url)
    return r.text

def scrape(url):
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    for item in soup.find_all('img'):
        print(item['src'])

def main ():
    url = ("https://www.yohjiyamamoto.co.jp/collection/homme/spring-summer-2024-yyh/")
    scrape(url)

if __name__ == "__main__":
    main()