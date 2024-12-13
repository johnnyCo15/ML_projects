# Image scraper for DTP (Designer trend predictor)
# run "source venv/bin/activate" in ML_projects folder

import requests
from bs4 import BeautifulSoup

def getData(url):
    r = requests.get(url)
    return r.text

def scrape_images(url, seen_urls):
    htmldata = getData(url) 
    soup = BeautifulSoup(htmldata, 'html.parser')
    
    for item in soup.find_all('img'):
        img_url = item.get('src', '')
        
        if 'PICT' in img_url and img_url not in seen_urls: 
            seen_urls.add(img_url)
            with open('output.txt','a') as f:
                f.write(img_url + '\n')
            print(f"added: {img_url}")

def readURLs(file_path):
    with open(file_path, 'r') as file:
        urls = file.read().splitlines()
        return(urls)

def main():
    print("starting...")
    
    file_path = "yohji_urls.txt"
    urls = readURLs(file_path)
    seen_urls = set()

    for url in urls: 
        # print(f"Processing URL: {url}")
        scrape_images(url, seen_urls)

    print("finished.")

if __name__ == "__main__":
    main()