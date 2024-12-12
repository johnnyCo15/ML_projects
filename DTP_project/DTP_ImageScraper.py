# Image scraper for DTP (Designer trend predictor)
# run "source venv/bin/activate" before running

import requests
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

def getData(url):
    r = requests.get(url)
    return r.text

def scrape_images(url):
    htmldata = getData(url) 
    soup = BeautifulSoup(htmldata, 'html.parser')
    for item in soup.find_all('img'):
        img_url = item.get('src', '')
        if 'PICT' in img_url: 
            print(img_url)

def read_urls_from_txt(file_path):
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file.readlines() if line.strip()]  # Clean up the list
    return urls

def main():
    print("starting...")
    file_path = "/Users/jonnguyen/Downloads/GITHUB/ML_projects/Yohji_urls.txt"
    
    # Read URLs from the file
    urls = read_urls_from_txt(file_path)
    
    # Loop through each URL and scrape images
    for url in urls:
        print(f"Scraping {url}...")
        scrape_images(url)

    print("finished.")

if __name__ == "__main__":
    main()
