# Image scraper for DTP (Designer trend predictor)
# run "source venv/bin/activate" in ML_projects folder

# NOT WORKING PROPERLY

import requests
import csv 
import re
from bs4 import BeautifulSoup

def getData(url):
    r = requests.get(url)
    return r.text

# def scrape_images(url, seen_urls, writer):
def scrape_images(url, writer):
    htmldata = getData(url) 
    soup = BeautifulSoup(htmldata, 'html.parser')

    # extracting season/ year from url
    season, year = parse_season_year(url)
    
    # scrape image urls + add to csv
    for item in soup.find_all('img'):
        img_url = item.get('src', '')
        
        # if 'PICT' in img_url and img_url not in seen_urls: 
        #     seen_urls.add(img_url)
        if 'PICT' in img_url or 'LOOK': 

            # write image url, season, and year to csv
            writer.writerow({'URL': img_url, 'SEASON': season, 'YEAR': year})
            print(f"added: URL: {img_url} | SEASON: {season} | YEAR: {year}")

# extract szn/ yr using regex
def parse_season_year(url):
    match = re.search(r'\/(autumn-winter|spring-summer)-(\d{4})(?:-(\d{2}|\d{4}))?-yyh', url)

    if match: 
        season = match.group(1).capitalize()
        year_start = match.group(2) 
        year_end = match.group(3)
    
        if year_end:
            #  if there is an end yr
            year = f"{year_start}-{year_end}"
        else: 
            # if there is no end yr, just use start yr
            year = year_start

        return season, year
    else: 
        # return none if szn/ yr cannot be parsed
        return None, None

def readURLs(file_path):
    with open(file_path, 'r') as file:
        urls = file.read().splitlines()
        return(urls)
    
# create csv with "URL/ SEASON/ YEAR" headers
def csv_create():
    csv_filepath = 'working.csv'
    with open(csv_filepath, mode = 'w', newline = '') as file:
        fieldnames = ['URL', 'SEASON', 'YEAR']
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()

    print(f"CSV file {csv_filepath} created")


def main():
    print("starting...")
    
    file_path = "yohji_urls.txt"
    # read urls from file 
    urls = readURLs(file_path) 
    # seen_urls = set()

    # create csv file
    csv_create()

    # open csv file to write
    with open('working.csv', mode = 'a', newline = '') as file:
        fieldnames = ['URL', 'SEASON', 'YEAR']
        writer = csv.DictWriter(file, fieldnames = fieldnames)

        for url in urls: 
        # print(f"Processing URL: {url}")
            # scrape_images(url, seen_urls, writer)
            scrape_images(url, writer)

    print("finished.")

if __name__ == "__main__":
    main()