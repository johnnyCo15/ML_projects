# Image scraper for DTP (Designer trend predictor)
# run "source venv/bin/activate" in ML_projects folder

import requests
import asyncio
import aiohttp
import csv 
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_page_content(url):
    """Fetch the webpage content with error handling and retries."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to fetch {url} after {max_retries} attempts: {str(e)}")
                return None

def extract_season_year_from_url(url):
    """Extract season and year information from the URL."""
    # Pattern to match both old and new URL formats
    pattern = r'/(spring-summer|autumn-winter)-(\d{4})(?:-\d{2})?'
    match = re.search(pattern, url.lower())
    
    if match:
        season = match.group(1)  # spring-summer or autumn-winter
        year = match.group(2)    # year
        return season, year
    return None, None

def is_fashion_look_image(img_url):
    """Check if the image is a fashion look image."""
    # Exclude common non-fashion image patterns
    exclude_patterns = [
        'wp-content/themes',  # Theme images
        'sns',               # Social media icons
        'icon',             # Icons
        'logo',             # Logos
        'banner',           # Banners
        'background',       # Background images
        'menu',            # Menu images
        'footer',          # Footer images
        'header'           # Header images
        'brand'          # brand images
    ]
    
    # Check if URL contains any exclude patterns
    if any(pattern in img_url.lower() for pattern in exclude_patterns):
        return False
    
    # Include patterns specific to fashion look images
    include_patterns = [
        'look',            # Look images
        'collection',      # Collection images
        'homme'           # Men's collection
    ]
    
    # Check if URL contains any include patterns
    if any(pattern in img_url.lower() for pattern in include_patterns):
        return True
    
    # If the URL contains a number (likely a look number)
    if re.search(r'\d+', img_url):
        return True
    
    return False

def scrape_images(url):
    """Scrape all images from the given URL."""
    print(f"\nProcessing URL: {url}")
    
    # Extract season and year from URL
    season, year = extract_season_year_from_url(url)
    if not season or not year:
        print(f"Could not extract season/year from URL: {url}")
        return []
    
    print(f"Extracted season: {season}, year: {year}")
    
    # Get page content
    content = get_page_content(url)
    if not content:
        return []
    
    # Parse HTML
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find all image elements
    images = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            # Convert relative URLs to absolute
            full_url = urljoin(url, src)
            # Only include fashion look images
            if is_fashion_look_image(full_url):
                images.append({
                    'url': full_url,
                    'season': season,
                    'year': year
                })
    
    print(f"Found {len(images)} fashion look images")
    return images

def write_to_csv(images, filename='yohji_images.csv'):
    """Write image data to CSV file."""
    if not images:
        print("No images to write to CSV")
        return
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'season', 'year'])
        writer.writeheader()
        writer.writerows(images)
    
    print(f"Successfully wrote {len(images)} images to {filename}")

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urls_file = os.path.join(script_dir, 'yohji_urls.txt')
    
    # Read URLs from file
    try:
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: yohji_urls.txt not found")
        return
    
    print(f"Found {len(urls)} URLs to process")
    
    # Process each URL
    all_images = []
    for url in urls:
        images = scrape_images(url)
        all_images.extend(images)
        time.sleep(1)  # Be nice to the server
    
    # Write all images to CSV
    write_to_csv(all_images)

if __name__ == "__main__":
    main()