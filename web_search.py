# web_search.py
import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from readability import Document  # pip install readability-lxml
import time
from typing import List, Optional

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def search_serpapi(query: str, api_key: str, num_results: int = 5) -> List[str]:
    """Return URLs from Google Search via SerpAPI."""
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
    }
    response = requests.get("https://serpapi.com/search", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    urls = []
    for result in data.get("organic_results", []):
        if "link" in result:
            urls.append(result["link"])
    return urls

def is_scraping_allowed(url: str) -> bool:
    """Check robots.txt for the given URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = RobotFileParser()
    rp.set_url(f"{base}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except:
        # If robots.txt is unreachable, assume allowed (but we still handle errors)
        return True

def scrape_page(url: str, timeout: int = 10) -> Optional[str]:
    """Extract main article text using readability-lxml."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        doc = Document(resp.text)
        # readability returns cleaned HTML, we want plain text
        # Use its summary() which gives HTML, then parse to text
        html = doc.summary()
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator="\n", strip=True)
        # Remove excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def fetch_web_content(query: str, serp_api_key: str, max_pages: int = 3) -> str:
    """Search, check legality, scrape, and return combined text."""
    urls = search_serpapi(query, serp_api_key, num_results=max_pages + 2)  # extra in case some fail
    if not urls:
        return ""
    texts = []
    for url in urls:
        if not is_scraping_allowed(url):
            print(f"Skipping {url} – disallowed by robots.txt")
            continue
        text = scrape_page(url)
        if text and len(text) > 200:  # ignore very short pages
            texts.append(text)
            if len(texts) >= max_pages:
                break
        time.sleep(1)  # be polite
    return "\n\n".join(texts)