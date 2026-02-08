import json
import requests
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_one_random_wiki_url():
    headers = {
        'User-Agent': 'WikiDataCollector/1.0 (contact: your@email.com)'
    }
    
    # 1. Hit the random endpoint
    # allow_redirects=True is the default, so it follows the 302 automatically
    try:
        response = requests.get("https://en.wikipedia.org/wiki/Special:Random", headers=headers, timeout=15, verify=False)
        
        # 2. Check if we were successful
        if response.status_code == 200:
            # .url gives the final URL after the 302 redirect
            return response.url
        else:
            print(f"Error: Received status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching random URL: {e}")
        return None
    
def get_random_wiki_urls(count):
    urls = set()
    for i in range(count):
        url = get_one_random_wiki_url()
        if url:
            urls.add(url)
            print(f"Collected: {len(urls)}/{count}", end="\r")
    print("\nCollection complete.")
    return list(urls)

def initialize_random_set(random_json_path):
    random_urls = set()
    try:
        with open(random_json_path, 'r') as f:
            if f.read().strip() == "":
                print("Random URLs file is empty. Generating new set...")
                random_urls = get_random_wiki_urls(300)
            else:
                f.seek(0)
                random_urls = json.load(f)
                if not isinstance(random_urls, list) or len(random_urls) != 300:
                    print("Invalid random URLs found. Regenerating...")
                    random_urls = get_random_wiki_urls(300)
    except FileNotFoundError:
        print("No random URLs found. Generating new set...")
        random_urls = get_random_wiki_urls(300)

    with open(random_json_path, 'w') as f:
        json.dump(list(random_urls), f)

# Execution
if __name__ == "__main__":
    initialize_random_set("data/random_urls.json")
    print(f"Success! Created fixed and random UR sets.")