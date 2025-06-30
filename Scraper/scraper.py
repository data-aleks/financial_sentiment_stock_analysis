from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException

from warnings import warn
from time import sleep
from random import randint
import pandas as pd
from datetime import datetime, timedelta
import re # For regular expressions, helpful for parsing dates
from newspaper import Article # Import the Article class from newspaper3k
from tqdm import tqdm # Import tqdm for progress bars

## Scraper not supported in python 3.13 and required max of python 3.12 to run this is due to one of the dependencies

# --- Define the list of keywords to search for ---
SEARCH_KEYWORDS = ["Pfizer", "pfizer news", "pfizer stock news", "pfizer stock", "pfizer vaccine news"] # Add more keywords as needed
# BASE URL FOR SCRAPING - updated to include the query
# The 'q' parameter is the search query, 'start' is for pagination
BASE_URL_TEMPLATE = "https://www.google.com/search?q={}&tbm=nws&sxsrf=AE3TifNAKR9kb9Blb4hleJYLDOlvaTFSng:1750717900071&start={}"
RESULTS_PER_PAGE = 10 # Google typically shows 10 results per page for news

# Variable to control the total number of pages to scrape for THIS KEYWORD.
MAX_PAGES_TO_SCRAPE = 30 # Set your desired number of pages per keyword here
# Variable to control the total number of articles to scrape across all pages for THIS KEYWORD.
MAX_ARTICLES_TO_SCRAPE = 300 # Set your desired max articles per keyword here


# --- WebDriver Setup ---
# Configure Chrome options for the WebDriver
chrome_options = Options()
# Run in headless mode for both main scraping and full-text fallback to reduce overhead
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev_shm_usage")
chrome_options.add_argument("--log-level=3")  # Suppress detailed browser logs for cleaner output
# Add a user-agent to mimic a real browser more closely, reducing chances of being blocked
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36")

try:
    # Initialize the Chrome WebDriver once for all scraping operations
    driver = webdriver.Chrome(options=chrome_options)
except WebDriverException as e:
    # Handle WebDriver initialization errors, guiding the user on common issues
    print(f"Error initializing WebDriver: {e}")
    print("Please ensure you have Chrome installed and ChromeDriver is in your system's PATH or its path is specified.")
    print("You can download ChromeDriver from: https://chromedriver.chromium.org/downloads")
    exit()  # Exit the script if the WebDriver cannot be initialized

# --- Progress Tracking Setup ---
start_time = datetime.now()
print(f"Scraping started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

# Define CSS selectors for extracting specific data points from each article snippet.
# These selectors are updated for Google News search results.
ARTICLE_SELECTORS = {
    # Selectors are now relative to the 'a.WlydOe' element (the new 'snippet')
    "title": 'div.SoAPf .n0jPhd',          # Selector for the article title
    "description": 'div.SoAPf .GI74Re',    # Selector for the article description
    "date": 'div.SoAPf .OSrXXb',            # Selector for the article date
    # The 'link' is extracted directly from the parent 'a' tag (the snippet itself)
    # so it's not listed here as a child selector
}

# --- Helper Function to Parse Date Strings ---
def parse_article_date(date_str):
    """
    Parses a date string from the search result into a datetime.date object.
    Handles relative dates (e.g., 'X days ago', 'X hours ago') and absolute dates.
    Returns None if parsing fails.
    """
    date_str = date_str.lower().strip()
    current_date = datetime.now().date()

    # Handle relative dates
    if "ago" in date_str:
        num_match = re.search(r'(\d+)\s+(minute|hour|day|week|month|year)s?', date_str)
        if num_match:
            value = int(num_match.group(1))
            unit = num_match.group(2)
            if unit == "minute":
                return (datetime.now() - timedelta(minutes=value)).date()
            elif unit == "hour":
                return (datetime.now() - timedelta(hours=value)).date()
            elif unit == "day":
                return (datetime.now() - timedelta(days=value)).date()
            elif unit == "week":
                return (datetime.now() - timedelta(weeks=value)).date()
            elif unit == "month":
                # Approximating months/years as 30/365 days for simplicity in date comparison
                return (datetime.now() - timedelta(days=value * 30)).date()
            elif unit == "year":
                return (datetime.now() - timedelta(days=value * 365)).date()
    elif date_str == "today":
        return current_date
    elif date_str == "yesterday":
        return current_date - timedelta(days=1)

    # Handle absolute dates (e.g., "June 22, 2024" or "Jun 22, 2024")
    try:
        # Try parsing various common formats
        formats = [
            "%b %d, %Y",  # Jun 22, 2024
            "%B %d, %Y",  # June 22, 2024
            "%a, %d %b %Y", # Mon, 22 Jun 2024
            "%d %b %Y", # 22 Jun 2024
            "%Y-%m-%d" # If by any chance it's in ISO format
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
    except Exception as e:
        warn(f"Could not parse date string '{date_str}': {e}")
    return None # Return None if parsing fails

# --- Function to scrape full article text using newspaper3k (primary) and Selenium (fallback) ---
def scrape_full_article_text_robust(article_url, driver_options, main_driver):
    """
    Attempts to scrape the full text of an article using newspaper3k.
    If newspaper3k fails or returns empty content, it falls back to using Selenium.
    Returns the full text as a string, or "N/A" if scraping fails.
    This version reuses the main_driver if possible or creates a temporary one for fallback.
    """
    if not article_url or article_url == "N/A":
        return "N/A"

    # --- Attempt with newspaper3k first ---
    try:
        article = Article(article_url)
        article.download()
        article.parse()
        if article.text and len(article.text.strip()) > 50: # Check if meaningful text is extracted
            return article.text.strip()
        else:
            # print(f"   Newspaper3k returned short/empty text for {article_url[:70]}..., attempting Selenium fallback.")
            pass # Suppress this print to keep progress bar clean
    except Exception as e:
        warn(f"   Newspaper3k failed for {article_url[:70]}...: {e}, attempting Selenium fallback.")

    # --- Fallback to Selenium if newspaper3k failed or returned too little content ---
    # print(f"   Attempting Selenium fallback for: {article_url[:70]}...") # Suppress this print too
    temp_driver = None
    try:
        # For full article scraping, it's safer to use a new driver instance
        # to avoid interference with the main search results driver's state.
        temp_driver = webdriver.Chrome(options=driver_options)
        temp_driver.get(article_url)
        
        # Wait for the page to load, looking for common content elements
        WebDriverWait(temp_driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body')) # Ensure body is loaded
        )
        sleep(randint(2, 4)) # Give more time for dynamic content to load

        # --- Generic content extraction with Selenium ---
        # Prioritize common article body containers
        content_elements = temp_driver.find_elements(By.CSS_SELECTOR, 'article, main, div.entry-content, div.post-content, div#content, div.article-body, div.article__content')
        
        full_text_parts = []
        if content_elements:
            for element in content_elements:
                full_text_parts.append(element.text.strip())
        else:
            # Fallback to all paragraph tags if no specific content container is found
            # print("     No specific article content container found, trying all paragraph tags.") # Suppress
            p_elements = temp_driver.find_elements(By.TAG_NAME, 'p')
            for p in p_elements:
                text = p.text.strip()
                if len(text) > 20: # Filter out very short paragraphs (e.g., captions, ads)
                    full_text_parts.append(text)

        extracted_text = "\n\n".join(full_text_parts).strip()
        if extracted_text and len(extracted_text) > 25:
            return extracted_text
        else:
            return ""

    except TimeoutException:
        warn(f"   Selenium fallback: Timeout loading {article_url[:70]}...")
        return "Failed to scrape (Timeout)"
    except WebDriverException as e:
        warn(f"   Selenium fallback: WebDriver error for {article_url[:70]}...: {e}")
        return "Failed to scrape (WebDriver Error)"
    except Exception as e:
        warn(f"   Selenium fallback: Unexpected error for {article_url[:70]}...: {e}")
        return "Failed to scrape (Unexpected Error)"
    finally:
        if temp_driver:
            temp_driver.quit() # Always close the temporary driver

# --- Main scraping function for a single keyword ---
def scrape_google_news_for_keyword(keyword, driver, chrome_options):
    """
    Scrapes Google News for a given keyword, collects article snippets,
    and then scrapes the full text for each article.
    """
    all_articles_data = [] # This list will store dictionaries for the current keyword
    scraped_articles_count = 0
    current_page_num = 0



    print(f"\n{'='*60}\n--- Starting scraping for keyword: '{keyword}' ---\n{'='*60}")

    # Use tqdm for the page iteration loop
    with tqdm(total=MAX_PAGES_TO_SCRAPE, desc=f"Keyword '{keyword}' Progress", unit="page") as pbar_pages:
        # The loop continues as long as the desired number of pages has not been scraped
        while current_page_num < MAX_PAGES_TO_SCRAPE:
            current_page_num += 1
            
            # Calculate the 'start' parameter for the current page
            start_offset = (current_page_num - 1) * RESULTS_PER_PAGE
            current_url = BASE_URL_TEMPLATE.format(keyword, start_offset)

            # print(f"\n--- Scraping Page {current_page_num} (URL: {current_url}) ---") # Suppress manual page print

            pbar_pages.set_description(f"Keyword '{keyword}' (Page {current_page_num}/{MAX_PAGES_TO_SCRAPE})")

            try:
                driver.get(current_url)

                # --- Handle Google's Privacy Consent Pop-up (if it appears) ---
                try:
                    # print("   Checking for privacy consent pop-up...") # Suppress
                    consent_button_selector = (By.XPATH,
                        "//button[.//span[contains(text(), 'I agree') or contains(text(), 'Accept all') or contains(text(), 'Accept the use of cookies') or contains(text(), 'Alle akzeptieren') or contains(text(), 'Akzeptieren')]] | "
                        "//div[@role='button' and (@aria-label='I agree' or @aria-label='Accept all')]"
                    )
                    
                    consent_button = WebDriverWait(driver, 5).until( # Reduced wait for consent
                        EC.element_to_be_clickable(consent_button_selector)
                    )
                    driver.execute_script("arguments[0].click();", consent_button)
                    # print("   Privacy consent pop-up handled.") # Suppress
                    sleep(randint(2, 4)) # Give time for the popup to dismiss and page to settle
                except TimeoutException:
                    # print("   No privacy consent pop-up found or it did not appear within 5 seconds.") # Suppress
                    pass
                except ElementClickInterceptedException:
                    warn("   Consent button click intercepted. This might be a temporary overlay. Trying to proceed.")
                    sleep(2) # Brief pause and continue
                except Exception as e:
                    warn(f"   An unexpected error occurred while handling consent pop-up: {e}. Trying to proceed.")

                # Wait until at least one article snippet (the 'a.WlydOe' element) is present on the page, indicating it has loaded.
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'a.WlydOe'))
                )
                # print("Page loaded successfully.") # Suppress
                sleep(randint(3, 5)) # Pause for a random duration to mimic human browsing behavior
            except TimeoutException:
                print(f'\nTimeout: No article snippets found on {current_url} after 20 seconds. This might indicate the last page or an error.')
                break # Exit the loop if no snippets are found on a new page
            except WebDriverException as e:
                print(f'\nWebDriver Error loading page {current_url}: {e}. Skipping this page.')
                continue # Continue to the next page if there's a WebDriver error

            # Get all article container elements (now 'a.WlydOe') from the current page
            try:
                article_snippets = driver.find_elements(By.CSS_SELECTOR, 'a.WlydOe')
                # print(f"Found {len(article_snippets)} article snippets on page {current_page_num} for '{keyword}'.") # Suppress
            except NoSuchElementException:
                print(f"\nNo article snippets found on page {current_page_num} for '{keyword}'. End of data or error.")
                break
            except StaleElementReferenceException:
                # Handle stale element exceptions by retrying the current page
                warn("Stale element reference when finding snippets. Retrying current page.")
                sleep(5)
                continue # Skip to the next iteration of the while loop to re-locate elements

            if not article_snippets:
                print(f"\nNo records found on page {current_page_num} for '{keyword}'. Likely reached the end of available articles.")
                break

            # --- Scrape data from each article snippet on the current page ---
            for i, snippet in enumerate(article_snippets):
                # Stop scraping if the maximum number of articles has been reached (secondary limit)
                if scraped_articles_count >= MAX_ARTICLES_TO_SCRAPE:
                    # print(f"Reached maximum articles to scrape for '{keyword}' ({MAX_ARTICLES_TO_SCRAPE}). Stopping.") # Suppress
                    break # Break from the inner for loop

                article = {} # Dictionary to store data for the current article
                article["keyword"] = keyword # Add the keyword to the article data

                # Extract the link directly from the current snippet (which is the 'a.WlydOe' element)
                article["link"] = snippet.get_attribute('href').strip() if snippet.get_attribute('href') else "N/A"

                for col_name, css_selector in ARTICLE_SELECTORS.items():
                    try:
                        # For other columns, find the element *within* the current snippet
                        element = snippet.find_element(By.CSS_SELECTOR, css_selector)
                        article[col_name] = element.text.strip()
                    except NoSuchElementException:
                        # If an element is not found for a specific column, mark it as "N/A"
                        article[col_name] = "N/A"
                        warn(f"Column '{col_name}' not found for article {i+1} on page {current_page_num} for '{keyword}'. Setting to 'N/A'.")
                    except Exception as e:
                        # Catch any other unexpected errors during data extraction
                        article[col_name] = "Error"
                        warn(f"Error scraping '{col_name}' for article {i+1} on page {current_page_num} for '{keyword}': {e}. Setting to 'Error'.")

                all_articles_data.append(article)
                scraped_articles_count += 1

                # Print a sample of scraped data periodically for progress tracking
                # if (scraped_articles_count % 10 == 0) or (scraped_articles_count == 1):
                # print(f"   Scraped Article {scraped_articles_count}: Title: {article.get('title', 'N/A')} | Date: {article.get('date', 'N/A')}") # Suppress

            pbar_pages.update(1) # Update page progress bar

            # After processing all articles on the current page, check if the target has been met
            if scraped_articles_count >= MAX_ARTICLES_TO_SCRAPE:
                # print(f"Target number of articles ({MAX_ARTICLES_TO_SCRAPE}) reached for '{keyword}'. Ending scraping.") # Suppress
                break # Exit the main scraping loop for this keyword

            # If the inner loop broke due to MAX_ARTICLES_TO_SCRAPE, the outer loop should also break.
            if scraped_articles_count >= MAX_ARTICLES_TO_SCRAPE:
                break
        
    print(f"\n--- Initial scraping complete for '{keyword}'! Total articles scraped: {len(all_articles_data)}.")

    # --- Scrape full text for collected articles for this keyword ---
    print(f"\n--- Scraping Full Article Text for '{keyword}' (using newspaper3k with Selenium fallback) ---")
    # print("NOTE: Selenium fallback will launch a new headless browser for each article it needs to parse, which can be slow.") # Suppress
    
    # Use tqdm for the full-text scraping loop
    with tqdm(total=len(all_articles_data), desc=f"Full Text for '{keyword}'", unit="article") as pbar_articles:
        for i, article_data in enumerate(all_articles_data):
            link = article_data.get('link')
            if link and link != "N/A":
                # Pass the chrome_options for the temporary Selenium driver in the fallback
                full_text = scrape_full_article_text_robust(link, chrome_options, driver)
                article_data['full_text'] = full_text
                # print(f"   Processed article {i+1}/{len(all_articles_data)}. Link: {link[:70]}... | Text Length: {len(full_text) if full_text else 'N/A'}") # Suppress
                sleep(randint(1, 3)) # Be polite to the websites between article full-text scrapes
            else:
                article_data['full_text'] = "No link available"
                # print(f"   Skipping full text for article {i+1} due to missing link.") # Suppress
            pbar_articles.update(1) # Update article progress bar
    
    return all_articles_data # Return data for the current keyword


# --- Initialize a list to hold all scraped data from all keywords ---
all_combined_articles_data = []

# --- Loop through each keyword and perform scraping ---
for keyword in SEARCH_KEYWORDS:
    keyword_data = scrape_google_news_for_keyword(keyword, driver, chrome_options)
    all_combined_articles_data.extend(keyword_data) # Add data for current keyword to the combined list

# --- ALWAYS REMEMBER TO CLOSE THE MAIN BROWSER AT THE END OF ALL OPERATIONS ---
driver.quit()
print("-" * 50)
print("Main WebDriver closed.")

# --- Create a single DataFrame and Save to CSV for all collected articles ---
if all_combined_articles_data:
    articles_df = pd.DataFrame(all_combined_articles_data)

    print("\n--- Scraped Data Preview (from combined DataFrame) ---")
    # Displaying first 10 rows, but showing truncated full_text for readability
    pd.set_option('display.max_colwidth', 200) # Temporarily increase column width for preview
    print(articles_df.head(10)[['keyword', 'title', 'date', 'link', 'full_text']].applymap(lambda x: x[:100] + '...' if isinstance(x, str) and len(x) > 100 else x))

    print("\n--- Combined DataFrame Info ---")
    articles_df.info()

    csv_filename = "scraped_articles_data_combined.csv"
    articles_df.to_csv(csv_filename, index=False)

    print(f"\nAll combined data successfully saved to {csv_filename}")
else:
    print("No data was scraped across all keywords.")

print("-" * 50)
print("All scraping and data saving operations complete.")
