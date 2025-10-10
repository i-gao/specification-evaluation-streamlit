import os
from playwright.sync_api import sync_playwright
import argparse
from PIL import Image
import json


def take_screenshot(url, output_file="screenshot.png", do_it_again=False):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    if os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)

            # Take the screenshot
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

            browser.close()
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file)


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description='Process two path strings.')

    # Define the arguments
    parser.add_argument('--html', type=str)
    parser.add_argument('--png', type=str)
    # Parse the arguments
    args = parser.parse_args()

    if args.html.endswith('.json'):
        from clean_predicted_html import clean_html

        js = json.load(open(args.html))
        predicted_html = js['predicted_html']
        predicted_html = clean_html(predicted_html)
        predicted_html_path = f"predicted_{os.getpid()}_{js['test_id']}.html"
        with open(predicted_html_path, "w") as f:
            f.write(predicted_html)
    else:
        predicted_html_path = args.html

    take_screenshot(predicted_html_path, args.png, do_it_again=True)

    if args.html.endswith('.json'):
        os.remove(predicted_html_path)
