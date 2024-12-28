# trial5.py
from webdriver_setup import WebDriverSetup

def main():
    chrome_path = r'"C:\\Users\\PRIYA\\Downloads\\chrome-win64\\chrome-win64\\chrome.exe"'  # Adjust as necessary
    web_driver = WebDriverSetup(chrome_path)  # Initialize WebDriver

    # Use web_driver.driver to interact with the browser
    # Example: Navigate to a URL
    web_driver.driver.get("https://www.gmail.com")

    # Your automation code here...

    # Clean up and close the browser
    web_driver.driver.quit()

if __name__ == "__main__":
    main()
