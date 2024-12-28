from chatgpt_selenium_automation.handler import ChatGPTAutomation


def response(sentence1):
    # Define the path where the chrome driver is installed on your computer
    chrome_driver_path = r"C:\\Users\\PRIYA\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"

    # the sintax r'"..."' is required because the space in "Program Files" 
    # in my chrome_path
    chrome_path = r'"C:\\Users\\PRIYA\\Downloads\\chrome-win64\\chrome-win64\\chrome.exe"'

    # Create an instance
    chatgpt = ChatGPTAutomation(chrome_path, chrome_driver_path)

    # Define a prompt and send it to chatGPT
    #sentence1 = "a lion that paints." 
    sentence2 = "Check if this is logical if yes, just return true, if no just return false. Don't give detailed anser just return if true or false."
    prompt = sentence1 + sentence2
    chatgpt.send_prompt_to_chatgpt(prompt)

    # Retrieve the last response from chatGPT
    response = chatgpt.return_last_response()
    print(response)

    # Save the conversation to a text file
    file_name = f"{sentence1}.txt"
    chatgpt.save_conversation(file_name)

    # Close the browser and terminate the WebDriver session
    chatgpt.quit()

    # Initialize a counter for prompts
    prompt_count = 0
    second_prompt_value = None

    file_path = f"E:\\FYP_review1\\project_root\\conversations\\{sentence1}.txt"
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate through the lines to find the second prompt
    for line in lines:
        line = line.strip()
        
        if line.startswith("prompt:"):
            prompt_count += 1
            # If it's the second prompt, extract its value
            if prompt_count == 2:
                second_prompt_value = line[len("prompt:"):].strip()  # Get the text after "prompt:"
                break  # Exit the loop since we found the second prompt

    # Output the extracted value
    if second_prompt_value is not None:
        return second_prompt_value  # This will print "False"
    else:
        return "Second prompt not found."
    
