import os
import google.generativeai as genai

# Configure the Gemini API key
genai.configure(api_key="AIzaSyD96PJQN2yCDlHkdnDn4xG1Hb85JMRaJrQ")

# Create generation configuration for the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def response(sentence1):
    # Instruction for logical check
    sentence2 = " Check if this is logical. If yes, just return 1, if no just return 0. Don't give a detailed answer, just return if true or false."
    prompt_final = sentence1 + sentence2
    
    # Start a chat session
    chat_session = model.start_chat(history=[])
    
    # Send the message and get the response
    completion = chat_session.send_message(prompt_final)
    result = completion.text.strip()
    
    # Check the response for logical correctness
    if result == "1":
        return "The Scenario is logically correct"
    else:
        return "The Scenario is not logically correct"

# Example usage
print(response("Sample input sentence to check logical correctness."))
