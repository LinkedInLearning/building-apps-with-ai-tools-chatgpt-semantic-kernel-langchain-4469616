import os

import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

while True:
    user_input = user_input = input("Enter a phrase and we’ll classify it as happy or sad\n")
    if user_input == "quit" or user_input == "exit":
        break
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment classification bot, print out if the user is happy or sad. Only print out happy or sad."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150,
    )

    response_message = response["choices"][0]["message"]
    print(response_message)