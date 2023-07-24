import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# Challenge: Turning Away Rude Customers
# Build a GPT-4 python app that talks with a user.
# End the conversation if they're being rude

# test case 1 'you're the worst human i've talked to' -> RUDE
# test case 2 'hey how's your day going'
# test case 3 'I like pizza. What do you like?'
# test case 4 'I bite my thumb at you!'
# test case 5 'I think this product doesnt work!' -> RUDE
