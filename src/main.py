from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import semantic_kernel as sk
from pathlib import Path
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
with open(Path.cwd() / "src" / "03_06.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text


kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service(
    "dv", OpenAIChatCompletion("gpt-4", api_key, org_id))

base_prompt = "You are a librarian." +\
    "Provide a recommendation to a book based on the following information. {{$input}}." +\
"Explain your thinking step by step including a list of top books you selected and how you got to your final choice."

recommendation = kernel.create_semantic_function(base_prompt,max_tokens=512)

print(recommendation(transcript))
