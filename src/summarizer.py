import os
import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

#prompt = kernel.create_semantic_function("""Give me the TDLR of Harry Potter""")

#print(prompt())

summarize = kernel.create_semantic_function("""{{$input}} \n\n one line TLDR with the fewest words""")

print(summarize("Harry Potter and the Goblet of Fire"))

