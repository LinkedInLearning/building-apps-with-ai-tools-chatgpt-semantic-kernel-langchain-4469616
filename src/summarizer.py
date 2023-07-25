from pathlib import Path
import json
from collections import defaultdict
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4", api_key, org_id))

with open(Path.cwd() / "src" / "order.txt") as f:
    order = f.read()
    f.seek(0)
    order_lines = f.readlines()
item_count = defaultdict(int)

# Loop over each line in the file
for line in order_lines:
    # Split the line into quantity and item
    quantity, item = line.strip().split(" x ")

    # Update the count in the dictionary
    item_count[item] += int(quantity)


prompt_prefix = """You are an order counting assisstant. Summarize the list into product name and total quantity, thinking step by step. Then output into a JSON object.\n"""
prompt_examples = """List: 
1 x apple
2 x orange
1 x apple
3 x fish
1 x apple
1 x apple
3 x duck
1 x apple

Output:
First we need to count the number of rows of each.
We have:
5 rows of apple
1 row of oranges
1 row of fish
1 row of duck

Next let's add all the order counts together.
apple = 1+1+1+1+1
orange = 2
fish = 3
duck = 3

Now lets output into JSON

{
    "apple" : 5,
    "orange": 2,
    "fish" : 2,
    "duck": 3
}

List:\n"""
prompt = f"{prompt_prefix}{prompt_examples}" + "{{$input}}" + "\nOutput:\n"
summarize = kernel.create_semantic_function(prompt, max_tokens=512, temperature=0.3)

# Summarize the list
summary_result = summarize(order)
print("GPT-4 Count", summary_result)
print("Actual Count:")
print(json.dumps(item_count, indent=4))