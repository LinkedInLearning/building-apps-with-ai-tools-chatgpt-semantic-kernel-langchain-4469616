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


prompt_prefix = """You are an order counting assisstant. Summarize the list into product name and total quantity into a JSON document. Only output the JSON, do not give an explanation.\n"""
prompt_examples = """List: 
1 x apple
2 x oranges
3 x fish
1 x apple
3 x duck
Output:
{
    "apple" : 2,
    "orange": 2,
    "fish" : 2,
    "duck": 3
}

List:"""
prompt = f"{prompt_prefix}{prompt_examples}" + "{{$input}}" + "Output:\n"
summarize = kernel.create_semantic_function(prompt)

# Summarize the list
summary_result = summarize(order)
print("GPT-4 Count", summary_result)
print("Actual Count:")
print(json.dumps(item_count, indent=4))

# self_assessment_prefix = (
#     "Evaluate how well the task was completed. Below is the original task:"
# )
# eval_suffix = "\n Is following result correct?\n"
# eval_suffix_2 = """\nVerify the total quantities, make sure they add up."""
# # self assess
# self_assessment = kernel.create_semantic_function(
#     f"{self_assessment_prefix}{prompt_prefix}{order}{eval_suffix}"
#     + "{{$input}}"
#     + f"{eval_suffix_2}"
# )

# self_assessment_result = self_assessment(summary_result.result)
# print("Self Assesment result", self_assessment_result)
