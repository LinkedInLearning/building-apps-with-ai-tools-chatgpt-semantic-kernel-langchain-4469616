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


prompt_prefix = """You are an order counting assistant. Summarize the list into product name and total quantity into a JSON document. Only output the JSON, do not give an explanation.\n"""
prompt = f"{prompt_prefix}" + "{{$input}}" + "Output:\n"
summarize = kernel.create_semantic_function(prompt)

# Summarize the list
summary_result = summarize(order).result
print("GPT-4 Count", summary_result)

formatted_actual = json.dumps(item_count, indent=4)
print("Actual Count:")
print(formatted_actual)


def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    import difflib
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)

    diff = difflib.unified_diff(expected, actual)

    return ''.join(diff)

print(_unidiff_output(summary_result,formatted_actual))

# we'll use this data in some downstream process
assert summary_result == formatted_actual
