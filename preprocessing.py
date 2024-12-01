import json
from transformers import AutoTokenizer

with open("dataset.json", "r") as file:
    data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def preprocess_data(data):
    examples = []
    for item in data:
        input_text = f"### Question {item['question']}\n"
        output_text = f"### Answer {item['answer']}\n"
        examples.append({"input": input_text, "output": output_text})
    return examples


preprocess_data = preprocess_data(data)


# Tokenize
def tokenize(examples):
    return tokenizer(examples["input"], text_target=examples["output"], truncation=True, max_length=512)


tokenized_data = [tokenize(item) for item in preprocess_data]
