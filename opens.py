from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi")

# Prepare input prompt
prompt = "def add_two_numbers(a, b):"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate code
outputs = model.generate(inputs["input_ids"], max_length=100, temperature=0.7)
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)
