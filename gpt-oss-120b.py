##################################################
# Open AI open source model
#   https://huggingface.co/openai/gpt-oss-120b
##################################################

pip install -U transformers

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openai/gpt-oss-120b")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-120b")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


