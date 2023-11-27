# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2-xl')
# set_seed(42)
# print(generator("I know the answer for `When is the independence day of US`, it is", max_length=50, num_return_sequences=3))

from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs)

last_hidden_states = outputs.last_hidden_state

