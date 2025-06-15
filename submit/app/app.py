from fastapi import FastAPI
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load tokenizer dan model GPT-2 yang sudah di-fine-tune menggunakan TensorFlow
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("./data/results")  # Lokasi model hasil fine-tuning

# FastAPI instance
app = FastAPI()

@app.get("/")
async def generate_description(prompt: str):
    # Encode prompt dan generate output
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_description": description}
