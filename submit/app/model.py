import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load tokenizer dan model GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Menambahkan padding token
tokenizer.pad_token = tokenizer.eos_token

# Memuat dataset dan memprosesnya
df = pd.read_csv("../data/Week3NikeProductDescriptionsGenerator.csv")
descriptions = df['Product Description'].tolist()

# Tokenisasi deskripsi produk
def preprocess(desc):
    encodings = tokenizer(desc, truncation=True, padding=True, max_length=512)
    return Dataset.from_dict(encodings)

train_dataset = preprocess(descriptions)

# Menyiapkan collator untuk data training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./data/results",  # Tempat menyimpan model yang sudah di-fine-tune
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

# Inisialisasi Trainer untuk fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Fine-tuning model
trainer.train()
