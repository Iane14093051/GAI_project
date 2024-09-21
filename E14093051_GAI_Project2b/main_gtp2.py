from transformers import BertTokenizer,T5Tokenizer, T5ForConditionalGeneration, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq ,Text2TextGenerationPipeline
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import re
import os
from ignite.metrics import Rouge
import jieba
class CustomerDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['text']
        summary = item['summary']
        inputs = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        labels = self.tokenizer(summary, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # Ignoring loss for padding tokens
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(), "labels": labels.squeeze()}

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
model = T5ForConditionalGeneration.from_pretrained("uer/t5-small-chinese-cluecorpussmall").cuda() if torch.cuda.is_available() else T5ForConditionalGeneration.from_pretrained("t5-small")
rouge = Rouge(variants=[1,2,"L"], multiref="best")
# Check and print the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your dataset from JSON Lines file
data = []
data = []
with open('train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

train_data = data[:12800]

data2 = []
with open('valid.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data2.append(json.loads(line))

valid_data = data2[:1280]
# Create dataset and data loader
train_dataset = CustomerDataset(train_data, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataset = CustomerDataset(valid_data, tokenizer, max_len=128)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Initialize optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)


def remove_ignore_tokens_from_list(tensor_list, ignore_token=-100):
    filtered_tensor_list = [tensor[tensor != ignore_token] for tensor in tensor_list]
    return filtered_tensor_list

def evaluate(model):
    pbar = tqdm(valid_loader)
    pbar.set_description(f"Evaluating")
    rouge.reset()
    for data in pbar:
        inputs = data["input_ids"].to(device)
        targets = data["labels"].to(device)
        output = [tokenizer.batch_decode(model.generate(inputs, max_length=128), skip_special_tokens=True)]
        targets = remove_ignore_tokens_from_list(targets, ignore_token=-100)
        targets = [tokenizer.batch_decode(targets, skip_special_tokens=True)]
        for i in range(len(output)):
            sentences = [s.replace('。', ' 。').split() for s in output[i]]
            ground_truths = [t.replace('。', ' 。').split() for t in targets[i]]
            for s in sentences:
                rouge.update(([s], [ground_truths]))
    return rouge.compute()

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as train_pbar:
        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_pbar.set_postfix({'Loss': total_loss / (len(train_pbar))})

    # Save the entire model at the end of each epoch
    folder_name = "t5_model"   
    os.makedirs(folder_name, exist_ok=True)
    torch.save(model, f"{folder_name}/model_t5_epoch_{epoch}.pt")

    # Evaluate on validation set
    print(f"Rouge-2 score on epoch {epoch}:", evaluate(model))
