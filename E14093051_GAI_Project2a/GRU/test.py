import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import os
import opencc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
from transformers import AutoTokenizer  #还有其他与模型相关的tokenizer，如BertTokenizer


max_ = -1e9
min_ = 1e9

# 加载测试数据集
def load_test_data(file_path):
    df = pd.read_csv(file_path)
    return df['src'], df['tgt']

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNN, self).__init__()
        # Embedding層
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=0)

        # RNN層
        self.rnn_layer1 = torch.nn.GRU(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)

        output, _ = self.rnn_layer1(embedded)
        output = self.fc(output[:, -1,:])
        return output.squeeze(1)
    
# 定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, equations, results, tokenizer):
        self.equations = equations
        self.results = results
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, idx):
        equation_encoded = self.tokenizer.encode_plus(
            self.equations[idx],
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )
        result = torch.tensor(self.results[idx], dtype=torch.float32)
        return equation_encoded['input_ids'].squeeze(0), result,self.equations[idx]

# 定义测试函数
def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for equations, targets,q in test_loader:
            outputs = model(equations.to(device))
            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item()
            a = outputs.tolist()
            ans = targets.tolist()
            print(f"{q}{(round(a[0]))}")
            
    average_loss = total_loss / len(test_loader)
        
    print(f"Test Loss: {average_loss} Accuracy:{1-average_loss}")

# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 加载模型
model = RNN(vocab_size=tokenizer.vocab_size, embed_dim=256, hidden_dim=256)
model.load_state_dict(torch.load(model_path))
model.to(device)

# 加载测试数据
test_equations, test_results = load_test_data('MyTest.csv')
test_dataset = TestDataset(test_equations, test_results, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义损失函数
criterion = torch.nn.MSELoss()

# 测试模型
test_model(model, test_loader, device)
