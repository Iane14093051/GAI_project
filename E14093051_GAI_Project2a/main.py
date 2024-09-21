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
import matplotlib.pyplot as plt

#data_path = '/data'

#df = pd.read_csv(os.path.join(data_path + '/arithmetic.csv'))
# 看一下前幾筆資料是什麼樣子
df = pd.read_csv(os.path.join( 'MyFile.csv'))
df.head()
#for i in range(0,df.shape[0]):
  #df['tgt'][i] = str(df['tgt'][i])

#df = df.iloc[:100000]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 参数设置
if torch.cuda.is_available():
    print("gpu")
    device = torch.device("cuda" )
else:
    print("cpu")
    device = torch.device("cpu" )

input_size = tokenizer.vocab_size
hidden_size = 256
learning_rate =0.001
batch_size = 256
num_epochs = 20
embed_size = 256
best_loss = float('inf')  # 初始化为正无穷大
model_path = "best_model.pth"  # 保存最佳模型的文件路径
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
# 准备数据集和数据加载器


    
class ArithmeticDataset(Dataset):
    def __init__(self, equation,result):
        self.equation = equation
        self.result = result
    def __len__(self):
        return len(self.equation)

    def __getitem__(self, idx):
        equation_encoded = tokenizer.encode_plus(
            self.equation[idx],
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )
        result_encoded = torch.tensor(self.result[idx], dtype=torch.float32)
        return equation_encoded['input_ids'].squeeze(0), result_encoded

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNN, self).__init__()
        # Embedding層
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=0)

        # RNN層
        self.rnn_layer1 = torch.nn.RNN(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)

        output, _ = self.rnn_layer1(embedded)
        output = self.fc(output[:, -1,:])
        return output.squeeze(1)

# 定义训练函数
def train_model(model, criterion, optimizer, dataloader, data_loader_v, best_loss, model_path, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for equations, targets in dataloader:
            optimizer.zero_grad()

            inputs = equations.to(device)

            #inputs = torch.flatten(inputs, start_dim=1)  # 展平输入张量
            outputs = model(inputs)    

            loss = criterion(outputs, targets.squeeze().to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #train_accuracy = calculate_accuracy(outputs, targets)
        print(f"Epoch {epoch+1}/{num_epochs},Train Loss: {running_loss/len(dataloader)}")
        _, best_loss = validate_model(model, criterion, data_loader_v, best_loss, model_path)
        train_loss = running_loss / len(dataloader)
        train_losses.append(train_loss)
        #train_accuracies.append(train_accuracy)

def validate_model(model, criterion, dataloader, best_loss, model_path):
        model.eval()  # 设置模型为评估模式
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():  # 在验证过程中不需要计算梯度
            for equations, targets in dataloader:
                outputs = model(equations.to(device))
                loss = criterion(outputs, targets.squeeze().to(device))
                total_loss += loss.item()
                predicted = torch.round(outputs)  # 四舍五入到最接近的整数
                correct_predictions += (predicted == targets.to(device)).sum().item()
                total_predictions += targets.size(0)
        accuracy = correct_predictions / total_predictions
        average_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {average_loss}")
        test_losses.append(average_loss)
        #test_accuracies.append(accuracy)
    
    # 如果当前的验证损失比之前记录的最佳损失更低，保存模型
        if average_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss = average_loss
    
        return average_loss, best_loss




dataset = ArithmeticDataset(df['src'],df['tgt'])

# The latter 10% of data for each case is used for the validation set, while the remaining 90% is used for the training set.
case_lengths = [50000,50000]

case_indices = []
start_index = 0
for length in case_lengths:
    end_index = start_index + length
    case_indices.append(list(range(start_index, end_index)))
    start_index = end_index


train_indices = []
val_indices = []
for indices in case_indices:
    train_size = int(0.9 * len(indices))
    train_indices.extend(indices[:train_size])
    val_indices.extend(indices[train_size:])

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

data_loader_t = torch.utils.data.DataLoader(train_dataset ,
                                          batch_size=batch_size,
                                          shuffle=True)

data_loader_v = torch.utils.data.DataLoader(val_dataset ,
                                          batch_size=batch_size,
                                          shuffle=True)



# 初始化模型、损失函数和优化器
model = RNN(input_size,
                embed_size,
                hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_model(model, criterion, optimizer, data_loader_t, data_loader_v, best_loss, model_path, num_epochs)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig("training_curve.png")
plt.show()

