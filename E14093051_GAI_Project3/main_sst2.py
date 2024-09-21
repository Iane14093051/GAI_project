# 1. 確認所需套件的版本
import torch
print("PyTorch 的版本為: {}".format(torch.__version__))

import transformers
print("Hugging Face Transformers 的版本為: {}".format(transformers.__version__))
import tensorflow.keras.backend as K

import datasets
print("Hugging Face Datasets 的版本為: {}".format(datasets.__version__))

import peft
print("PEFT 的版本為: {}".format(peft.__version__))
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, TaskType
from peft import get_peft_model
from datasets import load_dataset, load_metric
sentence1_key, sentence2_key = ("sentence", None)
tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples[sentence1_key], truncation=True)

dataset = load_dataset("glue", "sst2")
metric = load_metric('glue',"sst2")
encoded_dataset = dataset.map(preprocess_function, batched=True)

num_labels = 2 #if task.startswith("mnli") else 1 if task=="stsb" else 2
model =transformers.AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)


#LORA
#lora_config = LoraConfig(
#    task_type=TaskType.SEQ_CLS, r=2, lora_alpha=8, lora_dropout=0.01
#)
#model = get_peft_model(model, lora_config)
#model.print_trainable_parameters()

#BITFIT
for name, param in model.named_parameters():
    if "bias" not in name:
       param.requires_grad = False



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 19. 訓練模型

args = transformers.TrainingArguments(
    output_dir= "/save",          # 輸出的資料夾
    num_train_epochs= 5,              # 總共訓練的 epoch 數目
    learning_rate= 2e-5,              # 學習率
    per_device_train_batch_size= 16,  # 訓練模型時每個裝置的 batch size
    per_device_eval_batch_size= 16,   # 驗證模型時每個裝置的 batch size
    gradient_accumulation_steps= 4,   # 梯度累積的步數
    warmup_steps= 100,                # learning rate scheduler 的參數
    weight_decay= 0.01,               # 最佳化演算法 (optimizer) 中的權重衰退率
    evaluation_strategy= "epoch",     # 設定驗證的時機
    save_strategy= "epoch",           # 設定儲存的時機
    save_steps= 100,                  # 設定多少步驟儲存一次模型
    eval_steps= 100,                  # 設定多少步驟驗證一次模型
    #report_to= ,         # 是否將訓練結果儲存到 TensorBoard
    #save_total_limit= ,              # 最多儲存幾個模型
    #logging_dir= ,            # 存放 log 的資料夾
    #logging_steps= ,
    #seed= ,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer =  transformers.Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 指定使用 1 個 GPU 進行訓練
trainer.args._n_gpu=1

# 開始進行模型訓練
trainer.train()

trainer.evaluate()
