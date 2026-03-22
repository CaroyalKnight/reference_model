import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------- 1. 强制使用CPU & 基础配置 --------------------------
# 固定使用CPU，禁用GPU
device = torch.device("cpu")
torch.cuda.is_available = lambda: False  # 彻底屏蔽GPU检测

# 加载你指定的XLM-RoBERTa-base模型和分词器
model_name = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载掩码语言模型（仅用其编码器部分）
base_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

# -------------------------- 2. 自定义分类模型（基于MaskedLM） --------------------------
class XLMRefClassifier(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        # 提取XLM-RoBERTa的核心编码器（丢弃掩码预测头）
        self.roberta_encoder = base_model.roberta
        # 获取模型隐藏层维度（xlm-roberta-base为768）
        self.hidden_size = self.roberta_encoder.config.hidden_size
        # 添加二分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(self.hidden_size, num_labels)  # 768 -> 2（是/否参考文献）
        )
    
    def forward(self, input_ids, attention_mask):
        # 前向传播：获取<s> token（CLS）的隐藏状态作为句子表征
        outputs = self.roberta_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 取[CLS] token的输出（batch_size, hidden_size）
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 分类预测
        logits = self.classifier(cls_output)
        return logits

# 初始化分类模型
model = XLMRefClassifier(base_model).to(device)

# -------------------------- 3. 数据集定义 --------------------------
class RefDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词处理
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 去除batch维度，返回tensor
        return {
            "input_ids": encoding["input_ids"].squeeze(0).to(device),
            "attention_mask": encoding["attention_mask"].squeeze(0).to(device),
            "label": torch.tensor(label, dtype=torch.long).to(device)
        }

# -------------------------- 4. 准备数据 --------------------------
# 示例标注数据（替换为你的真实数据）
# 读取csv
df = pd.read_csv("reference_classification_dataset.csv")

# 划分训练集/测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# 创建数据集和数据加载器（CPU适配：batch_size调小）
train_dataset = RefDataset(train_texts, train_labels, tokenizer)
test_dataset = RefDataset(test_texts, test_labels, tokenizer)

# DataLoader：num_workers=0避免Windows多进程报错，batch_size=2适配CPU内存
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

# -------------------------- 5. 训练配置（CPU优化） --------------------------
# 损失函数（二分类交叉熵）
criterion = nn.CrossEntropyLoss()
# 优化器（学习率调低，避免震荡）
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
# 训练轮次（少量数据无需过多轮次）
epochs = 3

# -------------------------- 6. 训练函数 --------------------------
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        # 计算损失
        loss = criterion(outputs, batch["label"])
        total_loss += loss.item()
        # 反向传播+更新参数
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Average Loss: {avg_loss:.4f}")

# -------------------------- 7. 评估函数 --------------------------
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    # 关闭梯度计算，提速+省内存
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # 获取预测标签
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# -------------------------- 8. 开始训练（纯CPU） --------------------------
if __name__ == "__main__":
    print("===== 开始训练（仅使用CPU） =====")
    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer, epoch)
        evaluate_model(model, test_loader)
    
    # ========== 核心：保存训练好的模型 ==========
    # 1. 定义模型保存路径（建议创建专门的文件夹）
    save_dir = "./trained_xlm_ref_model"  # 相对路径，保存在当前代码目录下的trained_xlm_ref_model文件夹
    # 确保保存目录存在（不存在则创建）
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 2. 保存模型的两种方式（推荐第二种，更易用）
    
    # 方式1：保存完整模型（包含结构+权重，体积大）
    torch.save(model, os.path.join(save_dir, "full_model.pth"))
    
    # 方式2：仅保存模型权重（推荐，体积小，加载灵活）
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
    
    # 3. 同时保存分词器（推理时需要）
    tokenizer.save_pretrained(save_dir)
    
    print(f"\n===== 模型保存完成 ======")
    print(f"模型文件保存路径：{os.path.abspath(save_dir)}")
    print(f"完整模型文件：full_model.pth")
    print(f"模型权重文件：model_weights.pth")
    print(f"分词器文件：tokenizer_config.json、vocab.json 等")
    
    # -------------------------- 9. 推理预测（核心功能） --------------------------
    def predict_is_reference(text):
        """
        判断单条文本是否为参考文献
        :param text: 待判断文本
        :return: bool（True=是参考文献）, 置信度
        """
        model.eval()
        with torch.no_grad():
            # 分词
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )
            # 推理
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device)
            )
            # 计算置信度
            probs = torch.softmax(outputs, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        return pred_label == 1, round(confidence, 4)
    
    # 测试预测
    test_text1 = "Wang, H. (2021). NLP for reference detection. IEEE Transactions, 15(3), 200-210."
    test_text2 = "这是一段普通的论文正文内容，没有参考文献格式。"
    
    print("\n===== 预测结果 =====")
    print(f"文本1：{test_text1}")
    print(f"是否为参考文献：{predict_is_reference(test_text1)}")
    print(f"\n文本2：{test_text2}")
    print(f"是否为参考文献：{predict_is_reference(test_text2)}")