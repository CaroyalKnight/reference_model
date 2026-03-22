import torch
import torch.nn as nn  # 必须导入nn，因为模型类继承了nn.Module
import os
from transformers import AutoTokenizer

# -------------------------- 1. 关键修复：添加自定义类到安全白名单 --------------------------
# 第一步：重新定义和训练时完全一致的XLMRefClassifier类
class XLMRefClassifier(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.roberta_encoder = base_model.roberta
        self.hidden_size = self.roberta_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# 第二步：将自定义类加入PyTorch安全全局列表（核心修复）
torch.serialization.add_safe_globals([XLMRefClassifier])

# -------------------------- 2. 基础配置 --------------------------
device = torch.device("cpu")
torch.cuda.is_available = lambda: False

config = {
    "trained_model_dir": r"D:\model_train\reference\trained_xlm_ref_model",
    "max_length": 128
}

# -------------------------- 3. 加载完整模型（已适配PyTorch 2.6+） --------------------------
def load_full_trained_model(config):
    full_model_path = os.path.join(config["trained_model_dir"], "full_model.pth")
    # 无需修改weights_only，已通过安全白名单允许加载自定义类
    model = torch.load(full_model_path, map_location=device, weights_only=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["trained_model_dir"])
    print("✅ 完整模型加载成功！")
    return model, tokenizer

# -------------------------- 4. 预测函数（不变） --------------------------
def predict_reference(text, model, tokenizer, max_length=128):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
    
    return pred_label == 1, round(confidence, 4)

# -------------------------- 5. 执行预测 --------------------------
if __name__ == "__main__":
    model, tokenizer = load_full_trained_model(config)
    
    test_texts = [
        "张三, 李四. (2020). 跨语言预训练模型研究[J]. 计算机学报, 43(8), 1567-1589.",
        "本研究在CPU环境下训练XLM模型，实现了参考文献的高效识别。",
        "Wang X, de Araujo J F, Ju W, et al. Mechanistic reaction pathways of enhanced ethylene yields during electroreduction of CO2-CO co-feeds on Cu and Cutandem electrocatalysts [J]. Nature Nanotechnology, 2019, 14(11): 1063-70. ",
        "Li L, Wang H, Han J, et al. A density functional theory study on reduction induced structural transformation of copper-oxide-based oxygen carrier [J]. Journal of Chemical Physics, 2020, 152(5): 054709."
    ]
    import time
    print("===== 预测结果 =====")
    for text in test_texts:
        # 打印时间
        start_time = time.time()
        is_ref, conf = predict_reference(text, model, tokenizer, config["max_length"])
        print(f"文本：{text[:50]}...")
        print("花费时间毫秒：", (time.time() - start_time) * 1000)
        print(f"是否为参考文献：{is_ref}，置信度：{conf}\n")
        
    print("===== 预测结束 =====")