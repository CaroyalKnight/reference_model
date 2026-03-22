import torch
import os
import onnx
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch import nn

# ===================== 1. 基础配置（CPU固定+路径配置） =====================
# 固定使用CPU
device = torch.device("cpu")
torch.cuda.is_available = lambda: False

# 模型路径配置（和你的训练代码保持一致）
model_save_dir = "./trained_xlm_ref_model"
weights_path = os.path.join(model_save_dir, "model_weights.pth")
onnx_save_path = os.path.join(model_save_dir, "xlm_ref_model.onnx")

# 模型核心参数（必须和训练代码一致）
model_name = "FacebookAI/xlm-roberta-base"
max_length = 128
num_labels = 2

# ===================== 2. 模型结构定义（和训练代码完全一致） =====================
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
        outputs = self.roberta_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# ===================== 3. 加载训练好的模型 =====================
# 加载基础XLM-RoBERTa模型
base_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

# 初始化分类模型并加载权重
model = XLMRefClassifier(base_model, num_labels=num_labels).to(device)

# 兼容不同PyTorch版本的权重加载方式
try:
    # 高版本PyTorch（2.0+）
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
except TypeError:
    # 低版本PyTorch（无weights_only参数）
    model.load_state_dict(torch.load(weights_path, map_location=device))

model.eval()  # 关键：必须切换到评估模式
print("✅ 模型权重加载完成")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_save_dir)
print("✅ 分词器加载完成")

# ===================== 4. 构造Dummy Input（匹配模型输入） =====================
# 构造输入示例：batch_size=1, sequence_length=max_length
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_length), device=device)
dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long, device=device)

# ===================== 5. 核心：导出ONNX（兼容所有PyTorch版本） =====================
# 移除所有高版本专属参数，只保留通用参数
export_kwargs = {
    "model": model,
    "args": (dummy_input_ids, dummy_attention_mask),  # 模型输入（input_ids, attention_mask）
    "f": onnx_save_path,
    "opset_version": 12,  # 12是兼容性最好的版本（适配绝大多数PyTorch版本）
    "do_constant_folding": True,
    "input_names": ["input_ids", "attention_mask"],  # 输入节点名称
    "output_names": ["logits"],  # 输出节点名称
    "dynamic_axes": {  # 动态batch_size配置
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    "verbose": False
}

# 兼容高版本PyTorch的training参数
if hasattr(torch.onnx, "TrainingMode"):
    export_kwargs["training"] = torch.onnx.TrainingMode.EVAL

# 执行导出
try:
    torch.onnx.export(**export_kwargs)
    print(f"✅ ONNX模型导出成功！保存路径：{onnx_save_path}")
except Exception as e:
    print(f"❌ 导出失败，尝试禁用动态维度重试：{str(e)}")
    # 降级方案：关闭动态维度（仅支持batch_size=1）
    export_kwargs.pop("dynamic_axes")
    torch.onnx.export(**export_kwargs)
    print(f"✅ 禁用动态维度后导出成功！保存路径：{onnx_save_path}")

# ===================== 6. 验证ONNX模型有效性 =====================
try:
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型格式验证通过")
except Exception as e:
    print(f"⚠️ ONNX模型验证警告（不影响基本使用）：{str(e)}")

# ===================== 7. ONNX推理测试（对比原模型） =====================
# 1. 原PyTorch模型推理（基准）
def predict_with_pytorch(text):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return pred == 1, round(conf, 4)

# 2. ONNX Runtime推理
ort_session = ort.InferenceSession(
    onnx_save_path,
    providers=["CPUExecutionProvider"]  # 仅用CPU
)

def predict_with_onnx(text):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    # 转换为numpy格式（ONNX Runtime要求）
    ort_inputs = {
        "input_ids": encoding["input_ids"].cpu().numpy(),
        "attention_mask": encoding["attention_mask"].cpu().numpy()
    }
    # 执行推理
    ort_outputs = ort_session.run(["logits"], ort_inputs)
    # 计算softmax得到置信度
    ort_probs = np.exp(ort_outputs[0]) / np.sum(np.exp(ort_outputs[0]), axis=1, keepdims=True)
    pred_label = np.argmax(ort_probs, axis=1)[0]
    confidence = round(ort_probs[0][pred_label], 4)
    return pred_label == 1, confidence

# 测试对比
test_text1 = "Wang, H. (2021). NLP for reference detection. IEEE Transactions, 15(3), 200-210."
test_text2 = "这是一段普通的论文正文内容，没有参考文献格式。"

print("\n===== 推理结果对比 =====")
print(f"【文本1】{test_text1[:50]}...")
print(f"PyTorch预测：{predict_with_pytorch(test_text1)}")
print(f"ONNX预测：{predict_with_onnx(test_text1)}")

print(f"\n【文本2】{test_text2[:30]}...")
print(f"PyTorch预测：{predict_with_pytorch(test_text2)}")
print(f"ONNX预测：{predict_with_onnx(test_text2)}")

# ===================== 8. 性能测试（可选） =====================
import time

def test_inference_speed(text, repeat=100):
    # PyTorch速度
    start = time.time()
    for _ in range(repeat):
        predict_with_pytorch(text)
    torch_time = time.time() - start

    # ONNX速度
    start = time.time()
    for _ in range(repeat):
        predict_with_onnx(text)
    onnx_time = time.time() - start

    print(f"\n===== 性能测试（{repeat}次推理） =====")
    print(f"PyTorch耗时：{torch_time:.4f}秒")
    print(f"ONNX耗时：{onnx_time:.4f}秒")
    print(f"ONNX提速：{torch_time/onnx_time:.2f}倍")

# 运行性能测试（可选）
test_inference_speed(test_text1, repeat=100)