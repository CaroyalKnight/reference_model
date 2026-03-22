import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------- 1. 配置参数 --------------------------
# 原始数据集路径（替换为你的实际路径）
dataset_path = "reference_classification_dataset.csv"
# 训练集/测试集保存路径（可选）
train_save_path = "train_dataset.csv"
test_save_path = "test_dataset.csv"
# 切分比例：测试集占20%，训练集占80%
test_size = 0.2
# 随机种子（固定值保证切分结果可复现）
random_state = 42

# -------------------------- 2. 加载原始数据集 --------------------------
# 加载CSV文件（确保编码正确）
df = pd.read_csv(dataset_path, encoding="utf-8")

# 验证数据集完整性
print(f"原始数据集总样本数：{len(df)}")
print(f"参考文献样本数：{len(df[df['label']==1])}")
print(f"普通文本样本数：{len(df[df['label']==0])}")

# -------------------------- 3. 切分训练集和测试集 --------------------------
# 提取文本和标签
X = df["text"].tolist()  # 特征：文本
y = df["label"].tolist()  # 标签：0/1

# 按8:2切分（stratify=y 保证训练集/测试集的标签分布和原始数据一致）
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y  # 关键：分层抽样，避免训练/测试集标签分布不均
)

# -------------------------- 4. 转换为DataFrame并保存（可选） --------------------------
# 构造训练集DataFrame
train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

# 构造测试集DataFrame
test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

# 保存为CSV（方便后续直接加载，无需重复切分）
train_df.to_csv(train_save_path, index=False, encoding="utf-8")
test_df.to_csv(test_save_path, index=False, encoding="utf-8")

# -------------------------- 5. 验证切分结果 --------------------------
print("\n===== 数据切分结果 =====")
print(f"训练集样本数：{len(train_df)} (占比 {1-test_size:.0%})")
print(f"训练集-参考文献样本数：{len(train_df[train_df['label']==1])}")
print(f"训练集-普通文本样本数：{len(train_df[train_df['label']==0])}")

print(f"\n测试集样本数：{len(test_df)} (占比 {test_size:.0%})")
print(f"测试集-参考文献样本数：{len(test_df[test_df['label']==1])}")
print(f"测试集-普通文本样本数：{len(test_df[test_df['label']==0])}")

# -------------------------- 6. 供训练代码直接使用的输出 --------------------------
# 如果你想直接在训练代码中使用，可直接获取以下变量：
# train_texts = X_train
# train_labels = y_train
# test_texts = X_test
# test_labels = y_test
print("\n===== 可直接用于训练的变量 =====")
print(f"train_texts 长度：{len(X_train)}, train_labels 长度：{len(y_train)}")
print(f"test_texts 长度：{len(X_test)}, test_labels 长度：{len(y_test)}")