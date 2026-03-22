import pandas as pd

# 构造标注数据集
# ========== 第一步：分别定义参考文献和普通文本 ==========
# 参考文献样本（label=1）
ref_texts = [
    # 英文期刊参考文献
    "Smith, J. D., & Johnson, L. K. (2020). Cross-lingual language models for text classification. Journal of Artificial Intelligence, 35(2), 112-134.",
    "Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877-1901.",
    "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). Bert: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT (Vol. 1, pp. 4171-4186).",
    "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. ACL, 8440-8451.",
    "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.",
    "Zhang, H., & Zhou, Z. (2018). XLM: Cross-lingual language model pretraining. EMNLP, 579XL-589XL.",
    "Miller, A. T. (2017). Natural language processing basics. MIT Press, Cambridge, MA.",
    "Wilson, E. B., & Taylor, R. M. (2019). Evaluating transformer models for reference detection. ACL Workshop on Text Classification, 56-72.",
    "Thomas, C. J., & Harris, P. L. (2021). Multilingual reference extraction from academic texts. IEEE Transactions on Knowledge and Data Engineering, 34(5), 2310-2322.",
    "Lee, S., Park, J., & Kim, H. (2022). A comparative study of XLM-RoBERTa and mBERT for cross-lingual NLP tasks. Computational Linguistics, 48(1), 145-178.",
    # 中文期刊参考文献
    "张三, 李四. (2020). 跨语言预训练模型在文本分类中的应用研究[J]. 计算机学报, 43(8), 1567-1589.",
    "王五, 赵六, 钱七. (2019). 基于XLM的多语言参考文献识别方法[J]. 中文信息学报, 33(5), 78-89.",
    "周八, 吴九. (2021). 学术论文参考文献自动标注技术综述[J]. 情报学报, 40(3), 321-335.",
    "郑十, 孙十一. (2018). 深度学习在参考文献抽取中的应用[J]. 模式识别与人工智能, 31(10), 901-912.",
    "杨十二, 朱十三, 马十四. (2022). 面向多语言的参考文献分类模型研究[J]. 自动化学报, 48(2), 345-358.",
    # 中文会议参考文献
    "黄十五, 林十六. (2020). 基于BERT的参考文献文本分类方法[C]//第二十二届全国计算语言学学术会议论文集. 北京: 清华大学出版社, 2020: 456-467.",
    "欧阳十七, 薛十八. (2019). 跨语言参考文献识别的XLM模型优化[C]//国际自然语言处理大会论文集. 上海: 上海交通大学出版社, 2019: 789-801.",
    # 中文书籍参考文献
    "何十九, 吕二十. (2021). 学术论文写作与参考文献规范[M]. 北京: 科学出版社, 123-156.",
    "施二一, 张二二. (2018). 自然语言处理实战[M]. 北京: 机械工业出版社, 2018: 345-367.",
    # 中英文混合参考文献
    "Li, M., & 王二三. (2020). Multilingual reference detection with XLM-RoBERTa[J]. 中国科学:信息科学, 50(7), 1023-1045.",
    "Zhang, Q., 陈二四, & Wang, Y. (2021). A hybrid model for Chinese-English reference classification[C]//IEEE International Conference on Data Mining. 2021: 890-897.",
    # 更多英文参考文献（补充）
    "Adams, R. J., & Baker, S. L. (2019). Reference extraction from unstructured text. Information Processing & Management, 56(4), 1450-1468.",
    "Clark, K., Luong, M. T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. ICLR.",
    "Das, D., & Smith, N. A. (2019). Bilingual language model pretraining for cross-lingual text classification. EMNLP-IJCNLP, 4539-4549.",
    "Evans, O. N., & Foster, J. B. (2021). Evaluating reference classification models on low-resource languages. ACL, 6789-6801.",
    "Fischer, S., & Gomez, A. M. (2020). Transfer learning for reference detection in academic papers. Journal of Digital Libraries, 21(3), 201-218.",
    "García, M., & López, J. (2018). A rule-based approach to reference extraction. Journal of the Association for Information Science and Technology, 69(11), 1345-1358.",
    "Harris, T. M., & Jackson, R. J. (2022). Combining rule-based and neural methods for reference classification. Computational Linguistics, 48(3), 678-709.",
    "Ingram, P., & Jones, S. (2019). XLM vs. mBERT: A comparison for cross-lingual reference detection. NAACL, 5678-5689.",
    "Johnson, A., & King, L. (2020). Preprocessing techniques for reference text classification. ACL Workshop on Text Preprocessing, 123-135.",
    "Klein, D., & Lewis, M. (2021). Few-shot learning for reference detection with XLM-RoBERTa. EMNLP, 8901-8912.",
    "Lee, H., & Martin, S. (2018). Multilingual reference corpus construction. Language Resources and Evaluation, 52(4), 1023-1045.",
    "Martin, T., & Nguyen, V. (2022). Data augmentation for reference classification tasks. ACL, 7890-7901.",
    "Nguyen, T., & Olson, R. (2019). Evaluating the impact of text length on reference detection. Computational Linguistics, 45(2), 345-378.",
    "Olson, S., & Patel, N. (2020). A survey of reference extraction methods in academic texts. ACM Computing Surveys, 53(4), 1-38.",
    "Patel, M., & Quinn, J. (2021). Cross-lingual transfer learning for reference classification. IEEE Transactions on Neural Networks and Learning Systems, 32(11), 4890-4902.",
    "Quinn, R., & Roberts, T. (2018). Neural reference detection: A comparative study. Journal of Machine Learning Research, 19(1), 1234-1267.",
    "Roberts, S., & Smith, K. (2022). Lightweight XLM models for CPU-based reference classification. EMNLP, 9012-9023.",
    "Smith, L., & Taylor, M. (2019). Domain adaptation for reference detection in technical papers. ACL, 6789-6801.",
    "Taylor, R., & Upton, J. (2020). Unsupervised reference extraction from multilingual texts. NeurIPS, 12345-12356.",
    "Upton, S., & Vasquez, M. (2021). Hybrid XLM models for reference classification. ICLR Workshop on Multilingual NLP, 45-56.",
    "Vasquez, R., & Wang, L. (2022). Efficient reference detection on CPU with XLM-RoBERTa-small. Journal of Low-Resource Computing, 8(2), 78-90.",
    "Wang, M., & Xu, T. (2018). Rule-based vs. neural methods for reference extraction. COLING, 3456-3467.",
    "Xu, S., & Yang, J. (2019). Multilingual reference tagging with XLM. ACL, 7890-7901.",
    "Yang, L., & Zhang, Q. (2020). Data quality impact on reference classification models. EMNLP, 8901-8912.",
    "Zhang, R., & Zhao, S. (2021). Fine-tuning XLM-RoBERTa for reference detection. NAACL, 6789-6801.",
    "Zhao, T., & Zhou, M. (2022). CPU-optimized XLM models for reference classification. Journal of AI Research, 78, 123-156."
]

# 普通文本样本（label=0）
normal_texts = [
    # 论文正文
    "本文提出了一种基于XLM-RoBERTa的参考文献分类模型，该模型在CPU环境下也能高效运行。",
    "实验结果表明，轻量化的XLM模型在参考文献识别任务上的准确率达到了89.5%，优于传统的规则方法。",
    "参考文献识别是学术文本处理的重要环节，准确的识别结果能提升论文排版和引用分析的效率。",
    "本研究收集了来自不同领域的学术论文数据，构建了包含中英文的参考文献标注数据集。",
    "在模型训练过程中，我们采用了小批次梯度下降的方法，以适配CPU的内存限制。",
    "传统的参考文献识别方法主要依赖正则表达式匹配格式特征，但泛化能力较差。",
    "跨语言模型的出现为多语言参考文献识别提供了新的思路，无需为每种语言单独训练模型。",
    "本实验对比了XLM-RoBERTa-base和XLM-RoBERTa-small两个版本，发现small版本在CPU上速度提升3倍。",
    "数据预处理阶段，我们对文本进行了截断和补齐操作，将最大长度限制为128个token。",
    "模型的分类头采用了dropout层和线性层的组合，有效防止了过拟合现象的发生。",
    # 日常描述
    "今天我整理了实验数据，发现参考文献样本的标注准确率还有提升的空间。",
    "学术论文的结构通常包括摘要、引言、方法、实验、结论和参考文献等部分。",
    "在本地CPU环境下运行大型语言模型时，需要注意批次大小和文本长度的优化。",
    "我用Python的venv模块创建了虚拟环境，避免了不同项目之间的依赖冲突。",
    "Hugging Face的transformers库提供了丰富的预训练模型，方便开发者快速搭建NLP应用。",
    "文本分类任务的核心是学习文本的语义特征，从而将其划分到正确的类别中。",
    "为了提升模型的泛化能力，我们在训练集中加入了不同格式和语言的参考文献样本。",
    "CPU的计算速度虽然不如GPU，但对于小规模的模型训练和推理仍然足够使用。",
    "标注数据集时，需要严格区分参考文献和普通正文，确保标签的准确性。",
    "模型训练完成后，我们用测试集评估了准确率、精确率和召回率等指标。",
    # 实验说明
    "本次实验使用的硬件环境为Intel i7-12700H CPU，内存32GB，操作系统为Windows 11。",
    "训练轮次设置为3轮，学习率为2e-5，优化器选用AdamW，权重衰减系数为0.01。",
    "数据划分采用了8:2的比例，80%作为训练集，20%作为测试集，随机种子设置为42。",
    "在推理阶段，关闭梯度计算可以显著提升CPU上的模型运行速度，减少内存占用。",
    "本地加载模型时，需要确保模型文件的完整性，包括权重文件和配置文件。",
    "分词器的最大长度设置会影响模型的性能，过长会增加计算量，过短会丢失关键信息。",
    "二分类任务的损失函数选用交叉熵损失，能够有效衡量预测标签和真实标签的差异。",
    "模型的前向传播过程中，我们提取了CLS token的输出作为文本的整体语义表征。",
    "为了验证模型的有效性，我们对比了随机森林、逻辑回归等传统机器学习方法的结果。",
    "在Windows系统中，PowerShell的执行权限可能会影响虚拟环境的激活，需要提前设置。",
    # 补充普通文本
    "参考文献是论文的重要组成部分，但本研究的重点是识别方法而非参考文献内容本身。",
    "XLM模型的跨语言能力使其能够处理中英文混合的文本，这对于中文文献处理尤为重要。",
    "训练模型时，批次大小设置为2可以避免CPU内存不足的问题，同时保证训练效率。",
    "本研究的创新点在于将掩码语言模型改造为分类模型，保留了其编码器的语义理解能力。",
    "实验结果显示，增加训练数据量可以提升模型的准确率，但超过一定数量后提升效果趋于平缓。",
    "在本地保存模型可以避免重复下载，提升代码的运行效率，尤其适合离线环境使用。",
    "文本预处理时的截断操作需要基于参考文献的长度分布，确保大部分样本的信息不被丢失。",
    "模型评估时的准确率指标反映了整体分类效果，但对于不平衡数据集还需要关注精确率和召回率。",
    "CPU环境下的模型推理速度约为每条文本0.5秒，能够满足小规模批量处理的需求。",
    "本研究的代码已开源，包含数据预处理、模型训练、推理预测等完整流程，方便复现。",
    "跨语言预训练模型通过在多语言语料上训练，能够学习到语言无关的通用语义特征。",
    "参考文献的格式通常包含作者、年份、标题、期刊/会议名称、页码等关键信息。",
    "模型的分类头设计越简单，在CPU上的运行速度越快，但可能会损失一定的准确率。",
    "数据标注过程中，我们邀请了两位领域专家对标注结果进行审核，确保标签的一致性。",
    "虚拟环境中安装的依赖包版本需要和代码适配，避免因版本不一致导致的运行错误。",
    "本研究的局限性在于数据集的领域覆盖不够全面，未来可以扩展到更多学科的学术论文。",
    "通过调整dropout的概率，可以平衡模型的拟合能力和泛化能力，找到最优的参数设置。",
    "在推理函数中，使用torch.no_grad()可以关闭梯度计算，减少不必要的计算开销。",
    "普通文本和参考文献的核心区别在于是否包含标准化的引用格式和学术来源信息。"
]

# ========== 第二步：自动生成对齐的text和label列表 ==========
# 合并文本列表
texts = ref_texts + normal_texts
# 自动生成标签：参考文献标1，普通文本标0（长度和texts完全一致）
labels = [1]*len(ref_texts) + [0]*len(normal_texts)

# 构造字典（确保text和label长度一致）
data = {
    "text": texts,
    "label": labels
}

# ========== 第三步：生成DataFrame并保存 ==========
# 转换为DataFrame
df = pd.DataFrame(data)

# 保存为CSV文件，方便后续复用
df.to_csv("reference_classification_dataset.csv", index=False, encoding="utf-8")

# 验证数据长度
print("数据集生成成功！")
print(f"总样本数：{len(df)}")
print(f"参考文献样本数：{len(ref_texts)}")
print(f"普通文本样本数：{len(normal_texts)}")
print(f"text列表长度：{len(texts)}, label列表长度：{len(labels)}")