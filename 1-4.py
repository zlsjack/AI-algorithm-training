# 练习4:将你的用户数据“向量化”
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
file = "/Users/zhanglianshuai1/Documents/Obsidian Vault/2025年/AI算法学习/practice/simulated_user_data.csv"
"""
# 使用bge-m3模型实现
from FlagEmbedding import BGEM3FlagModel

# 初始化BGE-m3模型
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 假设df是包含类别特征的DataFrame
sentences = df['性别'].astype(str).tolist() + df['所在城市'].astype(str).tolist() + df['消费水平'].astype(str).tolist()

# 使用BGE-m3生成特征向量
embeddings = model.encode(sentences, batch_size=12, max_length=8192)['dense_vecs']

# 将特征向量拼接成一个高维向量
df['features'] = embeddings.tolist()
"""


# 读取CSV文件
save_path = file 
df = pd.read_csv(save_path)

# 检查数据列是否为空
if df['性别'].empty or df['所在城市'].empty or df['消费水平'].empty:
    raise ValueError("One or more columns are empty.")

# 特征向量化
# 假设性别、城市和消费水平是类别特征
vectorizer_gender = CountVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
vectorizer_city = CountVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
vectorizer_consumption = CountVectorizer(stop_words='english', token_pattern=r'\b\w+\b')

gender_vectors = vectorizer_gender.fit_transform(df['性别'].astype(str))
city_vectors = vectorizer_city.fit_transform(df['所在城市'].astype(str))
consumption_vectors = vectorizer_consumption.fit_transform(df['消费水平'].astype(str))

# 将向量化的特征转换为密集矩阵
gender_dense = gender_vectors.toarray()
city_dense = city_vectors.toarray()
consumption_dense = consumption_vectors.toarray()

# 向量拼接
features_dense = np.hstack((gender_dense, city_dense, consumption_dense))

# 降维与可视化
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features_dense)

# 创建二维散点图
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
plt.title('2D PCA of User Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


