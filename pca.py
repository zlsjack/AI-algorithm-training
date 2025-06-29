import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 假设user_embeddings是一个形状为(n_samples, n_features)的数组
user_embeddings = np.random.rand(100, 50)  # 示例数据

# 标准化数据
user_embeddings_standardized = (user_embeddings - np.mean(user_embeddings, axis=0)) / np.std(user_embeddings, axis=0)

# 应用PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(user_embeddings_standardized)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('User Embedding Visualization with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
