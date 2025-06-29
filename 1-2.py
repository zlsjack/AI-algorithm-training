# 练习2: 简单留出法体验

import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
save_path = '/Users/zhanglianshuai1/Documents/Obsidian Vault/2025年/AI算法学习/practice/simulated_user_data.csv'  # 请将此路径替换为你的CSV文件路径
df = pd.read_csv(save_path)

# 统计原始数据中消费水平的分布比例
original_distribution = df['消费水平'].value_counts(normalize=True) * 100
print("原始数据中消费水平的分布比例：")
print(original_distribution)

# 使用sklearn进行80/20训练集/测试集划分
# 这行代码从DataFrame df中删除名为“消费水平”的列，因为“消费水平”是我们的目标变量，而我们希望将其他列作为特征（X）。axis=1表示我们按列操作。
X = df.drop('消费水平', axis=1)

# 这行代码将“消费水平”列提取出来，作为我们的目标变量（y）
y = df['消费水平']

"""
train_test_split：这是sklearn库中的一个函数，用于将数据集随机划分为训练集和测试集。
X 和 y：分别是特征和目标变量。
test_size=0.2：表示测试集占总数据集的比例为20%，因此训练集将占80%。
random_state=42：这是一个随机种子，用于确保每次运行代码时划分的结果相同，从而使结果可重复。
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算训练集中消费水平的分布比例
train_distribution = y_train.value_counts(normalize=True) * 100
print("\n训练集中消费水平的分布比例：")
print(train_distribution)

# 计算测试集中消费水平的分布比例
test_distribution = y_test.value_counts(normalize=True) * 100
print("\n测试集中消费水平的分布比例：")
print(test_distribution)

# 观察思考：对比训练集、测试集与“黄金标准”的比例
print("\n观察思考：")
print("训练集与黄金标准的比例偏差：")
print(train_distribution - original_distribution)
print("\n测试集与黄金标准的比例偏差：")
print(test_distribution - original_distribution)
