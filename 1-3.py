
# 练习3:分层抽样体验
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# 读取CSV文件
save_path = '/Users/zhanglianshuai1/Documents/Obsidian Vault/2025年/AI算法学习/practice/simulated_user_data.csv'  # 请将此路径替换为你的CSV文件路径
df = pd.read_csv(save_path)

# 统计原始数据中消费水平的分布比例
original_distribution = df['消费水平'].value_counts(normalize=True) * 100
print("原始数据中消费水平的分布比例：")
print(original_distribution)

# 使用StratifiedShuffleSplit进行分层抽样
"""
使用StratifiedShuffleSplit进行分层抽样：
StratifiedShuffleSplit：这是sklearn库中的一个类，用于进行分层抽样。
n_splits=1：表示只进行一次划分。
test_size=0.2：表示测试集占总数据集的比例为20%。
random_state=42：这是一个随机种子，用于确保每次运行代码时划分的结果相同。
for train_index, test_index in sss.split(X, y)：这行代码遍历划分结果，并将索引分配给训练集和测试集。
"""
X = df.drop('消费水平', axis=1)
y = df['消费水平']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
