
# 用户画像模拟生成
# pandas是用于数据操纵和分析的Python软件库。 它建造在NumPy基础上，并为操纵数值表格和时间序列，提供了数据结构和运算操作
# NumPy 是Python 语言的一个第三方库，其支持大量高维度数组与矩阵运算
import pandas as pd
import numpy as np

num_users = 500
# 定义属性及其取值范围
attributes = {
    '性别': ['男', '女', '未透露'],
    '所在城市': ['北京', '上海', '广州', '深圳', '其他'],
    '消费水平': ['高', '中', '低'],
    '年龄': np.random.randint(18, 71, size=num_users),  # 修改为500个用户
    '最近活跃天数': np.random.randint(1, 31, size=num_users)  # 修改为500个用户
}

# 生成用户数据
data = {
    '性别': np.random.choice(attributes['性别'], size=num_users),
    '所在城市': np.random.choice(attributes['所在城市'], size=num_users),
    '消费水平': np.random.choice(attributes['消费水平'], size=num_users),
    '年龄': np.random.randint(18, 71, size=num_users),  # 修改为500个用户
    '最近活跃天数': np.random.randint(1, 31, size=num_users)  # 修改为500个用户
}

# 创建DataFrame
df = pd.DataFrame(data)

# 存储为CSV文件
df.to_csv('/Users/zhanglianshuai1/Documents/Obsidian Vault/2025年/AI算法学习/practice/simulated_user_data.csv', index=False)

print("模拟用户数据已生成并存储为CSV文件。")