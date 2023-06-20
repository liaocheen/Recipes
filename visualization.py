import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取文件
train_path = 'recipes_train.csv'
train_data = pd.read_csv(train_path)

# 1.各个地区的原料使用情况
# 用于存储给个地区的原料使用情况
all_regions = []
distribution = {}

for i, region in enumerate(train_data['cuisine']):
    #找出所有地区
    if region not in all_regions:
        all_regions.append(region)
        distribution[region] = {}
    #记录每个地区的原料使用量
    for j, material in enumerate(train_data.keys()):
        if j > 1:
            if material not in distribution[region]:
                distribution[region][material] = train_data[material][i]
            else:
                distribution[region][material] = distribution[region][material] + train_data[material][i]


# print(distribution['indian'])
# print(distribution['chinese'])
# print(distribution['korean'])
# print(distribution['thai'])

plt.figure()
subplot_count = len(distribution.keys())
#分地区绘制原料使用情况
for i in range(subplot_count):
    all_regions = list(distribution.keys())[i]
    plt.subplot(subplot_count, 1, i+1)
    plt.bar(range(len(distribution[all_regions].keys())), distribution[all_regions].values())
    plt.title(all_regions)
# 避免标题和子图重叠
plt.tight_layout()
plt.show()

#2.使用最多原料在各个地区的使用情况
#记录各种原料的总使用量
amount = {}
for i, material in enumerate(train_data.keys()):
    if i > 1:
        amount[material] = train_data[material].sum()

#找到使用量最大的原料并观察在各个地区的使用情况
max_material = max(amount, key=amount.get)
max_material_region = {}
for i, isUsing in enumerate(train_data[max_material]):
    if train_data['cuisine'][i] not in max_material_region:
        max_material_region[train_data['cuisine'][i]] = 1
    else:
        max_material_region[train_data['cuisine'][i]] = max_material_region[train_data['cuisine'][i]] + 1

#绘制柱形图
plt.bar(max_material_region.keys(), max_material_region.values())
plt.title(f'the usage of {max_material} in different regions')
plt.show()

#重做表格并保存（减少每次运行时的计算），按照地区为行，原料总数量为列
out_data = pd.DataFrame(distribution)
out_data = out_data.T
out_data.to_csv('out_train_data.csv', index=True)

#3.相关性计算
correlation_path = 'out_train_data.csv'
correlation_data = pd.read_csv(correlation_path)
# 计算相关矩阵
# 剔除和数据全为0的列
train_data_amount = correlation_data
for key in amount.keys():
    if amount[key] == 0:
        train_data_amount = train_data_amount.drop(columns=[key])
#相关矩阵计算
correlation = train_data_amount.corr()
print(correlation)
# 绘制散点图（选择使用量最多的前几种绘制）
amount_sorted = sorted(amount, key=amount.get, reverse=True)
print(amount_sorted[0:6])
train_data_max6 = correlation_data
for i, material in enumerate(amount_sorted):
    if i > 5:
        train_data_max6 = train_data_max6.drop(columns=[material])
print(train_data_max6)
pd.plotting.scatter_matrix(train_data_max6, figsize=(12, 12), range_padding=0.5)
plt.show()


