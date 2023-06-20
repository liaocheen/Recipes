import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

# 随机有放回取样
def subsample(dataset_x, dataset_y, ratio=1.0):
    sample_x = list()
    sample_y = list()
    n_sample = round(len(dataset_x) * ratio)
    while len(sample_x) < n_sample:
        index = random.randrange(len(dataset_x))
        sample_x.append(dataset_x[index])
        sample_y.append(dataset_y[index])
    sample_x = np.array(sample_x)
    sample_y = np.array(sample_y)
    return sample_x, sample_y


# #取平均数，便于观察
# def mean(numbers):
#     return sum(numbers) / float(len(numbers))


train_path = 'recipes_train.csv'
test_path = 'recipes_test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

dataset_x = train_data.drop(columns=["cuisine"]).values
dataset_y = train_y = train_data["cuisine"].values

# print(f'特定列平均值为:{mean([row[28] for row in dataset_x])}')
# ratio = 0.10
# size = round(ratio*len(dataset_x))
# sample_mean = 0

# 利用随机取样数据生成新的数据集
# def creat_data(dataset_x, dataset_y, number=100, ratio=0.1):
#     all_sample_x = list()
#     all_sample_y = list()
#     for i in range(number):
#         sample_x, sample_y = subsample(dataset_x, dataset_y, ratio)
#         all_sample_x = all_sample_x + sample_x
#         all_sample_y = all_sample_y + sample_y
#     all_sample_x = np.array(all_sample_x)
#     all_sample_y = np.array(all_sample_y)
#     return all_sample_x, all_sample_y

# sample_mean = mean([row[28] for row in sample_x])
# print(f'取用{size}个数据，特定列平均值为{sample_mean}')

# print(all_sample_x.shape)
# print(all_sample_y.shape)
# all_sample_x1, all_sample_y1 = creat_data(dataset_x, dataset_y)
# all_sample_x2, all_sample_y2 = creat_data(dataset_x, dataset_y, number=50)2



# 定义一些变量
mean_train_accuracy_score = 0
mean_train_F1_score = 0
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)
# x_train = dataset_x
# y_train = dataset_y
# x_test = test_data.values
dataset_number = 100
ratio = 1.0
len_predict_y = len(x_test)
all_predict_y = np.empty(shape=(0, len_predict_y), dtype=object)
# len_train = round(len(y_train) * ratio)
# all_sample_x = np.empty(shape=(0, len_train), dtype=object)
# all_sample_y = np.empty(shape=(0, len_train), dtype=object)
for i in range(dataset_number):
    sample_x, sample_y = subsample(x_train, y_train, ratio)
    model = tree.DecisionTreeClassifier(random_state=42, max_depth=42)
    model.fit(sample_x, sample_y)
    predict_y = model.predict(x_test)
    all_predict_y = np.append(all_predict_y, values=np.array(predict_y).reshape(1, len(predict_y)), axis=0)
    # print(predict_y)
    train_accuracy_score = metrics.accuracy_score(model.predict(sample_x), sample_y)
    train_F1_score = metrics.f1_score(model.predict(sample_x), sample_y, average='macro')
    print(f'第{i + 1}次train accuracy：{train_accuracy_score}')
    print(f'第{i + 1}次train F1（采用macro）：{train_F1_score}')
    mean_train_accuracy_score = (mean_train_accuracy_score * i + train_accuracy_score) / (i + 1)
    mean_train_F1_score = (mean_train_F1_score * i + train_F1_score) / (i + 1)

# 选取各个模型中预测结果出现频率最高的作为最终预测结果
result_predict_y = np.empty(shape=len_predict_y, dtype=object)
for i in range(len(all_predict_y[0])):
    temp = all_predict_y[:, i]
    count = Counter(temp)
    max_count = max(count.values())
    result_predict_y[i] = list(count.keys())[list(count.values()).index(max_count)]

# print(result_predict_y)
# print(result_predict_y.shape)


print("mean train accuracy:", mean_train_accuracy_score)
print("train F1（采用macro）:", mean_train_F1_score)
print("test accuracy:", metrics.accuracy_score(result_predict_y, y_test))
print("test F1（采用macro）:", metrics.f1_score(y_test, result_predict_y, average='macro'))

# print(result_predict_y.shape)
# # 导出结果
# output_df = pd.DataFrame()
# output_df["id"] = test_data["id"]
# output_df["cuisine"] = result_predict_y
#
# # 导出文件名需要为submission.csv，格式参考evaluation中的说明
# output_df.to_csv("submission_bagging(DecisionTree).csv", index=False)

