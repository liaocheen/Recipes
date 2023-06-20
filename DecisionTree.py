import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


#读取文件
train_path = 'recipes_train.csv'
test_path = 'recipes_test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# 拆分训练数据
train_x = train_data.drop(columns=["cuisine"]).values
train_y = train_data["cuisine"].values


# 分割训练集验证集
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)

# 选用模型
model = LogisticRegression(penalty="l2", C=1.0, solver="newton-cholesky")


# 训练模型
model.fit(x_train, y_train)

# print(model.predict(X_train))
# print(y_train)
print("train accuracy:", metrics.accuracy_score(y_train, model.predict(x_train)))
print("train F1（采用macro）:", metrics.f1_score(y_train, model.predict(x_train), average='macro'))
print("test accuracy:", metrics.accuracy_score(y_test, model.predict(x_test)))
print("test F1（采用macro）:", metrics.f1_score(y_test, model.predict(x_test), average='macro'))
# print("train recall:", metrics.recall_score(x_train, model.predict(x_train)))
# print("test recall:", metrics.recall_score(x_test, model.predict(x_test)))



# 模型预测
result = model.predict(test_data.values)

# print(type(result))
# # 导出结果
# output_df = pd.DataFrame()
# output_df["id"] = test_data["id"]
# output_df["cuisine"] = result
#
# # 导出文件名需要为submission.csv，格式参考evaluation中的说明
# output_df.to_csv("submission_DecisionTree.csv", index=False)
