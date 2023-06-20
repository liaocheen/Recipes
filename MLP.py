import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


class MLP(nn.Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # 隐藏层
        self.hidden1 = nn.Linear(n_inputs, 128)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(32, 16)
        nn.init.kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(16, 8)
        nn.init.kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(8, 5)
        nn.init.xavier_uniform_(self.hidden6.weight)
        # 输出层
        self.act6 = nn.Softmax(dim=1)

    # 前向输出
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        X = self.hidden5(X)
        X = self.act5(X)
        X = self.hidden6(X)
        X = self.act6(X)
        return X


# 模型训练
def train_model(train_dl, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # 迭代
    for epoch in range(1):
        #取每次迭代的平均损失
        mean_loss = 0
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.long()
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            mean_loss = (mean_loss * i + loss.data) / (i + 1)
            optimizer.step()
        #保存训练模型结果，避免重复工作
        # path = f'MLP_model/model{epoch+1}.pth'
        # torch.save(model, path)
        # print('epoch:', epoch + 1)
        # print('mean loss:', mean_loss)


def evaluate_model(test_dl, model):
    #记录输出结果
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        y_predict = model(inputs)
        y_predict = y_predict.detach().numpy()
        actual = targets.numpy()
        y_predict = np.argmax(y_predict, axis=1)
        actual = actual.reshape((len(actual), 1))
        y_predict = y_predict.reshape((len(y_predict), 1))
        predictions.append(y_predict)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    #计算精度
    accuracy = metrics.accuracy_score(actuals, predictions)
    F1_score = metrics.f1_score(actuals, predictions, average='macro')
    return accuracy, F1_score

#使用模型进行预测
def predict(test, model):
    test = torch.Tensor([test])
    y_predict = model(test)
    y_predict = y_predict.detach().numpy()
    return y_predict


# 数据准备
class CSVDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        #x为输入特征，y为分类结果
        self.X = df.drop(columns=['id', "cuisine"]).values
        self.y = df["cuisine"].values
        self.X = self.X.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.3):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

#将分类结果由数字转为文字
def number2text(test_predict):
    test_result = list()
    for result in test_predict:
        if result == 0:
            test_result.append('chinese')
        elif result == 1:
            test_result.append('indian')
        elif result == 2:
            test_result.append('japanese')
        elif result == 3:
            test_result.append('korean')
        else:
            test_result.append('thai')
    test_result = np.array(test_result)
    return test_result


train_path = 'recipes_train.csv'
train_data = CSVDataset(train_path)
# dataset_x = train_data.drop(columns=["cuisine"]).values
# dataset_y = train_y = train_data["cuisine"].values
# x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)
train, test = train_data.get_splits()
train_dl = DataLoader(train, batch_size=1, shuffle=True)
test_dl = DataLoader(test, batch_size=1024, shuffle=False)
features = train_dl.dataset.dataset.X.shape[1]
print(features)
# model = MLP(features)
model = torch.load('MLP_model/model75.pth')
print(model)
# train_model(train_dl, model)
train_accuracy, train_F1_score = evaluate_model(train_dl, model)
test_accuracy, test_F1_score = evaluate_model(test_dl, model)
print('train accuracy: ', train_accuracy)
print("train F1（采用macro）:", train_F1_score)
print('test accuracy: ', test_accuracy)
print("test F1（采用macro）:", test_F1_score)
test_path = 'recipes_test.csv'
test_data = pd.read_csv(test_path)
test_x = test_data.drop(columns=["id"])
test_predict = list()
for i in range(len(test_x)):
    recipe = test_x.loc[i].values
    y_predict = predict(recipe, model)
    test_predict.append(np.argmax(y_predict))
test_predict = number2text(test_predict)
# print(test_predict)
# # 导出结果
# output_df = pd.DataFrame()
# output_df["id"] = test_data["id"]
# output_df["cuisine"] = test_predict
#
# # 导出文件名需要为submission.csv，格式参考evaluation中的说明
# output_df.to_csv("submission_MLP.csv", index=False)
