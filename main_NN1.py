from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#特徴量重要度抽出
from sklearn.inspection import permutation_importance
from utils import *
from structures import *
from algorithms import *
import matplotlib.pyplot as plt

df = CsvStructure('dataset_Outdoor.csv').remove_columns(0,2)
X1_start = 26 #400nm=10 450=20 500 =30 700=70
X1_end = 41 #1000nm=-29
X2_start = 72
X2_end = -29
y_place = -6
X1_feature_names = np.array(df.get_row(0)[X1_start : X1_end],dtype=object)
X2_feature_names = np.array(df.get_row(0)[X2_start : X2_end],dtype=object)
X_feature_names = np.append(X1_feature_names,  X2_feature_names)
y_feature_names = df.get_row(0)[y_place]
df = df.as_float()
df.remove_row(0)
y = df.get_column(y_place)
X1 = df.get_columns(X1_start, X1_end)
X2 = df.get_columns(X2_start, X2_end)
X = np.concatenate([X1, X2], axis=1)
# df = CsvStructure('dataset_Outdoor.csv').remove_columns(0,2)
# X_start = 10 #400nm
# X_end = -29 #1000nm
# y_place = -6 
# X_feature_names = df.get_row(0)[X_start : X_end]
# y_feature_names = df.get_row(0)[y_place]
# df = df.as_float()
# df.remove_row(0)
# y = df.get_column(y_place)
# X = df.get_columns(X_start, X_end)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

mlp = MLPRegressor(random_state=0,max_iter=10000) #(activation='tanh',solver='sgd',hidden_layer_sizes=(100,),random_state=0,max_iter=1000)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# MSE
print('MSE: %.2f' % (mean_squared_error(y_test, y_pred)))
#予測値と正解値を描写する関数
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
#RMSE = np.sqrt(mean_squared_error(y,y_pred))
print("RMSE:",RMSE)
#R2 
R2=r2_score(y_test, y_pred)
print("R2:",R2)

# plt.text(0.6, 2.5, '$ R^{2} $=' + str(round(R2, 3)))
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test,color ='black',linestyle='-', label='LinearRegression')
plt.xlabel('true-value(TSS)')
plt.ylabel('predict-value(TSS)')
# x軸の範囲を変更する
plt.xlim(6, 18)
# y軸の範囲を変更する
plt.ylim(6, 18)
plt.show()




importances=permutation_importance(mlp, X_train, y_train, n_repeats=30,random_state=0)['importances_mean']
# 特徴量重要性を降順にソート
indices = np.argsort(importances)[::-1]

# 特徴量の名前を、ソートした順に並び替え
names = [X_feature_names[i] for i in indices]

# プロットの作成
plt.figure(figsize=(20,4)) #プロットのサイズ指定
plt.title("Feature Importance")
plt.bar(range(np.array(X).shape[1]), importances[indices])
plt.xticks(range(np.array(X).shape[1]), names, rotation=90)

plt.show()


