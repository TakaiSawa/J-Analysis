from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#特徴量重要度抽出
from sklearn.inspection import permutation_importance

data = np.loadtxt('2022sugar.csv', delimiter=',',skiprows=1, dtype=int, encoding="utf-8")
X = []
y = []
for row in data:
    X.append(row[0:-1])
    y.append(row[-1])

X_feature_names = ["350", "355", "360", "365", "370", "375", "380", "385", "390", "395", "400", "405", "410", "415", "420", "425", "430", "435", "440", "445", "450", "455", "460", "465", "470", "475", "480", "485", "490", "495", "500", "505", "510", "515", "520", "525", "530", "535", "540", "545", "550", "555", "560", "565", "570", "575", "580", "585", "590", "595", "600", "605", "610", "615", "620", "625", "630", "635", "640", "645", "650", "655", "660", "665", "670", "675", "680", "685", "690", "695", "700", "705", "710", "715", "720", "725", "730", "735", "740", "745", "750", "755", "760", "765", "770", "775", "780", "785", "790", "795", "800", "805", "810", "815", "820", "825", "830", "835", "840", "845", "850", "855", "860", "865", "870", "875", "880", "885", "890", "895", "900", "905", "910", "915", "920", "925", "930", "935", "940", "945", "950", "955", "960", "965", "970", "975", "980", "985", "990", "995", "1000", "1005", "1010", "1015", "1020", "1025", "1030", "1035", "1040", "1045", "1050", "1055", "1060", "1065", "1070", "1075", "1080", "1085", "1090", "1095", "1100"]
y_feature_names = ["Brix."]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

mlp = MLPRegressor(random_state=0,max_iter=10000)
mlp.fit(X_train, y_train)
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)


# MSE
print('MSE train: %.2f, test: %.2f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
#予測値と正解値を描写する関数
RMSE = np.sqrt(mean_squared_error(y_test,y_test_pred))
#RMSE = np.sqrt(mean_squared_error(y,y_pred))
print("Root Mean Square Error:",RMSE)
#R2 
print('R2 train: %.2f, test: %.2f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# 残差プロット
plt.figure(figsize=(8,4)) #プロットのサイズ指定

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='red', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='blue', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

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


