import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#特徴量重要度抽出
from sklearn.inspection import permutation_importance
import csv

data = data = np.loadtxt(r'C:\Users\bibbi\OneDrive - 国立大学法人 北海道大学\github\analysis\dataset_Indoor.csv', delimiter=',',skiprows=1,usecols=[76,78,87,106,116,-6], encoding="utf-8")
X = []
y = []
for row in data:
    X.append(row[0:4])
    y.append(row[5])

X_feature_names = ["720", "730","775", "870", "920"]
y_feature_names = ["Brix."]

with open('SVR_nonlinear.csv', 'w',newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(['seed_split', 'RMSE', 'R2'])
        for i in range(1,101):
                
                seed_split = i
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=seed_split)
                svr = SVR(kernel='rbf', C=1000, epsilon=0.25)
                svr.fit(X_train, y_train)
                y_train_pred = svr.predict(X_train)
                y_test_pred = svr.predict(X_test)

                #予測値と正解値を描写する関数
                RMSE = np.sqrt(mean_squared_error(y_test,y_test_pred))
                #R2 
                R2 =  r2_score(y_test, y_test_pred)
                writer.writerow((i,RMSE,R2))

# # 残差プロット
# plt.figure(figsize=(8,4)) #プロットのサイズ指定

# plt.scatter(y_train_pred,  y_train_pred - y_train,
#             c='red', marker='o', edgecolor='white',
#             label='Training data')
# plt.scatter(y_test_pred,  y_test_pred - y_test,
#             c='blue', marker='s', edgecolor='white',
#             label='Test data')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.legend(loc='upper left')
# plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
# plt.xlim([-10, 50])
# plt.tight_layout()

# plt.show()

# importances=permutation_importance(svr, X_train, y_train, n_repeats=30,random_state=0)['importances_mean']
# # 特徴量重要性を降順にソート
# indices = np.argsort(importances)[::-1]

# # 特徴量の名前を、ソートした順に並び替え
# names = [X_feature_names[i] for i in indices]

# # プロットの作成
# plt.figure(figsize=(20,4)) #プロットのサイズ指定
# plt.title("Feature Importance")
# plt.bar(range(np.array(X).shape[1]), importances[indices])
# plt.xticks(range(np.array(X).shape[1]), names, rotation=90)

# plt.show()