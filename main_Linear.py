from utils import *
from structures import *
from algorithms import *
import matplotlib.pyplot as plt
df = CsvStructure('dataset_Indoor.csv').remove_columns(0,2)
X_start = 10 #400nm
X_end = -9 #1000nm
y_place = -6
X_feature_names = df.get_row(0)[X_start : X_end]
y_feature_names = df.get_row(0)[y_place]
df = df.as_float()
df.remove_row(0)
y = df.get_column(y_place)
X = df.get_columns(X_start, X_end)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std[0]

model = LinearRegression()
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)


RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Square Error:", RMSE)
R2 = r2_score(y_test, y_pred) 
print("R2:", R2)
model.coef_
print("傾き",model.coef_)
model.intercept_
print("切片",model.intercept_)

with open('coefficient.csv','w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow([X_feature_names, model.coef_])

# plt.text(0.6, 2.5, '$ R^{2} $=' + str(round(R2, 3)))
# plt.scatter(y_test, y_pred)
# plt.plot(y_test, y_test,color ='black',linestyle='-', label='LinearRegression')
# plt.xlabel('true-value(TSS)')
# plt.ylabel('predict-value(TSS)')
# # x軸の範囲を変更する
# plt.xlim(6, 18)
# # y軸の範囲を変更する
# plt.ylim(6, 18)
# plt.show()

# importances=permutation_importance(rft, X_train, y_train, n_repeats=30,random_state=0)['importances_mean']
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