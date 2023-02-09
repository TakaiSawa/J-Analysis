from utils import *
from structures import *
from algorithms import *
import matplotlib.pyplot as plt

df = CsvStructure('dataset_Outdoor.csv').remove_columns(0,2)
X1_start =10 #26 #400nm=10 450=20 500 =30 700=70
X1_end = -29 #41 #1000nm=-29
# X2_start = 72
# X2_end = -49
y_place = -6
X1_feature_names = np.array(df.get_row(0)[X1_start : X1_end],dtype=object)
# X2_feature_names = np.array(df.get_row(0)[X2_start : X2_end],dtype=object)
X_feature_names = X1_feature_names #np.append(X1_feature_names,  X2_feature_names)
y_feature_names = df.get_row(0)[y_place]
df = df.as_float()
df.remove_row(0)
y = np.round(np.array(df.get_column(y_place))*10)
X1 = np.round(np.array(df.get_columns(X1_start, X1_end))*100).astype(int)
# X2 = np.round(np.array(df.get_columns(X2_start, X2_end))*100).astype(int)
X = X1 #np.concatenate([X1, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
rft = RandomForestRegressor(n_estimators=100, random_state=0,  max_depth=5)
rft.fit(X_train, y_train)
y_pred = rft.predict(X_test)/10
y_test = y_test/10
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)#Root Mean Square Error
R2 = r2_score(y_test, y_pred) 
print("R2:", R2)

plt.text(0.6, 2.5, '$ R^{2} $=' + str(round(R2, 3)))
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test,color ='black',linestyle='-', label='LinearRegression')
plt.xlabel('true-value(TSS)')
plt.ylabel('predict-value(TSS)')
# x軸の範囲を変更する
plt.xlim(6, 18)
# y軸の範囲を変更する
plt.ylim(6, 18)
plt.show()

importances=permutation_importance(rft, X_train, y_train, n_repeats=30,random_state=0)['importances_mean']
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