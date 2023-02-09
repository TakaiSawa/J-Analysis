from utils import *
from structures import *
from algorithms import *
import matplotlib.pyplot as plt
import csv



df = CsvStructure('dataset_Indoor.csv').remove_columns(0,2)
y_place = -6
y_feature_names = df.get_row(0)[y_place]
df.remove_row(0)
df = df.as_float()
with open('output.csv','w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['wavelength', 'R2', 'a','b'])

    for X_place in range(151):
    
        X_feature_names = 350 + X_place * 5
        y = np.array(df.get_column(y_place))
        X = np.array(df.get_column(X_place)).reshape(-1, 1)
        
        # 線形回帰モデルを作成
        model = LinearRegression()

        # モデルの訓練
        model.fit(X, y)
        y_pred = model.predict(X)

        r2_score = model.score(X, y)
        a = model.coef_
        b = model.intercept_
        #決定係数
        writer.writerow([X_feature_names, np.round(r2_score, decimals=2), np.round(a, decimals=2), np.round(b, decimals=2)])

        # 回帰直線
        # plt.text(0.6, 5, '$ y $=' + str(a)+'$ x $+'+str(b))
        # plt.text(0.6, 2.5, '$ R^{2} $=' + str(round(r2_score, 4)))
        # plt.scatter(X, y, color='blue', label='data')
        # plt.plot(X, y_pred, color='red', label='LinearRegression')
        # plt.title('Scatter Plot of X vs y')    # 図のタイトル
        # plt.xlabel('wavelength') # x軸のラベル
        # plt.ylabel('TSS(Brix.)')    # y軸のラベル
        # plt.legend(loc='lower right')
        # plt.xlim(0, 1.0) # (3)x軸の表示範囲
        # plt.ylim(0, 20.0) # (4)y軸の表示範囲
        # plt.show()