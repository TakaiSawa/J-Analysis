import pandas as pd
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import LogisticRegression
from interpret.glassbox import LinearRegression
from interpret.glassbox import ClassificationTree
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.glassbox import RegressionTree
from interpret.perf import ROC
from interpret.perf import RegressionPerf
from interpret import show
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
import xgboost as xgb
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


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
X_start = 10 #400nm
X_end = -29 #1000nm
y_place = -6 
X_feature_names = df.get_row(0)[X_start : X_end]
y_feature_names = df.get_row(0)[y_place]
df = df.as_float()
df.remove_row(0)
y = df.get_column(y_place)
X = df.get_columns(X_start, X_end)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#model = ExplainableBoostingClassifier(random_state=seed)
#model = LogisticRegression(random_state=seed)
#model = LinearRegression(random_state=seed)
#model = ClassificationTree(random_state=seed)
#model = RegressionTree(random_state=seed)

model = ExplainableBoostingRegressor(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Square Error:", RMSE)
R2 = r2_score(y_test, y_pred) 
print("R2:", R2)

# model_global = model.explain_global()
# show(model_global)


# #model_local = model.explain_local(X_test, y_test)
# #show(model_local)

# model_perf = RegressionPerf(model.predict).explain_perf(X_test, y_test, name='EBM')
# show(model_perf)
# #print(vars(model_local))

# print(model_perf._internal_obj["overall"]["r2"])
# print(vars(model_perf))