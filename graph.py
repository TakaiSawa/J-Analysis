#グラフの表示色々

#残差プロット
# 残差プロット
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