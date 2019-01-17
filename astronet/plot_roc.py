import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


y_true_vanilla, y_pred_vanilla = np.loadtxt('true_vs_pred_vanilla_run1.txt', unpack=True)
# y_true_choc, y_pred_choc = np.loadtxt('true_vs_pred_mango.txt', unpack=True)
p_v, r_v, _ = precision_recall_curve(y_true_vanilla, y_pred_vanilla)
# auc_v = auc(p_v, r_v)
# p_c, r_c, _ = precision_recall_curve(y_true_choc, y_pred_choc)
# auc_c = auc(p_c, r_c)


fig = plt.figure(figsize=(8,8))
plt.gca().xaxis.set_tick_params(labelsize=15)
plt.gca().yaxis.set_tick_params(labelsize=15)
plt.plot(r_v, p_v, label='Plain')
# plt.plot(r_c, p_c, label='Improved')
plt.ylabel('Precision', fontsize=15)
plt.xlabel('Recall', fontsize=15)
# plt.xrange(0, 0.7)
# plt.yrange(0.4,1)
plt.legend(fontsize=15)
plt.savefig('precision_recall.png', dpi=150)