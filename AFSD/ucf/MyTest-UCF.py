import json
import numpy as np
from sklearn import *
import torch
import matplotlib.pyplot as plt
with open("output/detection_results_UCF.json", 'r') as prediction:
    prediction = json.load(prediction)

prediction = prediction['result']

predict_ = []
y_true = []
for video_name in prediction:
    label = 1 if 'Normal' not in video_name else 0
    for clip in prediction[video_name]:
        if label == 0:
            print(clip)
        score = clip['score']
        y_true.append(label)
        predict_.append(score)


predict_ = torch.from_numpy(np.array(predict_))
predict_ = predict_.numpy()
print('y_true:{}\nprediction:{}'.format(y_true, predict_))
print("y_true:{},prediction:{}".format(len(y_true),len(predict_)))
fpr,tpr,thresh = metrics.roc_curve(y_true,predict_)
plt.plot(fpr,tpr,'r-',linewidth=1,label='ROC')
plt.title('UCF-CRIME-ROC',fontsize = 14, fontproperties = 'Times New Roman')
plt.xlabel("FPR",fontsize = 13,fontproperties = 'Times New Roman',fontweight='bold')
plt.ylabel("TPR",fontsize = 13,fontproperties = 'Times New Roman', fontweight='bold')
plt.yticks(fontproperties='Times New Roman', size=13 )#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=13)
plt.legend(loc=4)
plt.show()
roc = metrics.roc_auc_score(y_true, predict_)

print(roc)

# pr = np.array([[1., 2., 5., 67., 89., 0., 34., 4., 5., 6.]])
# print(pr)
# pr = torch.from_numpy(pr)
# pr = torch.softmax(pr, 1)
# count =0
# for i in range(pr.shape[1]):
#     count+=pr[0][i].item()
# print(pr)
# print(count)
