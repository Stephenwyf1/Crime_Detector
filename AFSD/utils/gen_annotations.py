import pandas as pd
import numpy as np
#
# data = pd.read_csv('thumos_annotations/val_Annotation.csv')
# df = pd.DataFrame(data)
# normal_list = np.load("thumos_annotations/normal_list.npy")
# abnormal_list = np.load("thumos_annotations/abnormal_list.npy")
# normal_values = []
# abnormal_values = []
# for d in df.values[:]:
#     if d[2] != 0:
#         if d[0] + '.npy' in normal_list:
#             normal_values.append(d)
#         elif d[0] + '.npy' in abnormal_list:
#             abnormal_values.append(d)
#
# print(len(normal_values))
# print(len(abnormal_values))  # 共3007行数据
# df2 = pd.DataFrame(normal_values + abnormal_values, columns=df.columns)
# df2.to_csv('thumos_annotations/val_Annotation_VAD.csv', index=False)
#

def get_ucf_anno():
    df = pd.DataFrame(pd.read_csv('UCF-Crime_annotations/Anomaly_Train.txt'))
    normal = []
    abnormal = []
    for i, d in enumerate(df.values[:]):
        videoName = str(d[0]).split('/')[0]
        label = 0
        if videoName != 'Training_Normal_Videos_Anomaly':
            label = 1
        df.iloc[i, 1] = label

    for d in df.values[:]:
        videoName = str(d[0]).split('/')[0]
        if videoName != 'Training_Normal_Videos_Anomaly':
            abnormal.append(d)
        else:
            normal.append(d)

    print(len(normal))
    print(len(abnormal))
    df = pd.DataFrame(normal + abnormal, columns=df.columns)
    df.to_csv('UCF-Crime_annotations/ucf_train_anno.csv', index=None)

if __name__ == '__main__':
    # get_ucf_anno()
    df = pd.DataFrame(pd.read_csv('UCF-Crime_annotations/ucf_train_anno.csv'))
    dict = {}
    for i,d in enumerate(df.values[:]):
        videoName = str(d[0]).split('/')[0]
        if videoName not in dict and videoName != 'Testing_Normal_Videos_Anomaly':
            dict[videoName] = 0
        if videoName != 'Testing_Normal_Videos_Anomaly' and dict[videoName] == 10:
            df.drop(index=i)
            continue
        if videoName == 'Testing_Normal_Videos_Anomaly':
            df.iloc[i, 1] = 0.
            df.loc[i, 'video'] = str('Normal/'+str(d[0]).split('/')[1])
        else:
            dict[videoName] += 1
    df.to_csv('UCF-Crime_annotations/ucf_train_anno.csv', index=None)
# data = pd.read_csv('thumos_annotations/test_Annotation.csv')
# df = pd.DataFrame(data)
#
# new_values = []
# for d in df.values[:]:
#     if d[2] != 0:
#         new_values.append(d)
#
# df2 = pd.DataFrame(new_values, columns=df.columns)
# df2.to_csv('thumos_annotations/test_Annotation_ours.csv', index=False)
