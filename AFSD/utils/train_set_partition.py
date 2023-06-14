import os

from AFSD.common.config import config
import pandas as pd
import numpy as np
import shutil

# generate the abnormal list
abnormal_class = ["CleanAndJerk", "BaseballPitch", "CricketBowling", "FrisbeeCatch", "JavelinThrow", "SoccerPenalty",
                  "Diving", "CliffDiving", "CricketShot"]
train_set_path = config["dataset"]["training"]["video_data_path"]
annotations = pd.DataFrame(pd.read_csv(config["dataset"]["training"]["video_anno_path"])).values[:]


def createAbnormalData():
    abnormal_data_list = []
    for i in annotations:
        video_name = i[0] + ".npy"
        video_class = i[1]
        if video_class in abnormal_class and video_name not in abnormal_data_list:
            abnormal_data_list.append(video_name)
            print(video_name)
    abnormal_data_path = "datasets/thumos14/abnormal/"
    if not os.path.exists(abnormal_data_path):
        os.makedirs(abnormal_data_path)
    # for video_name in abnormal_data_list:
    #     shutil.copy(train_set_path + video_name, abnormal_data_path)
    np.save("thumos_annotations/abnormal_list.npy", abnormal_data_list)
    print(len(abnormal_data_list))

def createNormalData():
    normal_data_list = []
    for i in annotations:
        video_name = i[0] + ".npy"
        video_class = i[1]
        if video_class not in abnormal_class and video_name not in normal_data_list:
            normal_data_list.append(video_name)
            print(video_name)
    normal_data_path = "datasets/thumos14/normal/"
    if not os.path.exists(normal_data_path):
        os.makedirs(normal_data_path)
    # for video_name in normal_data_list:
    #     shutil.copy(train_set_path + video_name, normal_data_path)
    np.save("thumos_annotations/normal_list.npy", normal_data_list)
    print(len(normal_data_list))


createAbnormalData()
createNormalData()
