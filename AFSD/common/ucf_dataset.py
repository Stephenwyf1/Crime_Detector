import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from AFSD.utils import videotransforms
from AFSD.common.config import config
import random
import math


def get_video_info(video_info_path):
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'fps': info[1],
            'sample_fps': info[2],
            'count': info[3],
            'sample_count': info[4]
        }
    return video_infos


def get_video_anno(video_infos,
                   video_anno_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        label = anno[1]
        video_annos[video_name] = label
    return video_annos


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res


def split_videos(video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 stride=config['dataset']['training']['clip_stride']):

    training_list = []
    for video_name in video_annos.keys():
        # if 'Normal' not in video_name:  # normal的结束点，从此开始采样异常数据
        #     print(len(training_list))
        sample_count = video_infos[video_name]['sample_count']
        annos = video_annos[video_name]
        if sample_count <= clip_length:
            offsetlist = [0]
            # print('{} is '.format(video_name))
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]
        for offset in offsetlist:
            training_list.append({
                'video_name': str(video_name).split('/')[1].split('.')[0],
                'offset': offset,
                'annos': annos,
            })
    return training_list


# def load_video_data(video_infos, npy_data_path):
#     data_dict = {}
#     print('loading video frame data ...')
#     for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
#         data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
#         data = np.transpose(data, [3, 0, 1, 2])
#         data_dict[video_name] = data
#     return data_dict


class UCF_Dataset(Dataset):
    def __init__(self,data_dict,
                 video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 crop_size=config['dataset']['training']['crop_size'],
                 stride=config['dataset']['training']['clip_stride'],
                 rgb_norm=True,
                 training=True,
                 origin_ratio=0.5):
        self.training_list = split_videos(
            video_infos,
            video_annos,
            clip_length,
            stride
        )
        # np.random.shuffle(self.training_list)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)
        self.rgb_norm = rgb_norm
        self.training = training
        self.data_dict = data_dict
        self.origin_ratio = origin_ratio

    def __len__(self):
        return len(self.training_list)

    # def get_bg(self, annos, min_action):
    #     annos = [[anno[0], anno[1]] for anno in annos]
    #     times = []
    #     for anno in annos:
    #         times.extend(anno)
    #     times.extend([0, self.clip_length - 1])
    #     times.sort()
    #     regions = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
    #     regions = list(filter(
    #         lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action, regions))
    #     # regions = list(filter(lambda x:x not in annos, regions))
    #     region = random.choice(regions)
    #     return [math.ceil(region[0]), math.floor(region[1])]

    # def augment_(self, input, annos, th):
    #     '''
    #     input: (c, t, h, w)
    #     target: (N, 3)
    #     '''
    #     try:
    #         gt = random.choice(list(filter(lambda x: x[1] - x[0] > 2 * th, annos)))
    #         # gt = random.choice(annos)
    #     except IndexError:
    #         return input, annos, False
    #     gt_len = gt[1] - gt[0]
    #     region = range(math.floor(th), math.ceil(gt_len - th))
    #     t = random.choice(region) + math.ceil(gt[0])
    #     l_len = math.ceil(t - gt[0])
    #     r_len = math.ceil(gt[1] - t)
    #     try:
    #         bg = self.get_bg(annos, th)
    #     except IndexError:
    #         return input, annos, False
    #     start_idx = random.choice(range(bg[1] - bg[0] - th)) + bg[0]
    #     end_idx = start_idx + th
    #
    #     new_input = input.clone()
    #     # annos.remove(gt)
    #     if gt[1] < start_idx:
    #         new_input[:, t:t + th, ] = input[:, start_idx:end_idx, ]
    #         new_input[:, t + th:end_idx, ] = input[:, t:start_idx, ]
    #
    #         new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]
    #         # new_annos = [[t-math.ceil(th/5), t+math.ceil(th/5)],
    #         #            [t+th-math.ceil(th/5), t+th+math.ceil(th/5)],
    #         #            [t+1, t+th-1]]
    #
    #     else:
    #         new_input[:, start_idx:t - th] = input[:, end_idx:t, ]
    #         new_input[:, t - th:t, ] = input[:, start_idx:end_idx, ]
    #
    #         new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]
    #         # new_annos = [[t-th-math.ceil(th/5), t-th+math.ceil(th/5)],
    #         #            [t-math.ceil(th/5), t+math.ceil(th/5)],
    #         #            [t-th+1, t-1]]
    #
    #     return new_input, new_annos, True

    # def augment(self, input, annos, th, max_iter=10):
    #     flag = True
    #     i = 0
    #     while flag and i < max_iter:
    #         new_input, new_annos, flag = self.augment_(input, annos, th)
    #         i += 1
    #     return new_input, new_annos, flag

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_data = self.data_dict[sample_info['video_name']]
        # video_data = self.data_dict[sample_info['video_name']]
        offset = sample_info['offset']
        annos = sample_info['annos']
        # th = self.th[sample_info['video_name']]
        input_data = video_data[:, offset: offset + self.clip_length]
        c, t, h, w = input_data.shape
        if t < self.clip_length:
            # padding t to clip_length
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
            input_data = np.concatenate([input_data, zero_clip], 1)

        # random crop and flip
        if self.training:
            input_data = self.random_flip(self.random_crop(input_data))
        else:
            input_data = self.center_crop(input_data)

        # import pdb;pdb.set_trace()
        input_data = torch.from_numpy(input_data).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        # ssl_input_data, ssl_annos, flag = self.augment(input_data, annos, th, 1)
        # annos = annos_transform(annos, self.clip_length)
        annos1 = []
        annos1.append(annos)
        target = np.array(annos1)
        # ssl_target = np.stack(ssl_annos, 0)

        # scores = np.stack([
        #     sample_info['start'],
        #     sample_info['end']
        # ], axis=0)
        # scores = torch.from_numpy(scores.copy()).float()

        return input_data, target


def detection_collate(batch):
    targets = []
    clips = []
    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(clips, 0), targets
