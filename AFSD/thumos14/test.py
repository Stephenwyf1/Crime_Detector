import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.utils import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_class_index_map
from AFSD.thumos14.model.BDNet import BDNet
from AFSD.utils.segment_utils import softnms_v2
from AFSD.common.config import config

num_classes = config['dataset']['num_classes']
conf_thresh = config['testing']['conf_thresh']
top_k = config['testing']['top_k']
nms_thresh = config['testing']['nms_thresh']
nms_sigma = config['testing']['nms_sigma']
clip_length = config['dataset']['testing']['clip_length']
stride = config['dataset']['testing']['clip_stride']
checkpoint_path = config['testing']['checkpoint_path']
json_name = config['testing']['output_json']
output_path = config['testing']['output_path']
softmax_func = True
if not os.path.exists(output_path):
    os.makedirs(output_path)
fusion = config['testing']['fusion']

# getting path for fusion
rgb_data_path = config['testing'].get('rgb_data_path',
                                      './datasets/thumos14/test_npy/')
flow_data_path = config['testing'].get('flow_data_path',
                                       './datasets/thumos14/test_flow_npy/')
rgb_checkpoint_path = config['testing'].get('rgb_checkpoint_path',
                                            './models/thumos14/checkpoint-15.ckpt')
flow_checkpoint_path = config['testing'].get('flow_checkpoint_path',
                                             './models/thumos14_flow/checkpoint-16.ckpt')

if __name__ == '__main__':
    video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
    originidx_to_idx, idx_to_class1 = get_class_index_map()
    idx_to_class = {1: 'abnormal', 0: 'normal'}
    npy_data_path = config['dataset']['testing']['video_data_path']
    net = BDNet(in_channels=config['model']['in_channels'], training=False)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval().cuda()

    if softmax_func:
        score_func = nn.Softmax(dim=-1)
        # score_func = torch.norm(dim=2)
    else:
        score_func = nn.Sigmoid()

    centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])

    result_dict = {}
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        result_dict[video_name] = []
        sample_count = video_infos[video_name]['sample_count']
        sample_fps = video_infos[video_name]['sample_fps']
        if sample_count < clip_length:
            offsetlist = [0]
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]

        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])  # 调换数据轴？？作用是什么
        data = centor_crop(data)
        data = torch.from_numpy(data)

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        tempList = []
        # print(video_name)
        count = 0
        for offset in offsetlist:
            clip = data[:, offset: offset + clip_length]
            clip = clip.float()
            clip = (clip / 255.0) * 2.0 - 1.0
            # if fusion:
            #     flow_clip = flow_data[:, offset: offset + clip_length]
            #     flow_clip = flow_clip.float()
            #     flow_clip = (flow_clip / 255.0) * 2.0 - 1.0
            # clip = torch.from_numpy(clip).float()
            if clip.size(1) < clip_length:
                tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                                   96, 96]).float()
                clip = torch.cat([clip, tmp], dim=1)
            clip = clip.unsqueeze(0).cuda()  # 行扩展

            with torch.no_grad():
                output_dict = net(clip)
            loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors'][0]
            conf_scores_ = [output_dict['conf_VAD_score'], output_dict['conf_prop_VAD_score']]
            prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
            center = output_dict['center']
            real_score = (conf_scores_[0].item() + conf_scores_[1].item()) / 2
            data_x = {'No': count, 'offset': offset, 'score': real_score}
            count += 1
            print(data_x)
            result_dict[video_name].append(data_x)

    result = {'result': dict(result_dict)}
    with open(os.path.join(output_path, json_name), "w") as out:
        json.dump(result, out)

