import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

from AFSD.common.ucf_dataset import UCF_Dataset, get_video_info, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from AFSD.ucf.model.BDNet import BDNet
from AFSD.ucf.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']
ngpu = config['ngpu']

train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('ssl weight: ', config['training']['ssl'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)
    print('gpu num: ', ngpu)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def load_video_data(video_infos, npy_data_path):
    data_dict = {}
    print('loading video frame data ...')
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        video_name = str(video_name).split('/')[1]
        video_name = video_name.split('.')[0]
        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data_dict[video_name] = data
    return data_dict


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    torch.save(model.module.state_dict(),
               os.path.join(checkpoint_path,
                            'VAD-checkpoint-ucf:{}-{}.ckpt'.format(config['training']['confidence_margin'], epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path,
                            'VAD-checkpoint-ucf:{}-{}.ckpt'.format(config['training']['confidence_margin'], epoch)))


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path,
                                  'VAD-checkpoint-ucf:{}-{}.ckpt'.format(config['training']['confidence_margin'],
                                                                         resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path,
                                  'VAD-checkpoint-ucf:{}-{}.ckpt'.format(config['training']['confidence_margin'],
                                                                         resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
    return start_epoch


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous().view(-1).cuda(),
                                        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous().view(-1).cuda(),
                                      reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, training=True, ssl=True):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]
    if training:
        if ssl:
            output_dict = net.module(clips, proposals=targets, ssl=ssl)
        else:
            output_dict = net(clips, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    loss_c, loss_prop_c = CPD_Loss(predictions=[output_dict['conf_VAD_score'], output_dict['conf_prop_VAD_score']],
                                   targets=targets)
    return loss_c, loss_prop_c


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_list = []
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets) in enumerate(pbar):
            loss_c, loss_prop_c = forward_one_epoch(net, clips, targets, training=training, ssl=False)
            loss_c = loss_c * config['training']['cw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            cost = loss_c + loss_prop_c
            if training:
                optimizer.zero_grad()
                if n_iter % 20 == 0:
                    loss_list.append(cost.item())
                    print('Epoch ' + str(epoch) + ' : ' + str(n_iter / 20) + ' , LOSS =' + str(cost.item()))
                cost.backward()
                optimizer.step()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    if training:
        prefix = 'Train'
        save_model(epoch, net, optimizer)
        np.array(loss_list)
        np.save('output/epoch_loss_UCF-Crime:{}'.format(epoch), loss_list)
    else:
        prefix = 'Val'

    # plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, ' \
    #        'prop_loc - {:.5f}, prop_conf - {:.5f}, ' \
    #        'IoU - {:.5f}, start - {:.5f}, end - {:.5f}'.format(
    #     i, prefix, cost_val, loss_loc_val, loss_conf_val, loss_prop_l_val, loss_prop_c_val
    #       loss_ct_val, loss_start_val, loss_end_val
    # )
    # plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, ' \
    #        'prop_loc - {:.5f}, prop_conf - {:.5f}, ' \
    #        .format(
    #     i, prefix, cost_val, loss_loc_val, loss_conf_val, loss_prop_l_val, loss_prop_c_val
    #     # loss_ct_val, loss_start_val, loss_end_val
    # )
    # plog = plog + ', Triplet - {:.5f}'.format(loss_trip_val)
    # print(plog)


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    """
    Setup model
    """
    net = BDNet(in_channels=config['model']['in_channels'],
                backbone_model=config['model']['backbone_model'])
    net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()

    """
    Setup optimizer
    """
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)

    """
    Setup dataloader
    """
    train_video_infos = get_video_info(config['ucf']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['ucf']['training']['video_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['ucf']['training']['video_data_path'])
    train_dataset = UCF_Dataset(train_data_dict, train_video_infos, train_video_annos)

    from AFSD.ucf.ucf_sampler import UCFSampler

    _batch_sampler = UCFSampler(train_dataset, batch_size)
    train_data_loader = DataLoader(train_dataset, num_workers=4, worker_init_fn=worker_init_fn
                                   , collate_fn=detection_collate,
                                   batch_sampler=_batch_sampler, pin_memory=True)
    # train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
    #                                num_workers=4, worker_init_fn=worker_init_fn,
    #                                collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net, optimizer, train_data_loader, len(train_dataset) // batch_size)
