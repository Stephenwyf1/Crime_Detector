import torch
import torch.nn as nn
from AFSD.common.config import config


def multi_loss(y_true, y_pred):
    confidence_margin = config['training']['confidence_margin']
    batch_size = config['training']['batch_size']
    y_pred = y_pred.transpose(0, 1)
    y_true = torch.tensor(y_true).cuda()
    y_pred = torch.mul((torch.ones(1, batch_size).cuda() - y_true), torch.abs(y_pred)) + \
             torch.mul(y_true, torch.max(torch.zeros(1, batch_size).cuda(), confidence_margin - y_pred))
    return torch.mean(y_pred)


class MultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, negpos_ratio, use_gpu=True,
                 use_focal_loss=False):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.negpos_ratio = negpos_ratio
        # self.mutlti_loss = multi_loss()
        self.use_gpu = use_gpu
        self.use_focal_loss = use_focal_loss

        self.center_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets, pre_locs=None):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """
        conf_vad_score, conf_prop_vad_score = predictions
        y_true = targets
        loss_c = multi_loss(y_true, conf_vad_score)
        loss_prop_c = multi_loss(y_true, conf_prop_vad_score)
        print(y_true)
        print(conf_prop_vad_score)
        return loss_c, loss_prop_c
