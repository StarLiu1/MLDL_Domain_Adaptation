from torch import nn
from torch.utils import model_zoo
import torch
from torch import nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, classes=100, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x, gt=None, flag=None, epoch=None) -> torch.Tensor:
        x = self.features(x)
        
        if flag:
            interval = 10
            if epoch % interval == 0:
                self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10

            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.classifier(x_new_view)
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                mask_all = self.spatial_RCS(num_rois, H, HW, spatial_mean)
            else:
                # -------------------------- channel ----------------------------
                mask_all = self.channel_RCS(num_rois, num_channel, channel_mean)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def channel_RCS(self, num_rois, num_channel, channel_mean):
        vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
        vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
        vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
        vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
        mask_all = vector.view(num_rois, num_channel, 1, 1)
        return mask_all

    def spatial_RCS(self, num_rois, H, HW, spatial_mean):
        spatial_drop_num = math.ceil(HW * 1 / 3.0)
        th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
        th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 36)
        mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
        mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
        return mask_all
    
def alex(pretrained=True, **kwargs):
    """
    Constructs a custom alexnet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        # Load the original AlexNet
        alexnet_original = models.alexnet(pretrained=True)

        # Copy weights for the common layers
        # Features
        for layer, param in model.features.named_parameters():
            param.data = alexnet_original.features.state_dict()[layer].data

        # Classifier (except for the final layer)
        for layer, param in model.classifier.named_parameters():
            if '6' not in layer:  # Skip the final layer
                param.data = alexnet_original.classifier.state_dict()[layer].data

    return model