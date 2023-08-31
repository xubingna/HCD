# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16
import warnings

warnings.filterwarnings('ignore')


# 计算特征提取模块的感知损失
def vgg16_loss(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss=loss_func(out,out_)
    return loss


# 获取指定的特征提取模块
def get_feature_module(layer_index,device=None):
    vgg = vgg16(pretrained=True, progress=True).features
    vgg.eval()

    # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False

    feature_module = vgg[0:layer_index + 1]
    feature_module.to(device)
    return feature_module


# 计算指定的组合模块的感知损失
class PerceptualLoss(nn.Module):
    def __init__(self,loss_func,layer_indexs=None,device=None):
        super(PerceptualLoss, self).__init__()
        self.creation=loss_func
        self.layer_indexs=layer_indexs
        self.device=device

    def forward(self,y,y_):
        loss=0
        for index in self.layer_indexs:
            feature_module=get_feature_module(index,self.device)
            loss+=vgg16_loss(feature_module,self.creation,y,y_)
        return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones((1, 3, 256, 256))
    y = torch.zeros((1, 3, 256, 256))
    x,y=x.to(device),y.to(device)

    layer_indexs = [3, 8, 15, 22]
    # 基础损失函数：确定使用那种方式构成感知损失，比如MSE、MAE
    loss_func = nn.MSELoss().to(device)
    # 感知损失
    creation = PerceptualLoss(loss_func, layer_indexs, device)
    perceptual_loss=creation(x,y)
    print(perceptual_loss)
