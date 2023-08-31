import os.path as osp
import logging
import time
import argparse
import glob
import os
import cv2
from collections import OrderedDict
import torch
import torch.nn  as nn
import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from models.modules.Quantization import Quantization


def main():

    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(opt)

    test_set_name = opt['datasets']['test_1']['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))


    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)


    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    folder_hr = opt['datasets']['test_1']['dataroot_GT']
    folder_lr = opt['datasets']['test_1']['dataroot_LQ']

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder_hr,'*')))):
        torch.cuda.empty_cache()
        img_name, img_hr, img_lr = get_image_pair(path, folder_lr,scale=opt['scale'])  #图像范围0-1
        img_lr = np.transpose(img_lr if img_lr.shape[2] == 1 else img_lr[:, :, [2, 1, 0]],(2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lr = torch.from_numpy(img_lr).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        _, _, h_old, w_old = img_lr.size()

        img_hr = np.transpose(img_hr if img_hr.shape[2] == 1 else img_hr[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_hr = torch.from_numpy(img_hr).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        chang = img_lr.shape[2]
        kuan = img_lr.shape[3]
        img_hr = img_hr[:,:,:chang*opt['scale'], :kuan*opt['scale']]
        #############################################
        # 1、梯度改变前向生成的LR图像
        # #gpu_tracker.track()
        # delta = attack_pgd(model,  img_hr, epsilon=0.3, alpha=20 / 255, attack_iters=opt['iters_LR'], restarts=1,norm="l_2",h_old=h_old, w_old=w_old, scale=opt['scale'])
        # delta = delta.detach()
        # #gpu_tracker.track()
        # delta = delta[:,:,:h_old,:w_old]
        #
        # model.my_feed_data(delta, img_hr)
        # model.my_test()
        #############################################
        # 2 预定义的LR图像
        # model.my_feed_data(img_lr, img_hr)
        # model.my_test()
        #############################################
        # 3、梯度改变HR图像
        # delta = attack_pgd_sr(model, img_lr, img_hr, epsilon=0.3, alpha=20 / 255, attack_iters=10, restarts=1, norm="l_2",
        #                    h_old=h_old, w_old=w_old, scale=opt['scale'])
        # delta = delta.detach()
        #
        # model.my_feed_data(img_lr, delta)
        # model.test()

        #############################################
        # 4、先改变HR，再改变LR
        HR_delta = attack_pgd_sr(model, img_lr, img_hr, epsilon=0.3, alpha=20 / 255, attack_iters=opt['iters_HR'], restarts=1,
                                 norm="l_2",
                                 h_old=h_old, w_old=w_old, scale=opt['scale'])
        HR_delta = HR_delta.detach()

        LR_delta = attack_pgd(model, HR_delta, epsilon=0.3, alpha=20 / 255, attack_iters=opt['iters_LR'], restarts=1, norm="l_2",
                              h_old=h_old, w_old=w_old, scale=opt['scale'])
        LR_delta = LR_delta.detach()

        model.my_feed_data(LR_delta, img_hr)
        model.my_test()

        #############################################

        #adv_output = torch.clamp(img_lr + delta[:img_lr.size(0)], min=0, max=1)
        #adv_output = adv_output[:, :, :h_old, :w_old]
        #############################################
        # 原结果
        # model.my_feed_data(img_lr, img_hr)
        # model.test()
        #############################################
        visuals = model.get_current_visuals()

        sr_img = util.tensor2img(visuals['SR'])  # uint8, rev SR
        lr_img = util.tensor2img(visuals['LR'])  # uint8, forward lr

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)


        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_LR.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_LR.png')
        util.save_img(lr_img, save_img_path)


        gt_img = util.tensor2img(visuals['GT'])

        gt_img = gt_img / 255.
        sr_img = sr_img / 255.


        crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
        if crop_border == 0:
            cropped_sr_img = sr_img
            cropped_gt_img = gt_img
        else:
            cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

        psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)


        if gt_img.shape[2] == 3:  # RGB image
            sr_img_y = bgr2ycbcr(sr_img, only_y=True)
            gt_img_y = bgr2ycbcr(gt_img, only_y=True)
            if crop_border == 0:
                cropped_sr_img_y = sr_img_y
                cropped_gt_img_y = gt_img_y
            else:
                cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
            psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
            ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)


            logger.info(
                '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                format(img_name, psnr, ssim, psnr_y, ssim_y))
        else:
            logger.info(
                '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}. '.format(img_name, psnr, ssim))

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])



    logger.info(
        '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. \n'.format(test_set_name, ave_psnr, ave_ssim))
    if test_results['psnr_y'] and test_results['ssim_y']:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])


        logger.info(
            '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.\n'.format(ave_psnr_y, ave_ssim_y))


def get_image_pair(path, folder_lr, scale):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_hr = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    img_lr = cv2.imread(f'{folder_lr}/{imgname}x{scale}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_hr, img_lr

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model,  y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               h_old=256, w_old=256, scale=2):
    model.set_grad_False()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    X = model.downscale(y).to(device)
    Quantizations = Quantization()
    X = Quantizations(X)

    max_loss = torch.tensor([100.0]).to(device)  # 加一个负号就是负无穷大。
    max_delta = torch.zeros_like(X).to(device)


    mseloss = nn.MSELoss(reduction='mean')
    for _ in range(restarts):
        #delta = torch.zeros_like(X).cuda()
        delta = torch.zeros_like(X)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            torch.cuda.empty_cache()
            output = model.my_upscale(X + delta, scale)
            output = output[..., :h_old * scale, :w_old * scale]

            index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = mseloss(output, y)
            loss.requires_grad_(True)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d - scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        new_output = model.my_upscale(X + delta, scale)
        new_output = new_output[..., :h_old * scale, :w_old * scale]

        all_loss = mseloss(new_output, y)

        max_delta[all_loss <= max_loss] = delta.detach()[all_loss <= max_loss]
        max_loss = torch.min(max_loss, all_loss)
    return max_delta + X

def attack_pgd_sr(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               h_old=256, w_old=256, scale=2):
    model.set_grad_False()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)

    max_loss = torch.tensor([100.0]).to(device)  # 加一个负号就是负无穷大。
    max_delta = torch.zeros_like(y).to(device)


    mseloss = nn.MSELoss(reduction='mean')
    for _ in range(restarts):
        #delta = torch.zeros_like(X).cuda()
        delta = torch.zeros_like(y)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, 0-y, 1-y)
        delta.requires_grad = True
        for _ in range(attack_iters):
            torch.cuda.empty_cache()
            output = model.my_downscale(y + delta)
            output = model.my_upscale(output, scale)
            output = output[..., :h_old * scale, :w_old * scale]

            index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = mseloss(output, y)
            loss.requires_grad_(True)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            m = y[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d - scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

            d = clamp(d, 0 - m, 1 - m)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        new_output = model.my_downscale(y + delta)
        new_output = model.my_upscale(new_output, scale)
        new_output = new_output[..., :h_old * scale, :w_old * scale]
        all_loss = mseloss(new_output, y)

        max_delta[all_loss <= max_loss] = delta.detach()[all_loss <= max_loss]
        max_loss = torch.min(max_loss, all_loss)
    return max_delta + y


if __name__ == '__main__':
    main()