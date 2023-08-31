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

    test_set_name = opt['datasets']['test_2']['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    #iters_name = str(opt['iters_LR'])
    iters_name = str(opt['iters_HR']) + str('HR') + str(opt['iters_LR'])+str('LR')

    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    dataset_dir = osp.join(dataset_dir, iters_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []


    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    folder_hr = opt['datasets']['test_2']['dataroot_GT']
    folder_lr = opt['datasets']['test_2']['dataroot_LQ']

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder_hr,'*')))):
        torch.cuda.empty_cache()
        img_name, img_hr, img_lr = get_image_pair(path, folder_lr,scale=opt['scale'])  #图像范围0-1
        img_lr = np.transpose(img_lr if img_lr.shape[2] == 1 else img_lr[:, :, [2, 1, 0]],(2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lr = torch.from_numpy(img_lr).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        b, c, h_old, w_old = img_lr.size()


        img_hr = np.transpose(img_hr if img_hr.shape[2] == 1 else img_hr[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_hr = torch.from_numpy(img_hr).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        chang = img_lr.shape[2]
        kuan = img_lr.shape[3]
        img_hr = img_hr[:,:,:chang*opt['scale'], :kuan*opt['scale']]
        #############################################
        # 分割patch， windowsize表示分割的patch大小
        window_size = 128
        #assert tile % window_size == 0, "tile size should be a multiple of window_size"
        #stride = tile - tile_overlap
        h_res = h_old % window_size
        w_res = w_old % window_size

        model.my_feed_data(img_lr, img_hr)
        model.test()
        visuals = model.get_current_visuals()
        ori_sr = visuals['SR'].cuda()

        h_idx_list = list(range(0, h_old - window_size, window_size))
        w_idx_list = list(range(0, w_old - window_size, window_size))
        len_h = len(h_idx_list)
        len_w = len(w_idx_list)
        E = torch.zeros(b, c, h_old*opt['scale'], w_old*opt['scale']).type_as(img_lr)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                print(w_idx)
                in_patch = img_lr[..., h_idx:h_idx+window_size, w_idx:w_idx+window_size]
                hr_patch = img_hr[..., h_idx * opt['scale']:(h_idx+window_size)*opt['scale'], w_idx*opt['scale']:(w_idx+window_size)*opt['scale']]
                # out_patch = model(in_patch)

                HR_delta = attack_pgd_sr(model, in_patch, hr_patch, epsilon=0.3, alpha=20 / 255,
                                         attack_iters=opt['iters_HR'], restarts=1,
                                         norm="l_2",
                                         h_old=h_old, w_old=w_old, scale=opt['scale'])
                HR_delta = HR_delta.detach()

                LR_delta = attack_pgd(model, HR_delta, hr_patch, epsilon=0.3, alpha=20 / 255, attack_iters=opt['iters_LR'],
                                      restarts=1, norm="l_2",
                                      h_old=h_old, w_old=w_old, scale=opt['scale'])
                LR_delta = LR_delta.detach()

                model.my_feed_data(LR_delta, hr_patch)
                model.my_test()
                visuals = model.get_current_visuals()

                out_patch = visuals['SR'].cuda()
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * opt['scale']:(h_idx+window_size)*opt['scale'], w_idx*opt['scale']:(w_idx+window_size)*opt['scale']].add_(out_patch)
                W[..., h_idx * opt['scale']:(h_idx+window_size)*opt['scale'], w_idx*opt['scale']:(w_idx+window_size)*opt['scale']].add_(out_patch_mask)

        patch1 = ori_sr[..., :h_old * opt['scale'], len_w * window_size * opt['scale']:]
        patch1_mask = torch.ones_like(patch1)
        E[..., :h_old * opt['scale'], len_w * window_size * opt['scale']:].add_(patch1)
        W[..., :h_old * opt['scale'], len_w * window_size * opt['scale']:].add_(patch1_mask)

        patch2 = ori_sr[..., len_h * window_size * opt['scale']:, :len_w * window_size * opt['scale'] ]
        patch2_mask = torch.ones_like(patch2)
        E[..., len_h * window_size * opt['scale']:, :len_w * window_size * opt['scale'] ].add_(patch2)
        W[..., len_h * window_size * opt['scale']:, :len_w * window_size * opt['scale'] ].add_(patch2_mask)

        sr_output = E.div_(W)


        sr_img = util.tensor2img(sr_output)
        gt_img = util.tensor2img(img_hr)

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)


        # calculate PSNR and SSIM
        #gt_img = util.tensor2img(visuals['GT'])

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
                '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}. '.
                format(img_name, psnr, ssim, psnr_y, ssim_y))
        else:
            logger.info(
                '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}. LR PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim,
                                                                                                   psnr_lr, ssim_lr))

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])



    logger.info(
        '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. \n'.format(
            test_set_name, ave_psnr, ave_ssim))
    if test_results['psnr_y'] and test_results['ssim_y']:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

        logger.info(
            '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.\n'.
            format(ave_psnr_y, ave_ssim_y))


def get_image_pair(path, folder_lr, scale):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_hr = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    img_lr = cv2.imread(f'{folder_lr}/{imgname}x{scale}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_hr, img_lr

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model,  y, y_gt, epsilon, alpha, attack_iters, restarts,
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
        elif norm == "my_defined":
            delta = 0.00392157 * torch.randint_like(X, low=-1, high=2, requires_grad=False)
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
            loss = mseloss(output, y_gt)
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

        all_loss = mseloss(new_output, y_gt)

        max_delta[all_loss <= max_loss] = delta.detach()[all_loss <= max_loss]
        max_loss = torch.min(max_loss, all_loss)
    return max_delta + X

def attack_pgd_sr(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               h_old=256, w_old=256, scale=2):
    model.set_grad_False()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    # X = model.downscale(y).to(device)

    max_loss = torch.zeros(y.shape[0]).to(device)
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
        elif norm == "my_defined":
            delta = 0.00392157 * torch.randint_like(X, low=-1, high=2, requires_grad=False)
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


