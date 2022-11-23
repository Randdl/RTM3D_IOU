"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.10
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process.kitti_dataloader import create_test_dataloader, create_train_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import rtm3d_decode, post_processing_2d, get_final_pred, draw_predictions
from utils.torch_utils import _sigmoid
from scipy.optimize import least_squares
from math import sin, cos

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for RTM3D Implementation')
    parser.add_argument('--saved_fn', type=str, default='rtm3d', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--head_conv', type=int, default=-1,
                        help='conv layer channels for output head'
                             '0 for no conv layer'
                             '-1 for default setting: '
                             '64 for resnets and 256 for dla.')
    parser.add_argument('--K', type=int, default=100,
                        help='the number of top K')
    parser.add_argument('--use_left_cam_prob', type=float, default=1.,
                        help='The probability of using the left camera')
    parser.add_argument('--dynamic-sigma', action='store_true',
                        help='If true, compute sigma based on Amax, Amin then generate heamap'
                             'If false, compute radius as CenterNet did')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--peak_thresh', type=float, default=0.2)

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_rtm3d', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only
    configs.input_size = (384, 1280)
    configs.hm_size = (96, 320)
    configs.down_ratio = 4
    configs.max_objects = 50

    if configs.head_conv == -1:  # init default head_conv
        configs.head_conv = 256 if 'dla' in configs.arch else 64

    configs.num_classes = 3
    configs.num_vertexes = 8
    configs.num_center_offset = 2
    configs.num_vertexes_offset = 2
    configs.num_dimension = 3
    configs.num_rot = 8
    configs.num_depth = 1
    configs.num_wh = 2
    configs.heads = {
        'hm_mc': configs.num_classes,
        'hm_ver': configs.num_vertexes,
        'vercoor': configs.num_vertexes * 2,
        'cenoff': configs.num_center_offset,
        'veroff': configs.num_vertexes_offset,
        'dim': configs.num_dimension,
        'rot': configs.num_rot,
        'depth': configs.num_depth,
        'wh': configs.num_wh
    }

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
    make_folder(configs.results_dir)

    return configs


def denormalize_img(img):
    mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    return ((img * std_rgb + mean_rgb) * 255).astype(np.uint8)


def np_computeBox3D(label, P):
    w = label[0]
    h = label[1]
    l = label[2]
    x = label[3]
    y = label[4]
    z = label[5]
    ry = label[6]

    R = np.array([[+cos(ry), 0, +sin(ry)],
                  [0, 1, 0],
                  [-sin(ry), 0, +cos(ry)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])

    corners_3D = R.dot(corners_3D)

    corners_3D += np.array([x, y, z]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))

    corners_2D = P.dot(corners_3D_1)

    corners_2D = corners_2D / corners_2D[2]

    corners_2D = corners_2D[:2]
    ver_coor = corners_2D[:, [4, 3, 2, 1, 5, 6, 7, 0]]
    corners_2D = corners_2D[:, [3, 2, 7, 6, 4, 1, 0, 5]]
    return corners_2D, corners_3D, ver_coor


def diff_fun(input, real_corners, P, pred_depth, dim, rot):
    vertices, corners_3D, _ = np_computeBox3D(input, P)
    # print('vertices', vertices)
    # mid_point = np.mean(corners_3D, axis=1)
    # depth = np.sqrt(np.sum(np.square(mid_point)))
    vertices = np.transpose(vertices)
    vertices = vertices.flatten()
    # print(real_corners)
    # print(vertices)
    diff = real_corners - vertices
    diff *= 1e-1

    dim_diff = input[0:3] - dim
    depth_diff = input[5] - pred_depth
    rot_diff = input[6] - rot
    return np.append(np.concatenate((diff.flatten(), dim_diff)), [depth_diff, rot_diff])
    # return diff.flatten()


def gen_position(kps,dim,rot,meta,const,x):
    b=kps.size(0)
    c=kps.size(1)
    # opinv=meta['trans_output_inv']
    calib=meta['calib']

    # opinv = opinv.unsqueeze(1)
    # opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)

    # hom = torch.ones(b, c, 1, 9).cuda()
    # kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)
    # kps = torch.bmm(opinv, kps).view(b, c, 2, 9)
    kps = kps.view(b, c, 2, 8)

    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
    alpna_pre = alpna_pre.unsqueeze(2)
    # alpna_pre=rot_gt

    # rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y = alpna_pre + torch.atan2(x - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()

    # kpoint = kps[:, :, :16]
    kpoint = kps

    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)
    cxy = cxy.repeat(1, 1, 8)  # b,c,16
    kp_norm = (kpoint - cxy) / f

    l = dim[:, :, 2:3]
    h = dim[:, :, 0:1]
    w = dim[:, :, 1:2]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)

    B = torch.zeros_like(kpoint)
    C = torch.zeros_like(kpoint)

    kp = kp_norm.unsqueeze(3)  # b,c,16,1
    const = const.expand(b, c, -1, -1)
    A = torch.cat([const, kp], dim=3)

    B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 1:2] = h * 0.5
    B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 3:4] = h * 0.5
    B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 5:6] = h * 0.5
    B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 7:8] = h * 0.5
    B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 9:10] = -h * 0.5
    B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 11:12] = -h * 0.5
    B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 13:14] = -h * 0.5
    B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 15:16] = -h * 0.5

    C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

    B = B - kp_norm * C

    # A=A*kps_mask1

    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 16)
    A = A.view(b * c, 16, 3)
    B = B.view(b * c, 16, 1).float()
    # mask = mask.unsqueeze(2)

    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv)  # b*c 3 3



    pinv = torch.bmm(pinv, AT)
    pinv = torch.bmm(pinv, B)
    pinv = pinv.view(b, c, 3, 1).squeeze(3)

    #pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2
    return pinv,rot_y,kps


if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    const = torch.Tensor([[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
         [-1, 0], [0, -1], [-1, 0], [0, -1]])
    const = const.unsqueeze(0).unsqueeze(0)
    const = const.to('cpu')

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs, metadatas) in enumerate(test_dataloader):
            batch_size = imgs.size(0)
            input_imgs = imgs.to(device=configs.device).float()
            t1 = time_synchronized()
            outputs = model(input_imgs)
            t2 = time_synchronized()
            outputs['hm_mc'] = _sigmoid(outputs['hm_mc'])
            outputs['hm_ver'] = _sigmoid(outputs['hm_ver'])
            outputs['depth'] = 1. / (_sigmoid(outputs['depth']) + 1e-9) - 1.
            # detections size (batch_size, K, 38)
            detections = rtm3d_decode(outputs['hm_mc'], outputs['hm_ver'], outputs['vercoor'], outputs['cenoff'],
                                      outputs['veroff'], outputs['wh'], outputs['rot'], outputs['depth'],
                                      outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy()
            detections = post_processing_2d(detections, configs.num_classes, configs.down_ratio)

            # position, rot_y, kps_inv = gen_position(kps, dim, rot, meta, const)

            detections = get_final_pred(detections[0], configs.num_classes, configs.peak_thresh)
            P2 = metadatas['calib'].squeeze().numpy()
            pad_size = metadatas['pad_size'].numpy().flatten()
            # print(pad_size)
            print(metadatas['id'])
            output = []

            for j in range(configs.num_classes):
                if len(detections[j] > 0):
                    for det in detections[j]:
                        # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:26, depth-26:27, dim-27:30)
                        _score = det[0]
                        _x, _y, _wh, _bbox, _ver_coor = int(det[1]), int(det[2]), det[3:5], det[5:9], det[9:25]
                        _rot, _depth, _dim, rots = det[25], det[26], det[36:39], det[26:34]
                        # print(_dim)
                        # print(_rot)
                        bbox = _bbox
                        _bbox = np.array(_bbox, dtype=np.int)
                        # print(_ver_coor)
                        # print(_bbox)
                        # position, rot_y, kps_inv = gen_position(torch.tensor(_ver_coor).unsqueeze(0).unsqueeze(0), torch.tensor(_dim).unsqueeze(0).unsqueeze(0)
                        #                                         , torch.tensor(rots).unsqueeze(0).unsqueeze(0), metadatas, const, torch.tensor(_x).unsqueeze(0).unsqueeze(0))
                        # print(position)
                        # print(rot_y)
                        # print(kps_inv)

                        pred_vertices = _ver_coor.copy()
                        pred_vertices = pred_vertices.reshape((8, 2))
                        pred_vertices[:, 0] -= pad_size[0]
                        pred_vertices[:, 1] -= pad_size[1]
                        pred_vertices = pred_vertices.flatten()

                        bbox[[0, 2]] -= pad_size[0]
                        bbox[[1, 3]] -= pad_size[1]

                        new_diff_fun = lambda a: diff_fun(a, pred_vertices, P2, _depth, _dim, _rot)
                        lower = np.array([-100, -100, -100, -100, -100, _depth - 2, -4])
                        upper = np.array([100, 100, 100, 100, 100, _depth + 2, 4])
                        x0 = np.array([1.42, 1.58, 3.48, 3.87, 1.64, _depth, -1.83])

                        res_1 = least_squares(new_diff_fun, x0, bounds=(lower, upper))
                        # vertices, corners_3D, _ = np_computeBox3D(x0, P2)
                        # vertices = vertices.transpose().flatten()

                        # print(diff_fun(x0, pred_vertices, P2, _depth, x0[0:3], x0[6]))

                        # print(vertices)
                        # print(pred_vertices)

                        value_3d = res_1.x
                        # print(_rot)
                        # print(_dim)
                        # print(_depth)
                        print(value_3d)
                        print(res_1.cost)
                        # ver, _, _ = np_computeBox3D(value_3d, P2)
                        # print(ver)
                        if j == 0:
                            cls = 'Pedestrian'
                        elif j == 1:
                            cls = 'Car'
                        else:
                            cls = 'Cyclist'
                        output_list = [cls, 0.00, 0, 0.00]
                        output_list += bbox.tolist()
                        output_list += value_3d.tolist()
                        output_list.append(_score.item())
                        print(output_list)
                        output_list = ' '.join([str(x) for x in output_list])
                        output_list += '\n'
                        output.append(output_list)
            img_fn = os.path.basename(img_paths[0])[:-4]
            output_file_name = os.path.join(configs.results_dir, '{}.txt'.format(img_fn))
            with open(output_file_name, 'w') as fp:
                fp.writelines(output)

            img = imgs.squeeze().permute(1, 2, 0).numpy()
            img = denormalize_img(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))

            # Draw prediction in the image
            out_img = draw_predictions(img, detections, cnf.colors, configs.num_classes)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(img_paths[0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            if configs.show_image:
                cv2.imshow('test-img', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
