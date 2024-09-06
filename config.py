import os
import argparse

parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--backbone', type=str, default='R50',
                    choices=['twins', 'segformer', 'res2net', 'shunted', 'pvtv2', 'pvtv2b2', 'swin', 'R50', 'dpt',
                             'replk', 'p2t'])
parser.add_argument('--decoder', type=str, default='base_decoder', choices=['base_decoder'])
parser.add_argument('--log_info', type=str, default='final')
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()

param = {}
param['device'] = 0
param['second_time'] = False
param['epoch'] = 100
param['batch_size'] = 8
param['lr_config'] = {'beta': [0.5, 0.999], 'lr': 1e-4, 'lr_dis': 1e-5,
                      'decay_rate': 0.75, 'decay_epoch': 20, 'gamma': 0.98}
param['save_epoch'] = 10
param['validate_epoch'] = 10
param['trainsize'] = 384
param['neck_channel'] = 32
param['num_workers'] = 0
param['backbone'] = args.backbone
param['decoder'] = args.decoder
param['seed'] = 1024
pretrain_dict = {"swin": "pth/swin_base_patch4_window12_384.pth",
                 "replk": "pth/RepLKNet-31B_ImageNet-1K_224.pth",
                 "pvtv2": "pth/pvt_v2_b4.pth",
                 "pvtv2b2": "pth/pvt_v2_b2.pth",
                 "p2t": "pth/p2t_large.pth",
                 "shunted": "pth/shunted_B.pth",
                 "res2net": "pth/res2net50_v1b_26w_4s.pth",
                 "segformer": "pth/segformer_b5.pth",
                 "twins": "pth/alt_gvt_large.pth",
                 "R50": None}
param['pretrain'] = pretrain_dict[param['backbone']]

param['training_info'] = str(param['lr_config']['lr']) + '_' + '{}_{}'.format(param['backbone'], param['decoder']) + '_' + args.log_info
param['log_path'] = 'experiments/' + param['training_info'] + '/'
param['ckpt_save_path'] = param['log_path'] + '/models/'

param['paths'] = {'image_root': '../data/Train/Image/',
                  'gt_root': '../data/Train/Scribble/',
                  'test_root': '../data/Test/'}


if args.ckpt is not None:
    model_path = param['log_path'] + 'models/'
    if args.ckpt.lower() == 'last':
        model_list = os.listdir(model_path)
        model_list.sort(key=lambda x: int(x.split('_')[0]))
        param['checkpoint'] = model_path + model_list[-1]
    else:
        param['checkpoint'] = model_path + args.ckpt
else:
    param['checkpoint'] = None

param['eval_save_path'] = param['log_path'] + 'save_images/'
if param['second_time']:
    param['paths']['gt_root'] = param['eval_save_path']+'soft_gt/'
param['datasets'] = ['CAMO', 'COD10K','NC4K']  # 'CHAMELEON',
