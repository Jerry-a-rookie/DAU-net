import argparse
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def str2bool(value):
    """Parse a human-readable boolean from the command line."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(
        "expected a boolean value: True/False, 1/0, yes/no, or on/off"
    )


def main():
    parser = argparse.ArgumentParser(description='DAU-Net long-term forecasting')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='supported task: long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, default='ETTh1_96_96', help='experiment id')
    parser.add_argument('--model', type=str, default='DAU-Net',
                        help='model name: DAU-Net (legacy aliases: SC-Net, SCUNet)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: M, S, MS')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='time feature frequency')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint directory')
    parser.add_argument('--output_path', type=str, default='./outputs/',
                        help='directory for predictions and metrics')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='decoder start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='kept for loader compatibility')

    # model definition
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=66, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='kept for setting names')
    parser.add_argument('--e_layers', type=int, default=2, help='number of DAU-Net levels')
    parser.add_argument('--d_layers', type=int, default=1, help='kept for setting names')
    parser.add_argument('--d_ff', type=int, default=64, help='hidden dimension')
    parser.add_argument('--factor', type=int, default=1, help='kept for setting names')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time feature encoding')
    parser.add_argument('--distil', action='store_false', default=True, help='kept for setting names')
    parser.add_argument('--activation', type=str, default='gelu', help='kept for compatibility')
    parser.add_argument('--output_attention', action='store_true', help='kept for compatibility')

    # DAU-Net
    parser.add_argument('--mscd_K', type=int, default=3, help='MSCD component count')
    parser.add_argument('--sc2r_M_avg', type=int, default=2, help='SC2R average edges per variable')
    parser.add_argument('--sc2r_K_edges', type=int, default=None, help='SC2R absolute edge count')
    parser.add_argument('--sc2r_d_h', type=int, default=64, help='SC2R score network hidden dimension')
    parser.add_argument('--gate_reduction', type=int, default=4, help='spectral gate reduction ratio')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader workers (0 is the most portable default)')
    parser.add_argument('--itr', type=int, default=1, help='experiment repeats')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0015, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='kept for compatibility')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate schedule')
    parser.add_argument('--use_amp', action='store_true', default=False, help='automatic mixed precision')

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True,
                        help='use GPU when available (True/False)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-GPU device ids')

    args = parser.parse_args()

    if args.model not in {'DAU-Net', 'SC-Net', 'SCUNet'}:
        raise ValueError('This release only supports --model DAU-Net (legacy aliases: SC-Net, SCUNet).')
    if args.task_name != 'long_term_forecast':
        raise ValueError('This release only supports --task_name long_term_forecast.')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    exp_cls = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
                args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
                args.distil, args.des, ii)

            exp = exp_cls(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.distil, args.des, ii)

        exp = exp_cls(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
