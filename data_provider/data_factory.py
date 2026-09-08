from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS, Dataset_Solar
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'illness': Dataset_Custom,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'solar': Dataset_Solar,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        supported = ', '.join(sorted(data_dict))
        raise ValueError(f'Unsupported dataset {args.data!r}. Available datasets: {supported}')
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if args.task_name != 'long_term_forecast':
        raise ValueError('This release only supports long_term_forecast.')

    if flag == 'test':
        shuffle_flag = False
        batch_size = 1
    else:
        shuffle_flag = True
        batch_size = args.batch_size
    # Keep every validation/test window. Dropping the final training batch is
    # conventional and avoids a smaller batch changing the optimization step.
    drop_last = flag == 'train'
    freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
