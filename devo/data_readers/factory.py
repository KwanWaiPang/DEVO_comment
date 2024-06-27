import pickle
import os
import os.path as osp

from .tartan import TartanAir, TartanAirEVS, TartanAirE2VID

# 读取数据集，kwargs是关键字参数，可以是任意个数的关键字参数，本质上就是传入一个dict参数
# 也就是下面的全部，将他们一起打包到db_list中然后返回
# datapath=args.datapath,
# n_frames=args.n_frames,
# fgraph_pickle=args.fgraph_pickle, 
# train_split=args.train_split,
# val_split=args.val_split, 
# strict_split=False, 
# sample=True, 
# return_fname=True, 
# scale=args.scale
def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = { 
        'tartan': (TartanAir, ),
        'tartan_evs': (TartanAirEVS, ),
        'tartan_e2vid':  (TartanAirE2VID, ),
    }
    
    if not all(x in dataset_map for x in dataset_list):
        ValueError("dataset_list {dataset_list} should be a subset of {dataset_map}.")
    
    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
