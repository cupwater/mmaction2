'''
Author: Peng Bo
Date: 2022-09-27 08:36:25
LastEditTime: 2022-09-27 09:05:03
Description: 

'''

import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='split a video into several snippets')
    parser.add_argument('--input', type=str,
                        help='The source root of input videos')
    parser.add_argument('-r', '--ratio', type=float,
                        default=0.7, help='The step of sampler')
    args = parser.parse_args()

    label_idx = 1
    file_label_list = []
    for f in os.listdir(args.input):
        sub_root = os.path.join(args.input, f)
        if os.path.isdir(sub_root):
            for sub_f in os.listdir(sub_root):
                if 'mp4' in sub_f:
                    file_label_list.append((os.path.join(f, sub_f), label_idx))
            label_idx += 1
    
    # split into train and val subset
    idxs = list(range(len(file_label_list)))
    train_idxs = np.random.choice( idxs, int(args.ratio*len(idxs)), replace=False )
    train_idxs = np.array(sorted(train_idxs), dtype=int)
    val_idxs   = np.array(sorted(list(set(idxs) - set(train_idxs))), dtype=int)
    train_list = [ file_label_list[idx] for idx in train_idxs.tolist()]
    val_list   = [ file_label_list[idx] for idx in val_idxs.tolist()]

    # write into file
    with open(os.path.join(args.input, 'train_list.txt'), 'w') as fout:
        res_str = [ v[0]+' '+str(v[1]) for v in train_list]
        fout.write("\n".join(res_str))
    with open(os.path.join(args.input, 'val_list.txt'), 'w') as fout:
        res_str = [ v[0]+' '+str(v[1]) for v in val_list]
        fout.write("\n".join(res_str))
