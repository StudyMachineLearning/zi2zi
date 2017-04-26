# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
# import cPickle as pickle
import pickle
import random
import model.my_util


def pickle_examples(paths, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p in paths:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    print("img %s" % p, label)
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
    parser.add_argument('--dir', dest='dir', help='path of examples')
    parser.add_argument('--save_dir', dest='save_dir', help='path to save pickled files')
    parser.add_argument('--split_ratio', type=float, default=0.1, dest='split_ratio',
                        help='split ratio between train and val')
    args = parser.parse_args()
    try:
        root_path = model.my_util.check_dir_disk(os.path.join("TMP", "zi4zi"))
        if args.dir is None or args.dir == "":
            args.dir = os.path.join(root_path, "output_pic")

        if args.save_dir is None or args.save_dir == "":
            args.save_dir = os.path.join(root_path, "data")

        output_path = model.my_util.check_dir(args.save_dir)
        train_path = os.path.join(args.save_dir, "train.obj")
        val_path = os.path.join(args.save_dir, "val.obj")
        pickle_examples(glob.glob(os.path.join(args.dir, "*.jpg")), train_path=train_path, val_path=val_path,
                        train_val_split=args.split_ratio)

        print("OK..")
    except Exception as e:
        print("initial validation failed")
        print(e)
        raise e
