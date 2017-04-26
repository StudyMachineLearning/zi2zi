# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import argparse
import os
import model.my_util as my_util
import datetime

args = None

from model.unet import UNet


def main(_):
    output_folder_root = os.path.join(args.experiment_dir, "epoch_{0}_{1}".format(args.epoch,
                                                                                  datetime.datetime.now().strftime(
                                                                                      '%m%d%H')))

    if args.debug == 1:
        output_folder_root = os.path.join(args.experiment_dir, "epoch_{0}".format(args.epoch))

    output_folder_root = my_util.check_dir(output_folder_root, clear_before=True)

    mylog = my_util.my_log(output_folder_root)
    mylog.info("args: {0}".format(args))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(args.experiment_dir, output_folder_root, batch_size=args.batch_size,
                     experiment_id=args.experiment_id,
                     input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                     embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                     Ltv_penalty=args.Ltv_penalty)
        model.register_session(sess)
        model.build_model(is_training=True, inst_norm=args.inst_norm)
        fine_tune_list = None
        if args.fine_tune:
            ids = args.fine_tune.split(",")
            fine_tune_list = set([int(i) for i in ids])
        model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                    sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--debug', type=int, default='1',
                        help='for indicate if debug')
    parser.add_argument('--experiment_dir', dest='experiment_dir',
                        help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                        help='sequence id for the experiments you prepare to run')
    parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                        help="size of your input and output image")
    parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
    parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
    parser.add_argument('--embedding_num', dest='embedding_num', type=int, default=40,
                        help="number for distinct embeddings")
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
    parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='number of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--schedule', dest='schedule', type=int, default=10,
                        help='number of epochs to half learning rate')
    parser.add_argument('--resume', dest='resume', type=int, default=0, help='resume from previous training')
    parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                        help="freeze encoder weights during training")
    parser.add_argument('--fine_tune', dest='fine_tune', type=str, default=None,
                        help='specific labels id to be fine tuned')
    parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                        help='use conditional instance normalization in your model')
    parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=50,
                        help='number of batches in between two samples are drawn from validation set')
    parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                        help='number of batches in between two checkpoints')
    args = parser.parse_args()
    try:
        if args.experiment_dir is None or args.experiment_dir == "":
            args.experiment_dir = my_util.check_dir_disk(os.path.join("TMP", "zi4zi"))

        if args.debug == 1:
            args.epoch = 10
            args.sample_steps = 2
        tf.app.run()
    except Exception as e:
        print("initial validation failed")
        print(e)
        raise e
