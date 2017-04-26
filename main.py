# -*- coding: utf-8 -*-

import os
import subprocess
import model.my_util as my_util
import pickle
import numpy as np

if __name__ == "__main__":
    os.system("python train.py --debug=0")
    # os.system("python train.py --debug=0 --epoch=300")
    subprocess.call(["shutdown", "/f", "/s", "/t", "20"])
    # root_path = my_util.check_dir_disk("TMP")
    # data_dir = os.path.join(root_path, "zi4zi", "epoch_10", "sample")
    # with open(os.path.join(data_dir, "fake_img_00_0004.dat"), 'rb') as f:
    #     real_imgs, fake_imgs = pickle.load(f)
    #
    # num = real_imgs.shape[0]
    # accuracy = 0
    # accuracy_all = 0
    # manhattan_norm = 0
    # zero_norm = 0
    # for idx, bm_pred in enumerate(fake_imgs):
    #     pred_y = ((bm_pred * 255.).astype(dtype=np.int16) % 256).reshape(-1, 256 * 256 * 3)[0]
    #     pred_y = my_util.preproc(my_util.to_binary(pred_y))
    #     y = real_imgs[idx].reshape(-1, 256 * 256 * 3)[0]
    #     acc_all, acc = my_util.validate_accuracy(pred_y, y)
    #     print("acc_all:{0:5f}, acc{1:5f}".format(acc_all, acc))
    #     accuracy_all += acc_all
    #     accuracy += acc
    #     n_m, n_0 =my_util. compare_images(pred_y, y)
    #     # print("Manhattan norm:{0} / per pixel:{1}".format(n_m, n_m / bm_x.size))
    #     # print("Zero norm:{0} / per pixel:{1}".format(n_0, n_0 * 1.0 / bm_x.size))
    #     manhattan_norm += n_m / bm_pred.size
    #     zero_norm += n_0 * 1.0 / bm_pred.size
    #
    # print("acc_all:{0:5f}, acc:{1:5f}".format(accuracy_all / num, accuracy / num))
    # print("manhattan_norm:{0:5f}, zero_norm:{1:5f}".format(manhattan_norm / num, zero_norm / num))
    # print("OK")
