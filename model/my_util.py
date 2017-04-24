# -*- coding: utf-8 -*-
import os
import shutil
import imageio
import glob
import datetime
import numpy as np
import scipy.misc as misc
from PIL import Image
import pickle

from os.path import expanduser


def check_dir(dir_path, clear_before=True, msg=""):
    try:
        if msg == "":
            msg = os.path.split(dir_path)[1]
        drive_letter = os.path.splitdrive(os.getcwd())[0]
        if drive_letter == "":
            drive_letter = expanduser("~")
        if not str.startswith(dir_path, drive_letter):
            dir_path = check_dir_disk(dir_path)
        if clear_before:
            if os.path.exists(dir_path):
                print("removing existing from Dir {0} : {0}".format(msg, dir_path))
                shutil.rmtree(dir_path)
        if not os.path.exists(dir_path):
            parent_dir = os.path.split(dir_path)[0]
            while not os.path.exists(parent_dir):
                check_dir(parent_dir)
            print("create Dir {0} : {0}".format(msg, dir_path))
            os.mkdir(dir_path)
    except Exception as e:
        print("Create the Dir failed, {0}: {0}".format(msg, dir_path))
        print(e)
        raise e
    return dir_path


def check_dir_disk(dir_path):
    try:
        drive_letter = os.path.splitdrive(os.getcwd())[0]
        if drive_letter == "":
            drive_letter = expanduser("~")
            dir_path = os.path.join(drive_letter, dir_path)
        else:
            d, dir_path = os.path.splitdrive(dir_path)
            dir_path = os.path.join(drive_letter, "\\", dir_path)
    except Exception as e:
        print("Check the Dir failed, {0}".format(dir_path))
        print(e)
        raise e
    return dir_path


def del_file_exist(file_path):
    try:
        drive_letter = os.path.splitdrive(os.getcwd())[0]
        if not str.startswith(file_path, drive_letter):
            file_path = check_dir_disk(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(e)
        raise e
    return file_path


def preproc(unclean_batch, unit_scale=False):
    """Convert values to range 0-1"""
    if unit_scale:
        unclean_batch = to_scale_binary(unclean_batch)
    if unclean_batch.max() > 1:
        temp_batch = unclean_batch / unclean_batch.max()
        return temp_batch
    else:
        return unclean_batch


def to_scale_binary(batch):
    ''' # scale 0-1 matrix back to gray scale bitmaps '''
    if len(batch.shape) == 3:
        num = batch.shape[0]
    else:
        num = 1

    # c = np.count_nonzero(batch[np.where(batch < 1)])
    # print(batch[np.where(np.logical_and(batch < 1, batch > 0))].any())
    if ((batch < 1) & (batch > 0)).any():
        batch = ((batch * 255.).astype(dtype=np.int16) % 256)
    if num > 1:
        for idx, bm in enumerate(batch):
            bm = to_binary(bm)
    else:
        batch = to_binary(batch)
    return batch


def to_binary(bm):
    # if bm[np.where(bm < 255)].max() > 1:
    if ((bm < 255) & (bm > 1)).any():
        rows, cols = bm.shape
        for h in range(rows):
            for w in range(cols):
                bm[h, w] = 0 if bm[h, w] <= 127 else 255
    return bm


def to_binary_by_threshold(bm, threshold=127):
    """
    Borrowed and modified from: http://blog.csdn.net/rickarkin/article/details/1919274
    TODO：not work for train result.
    """
    if bm[np.where(bm < 255)].max() > 1:
        # setup a converting table with constant threshold
        table = []
        for i in range(256):
            if i <= threshold:
                table.append(0)
            else:
                table.append(1)
        bm = bm.point(table, '1')
    return bm


# Save the sample file
def save_img1(bm, path):
    img = Image.fromarray(bm)
    img.save(path)


# Save the sample file
def save_img2(bm, path):
    misc.imsave(path, bm)
    return path


# Save the sample file
def save_img(bm, path):
    h, w = bm.shape
    canvas = np.zeros(shape=(h, w), dtype=np.uint16)
    # make the canvas all white
    canvas.fill(255)
    canvas[0:h, 0:w] = bm
    misc.toimage(canvas).save(path)
    return path


# Save the sample file
def save_sample_image(x, path, unit_scale=True, adj=1):
    num_imgs, h, w = x.shape
    # img_per_row = int(np.ceil(float(np.sqrt(num_imgs))))
    img_per_row = int(np.round(float(np.sqrt(num_imgs))))
    if img_per_row > 5:
        img_per_row = 5

    num_imgs = img_per_row ** 2
    if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
        bitmaps = ((x * 255.).astype(dtype=np.int16) % 256)[:num_imgs]
    else:
        bitmaps = x[:num_imgs]

    width = img_per_row * w
    height = int(np.ceil(float(num_imgs) / img_per_row)) * h
    canvas = np.zeros(shape=(height, width), dtype=np.int16)
    # make the canvas all white
    canvas.fill(255)
    for idx, bm in enumerate(bitmaps):
        x = h * int(idx / img_per_row)
        y = w * int(idx % img_per_row)
        if adj == 1:
            bm = to_binary(bm)
        elif adj == 2:
            bm = to_binary_by_threshold(bm)
        canvas[x: x + h, y: y + w] = bm
    misc.toimage(canvas).save(path)
    return path


def render_frame(x, frame_dir, step, sub_step=0):
    # add save to npy file for check.
    if step % 3 == 0 and sub_step == 0:
        npy_file_name = os.path.join(frame_dir, "step_{0:04d}_{1:03d}.dat".format(step, sub_step))
        with open(npy_file_name, 'wb', True) as f:
            pickle.dump((x), f, protocol=pickle.HIGHEST_PROTOCOL)
    frame_path = os.path.join(frame_dir, "step_{0:04d}_{1:03d}.png".format(step, sub_step))
    if step == 0:
        frame_path = os.path.join(frame_dir, "step_{0:04d}_{1:03d}.jpg".format(step, sub_step))

        frame_path_0 = os.path.join(frame_dir, "step_{0:4d}_0.jpg".format(step))
        save_sample_image(x, frame_path_0, False, 0)
    return save_sample_image(x, frame_path)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file


def check_msg(message):
    if message[-1:2] != ": ":
        message += " : "
    return message


def validate_array(pred_y, y, check_zero=False):
    num, w, h = y.shape
    batch_pred_y = preproc(to_scale_binary(pred_y)).reshape(-1, w * h)
    batch_y = preproc(to_scale_binary(y)).reshape(-1, w * h)
    # batch_pred_y = pred_y.reshape(-1, w * h)
    # batch_y = y.reshape(-1, w * h)
    accuracy = 0
    for idx, bm_x in enumerate(batch_pred_y):
        bm_y = batch_y[idx]
        accuracy += validate(bm_x, bm_y, check_zero) / num
    return accuracy


def validate(pred_y, y, check_zero=False):
    try:
        total_num = len(y)
        if check_zero:
            find_mask = np.nonzero(y == 0)
            pred_y = pred_y[find_mask]
            y = y[find_mask]
            total_num = len(y)

        correct_pred = np.sum(np.equal(pred_y, y))
        # print("correct_pred:{0}, total_num:{1}".format(correct_pred, total_num))
        accuracy = float(correct_pred) / total_num
        return accuracy
    except Exception as e:
        print(e)


def dice_similarity_array(pred_y, y, check_zero=False):
    num, w, h = y.shape
    batch_pred_y = preproc(to_scale_binary(pred_y)).reshape(-1, w * h)
    batch_y = preproc(to_scale_binary(y)).reshape(-1, w * h)
    # batch_pred_y = pred_y.reshape(-1, w * h)
    # batch_y = y.reshape(-1, w * h)
    accuracy = 0
    for idx, bm_x in enumerate(batch_pred_y):
        bm_y = batch_y[idx]
        accuracy += dice_similarity(bm_x, bm_y, check_zero) / num
    return accuracy


def dice_similarity(pred_y, y, check_zero=False):
    try:
        if check_zero:
            find_mask = np.nonzero(y == 0)
            pred_y = pred_y[find_mask]
            y = y[find_mask]
        # print(np.sum(pred_y))
        # print(np.sum(y))
        dice = np.sum(np.equal(pred_y, y)) * 2.0 / (np.sum(pred_y) + np.sum(y))
        return dice
    except Exception as e:
        print(e)


def get_batch(x, batch_size, curr_batch):
    start_index = batch_size * curr_batch
    if start_index >= x.shape[0]:
        start_index = 0

    end_index = start_index + batch_size
    if end_index >= x.shape[0]:
        end_index = x.shape[0]

    return x[start_index:end_index]


'''
stopwatch
'''


class StopWatch(object):
    """A simple timer class"""

    def __init__(self, logfile_folder):
        # # if logfile_folder == "":
        # #     logfile_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log")
        # # if not logfile_folder.endswith("log"):
        # #     logfile_folder = os.path.join(logfile_folder, "log")
        # if logfile_folder == "":
        #     logfile_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)))

        # logfile_folder = check_dir(logfile_folder, clear_before=False)
        # LOG_FILENAME = os.path.join(logfile_folder, 'mylog.log')
        # print("LogFileName:" + LOG_FILENAME)
        # self.my_logger = logging.getLogger('MyLogger')
        # self.my_logger.setLevel(logging.DEBUG)
        # # Add the log message handler to the logger
        # handler = logging.handlers.RotatingFileHandler(
        #     LOG_FILENAME, maxBytes=3 * 1024 * 1024, backupCount=10)

        # self.my_logger.addHandler(handler)
        self.my_logger = logging.getLogger('MyLogger')
        pass

    def start(self):
        """Starts the timer"""
        self.start = datetime.datetime.now()
        return self.start

    def stop(self, message="Total: "):
        """Stops the timer.  Returns the time elapsed"""
        self.stop = datetime.datetime.now()
        msg = check_msg(message) + str(self.stop - self.start)
        self.my_logger.info(msg)
        return msg

    def now(self, message="Now: "):
        """Returns the current time with a message"""
        msg = check_msg(message) + "[" + str(datetime.datetime.now()) + "]"
        self.my_logger.info(msg)
        return msg

    def elapsed(self, message="Elapsed: "):
        """Time elapsed since start was called"""
        msg = check_msg(message) + str(datetime.datetime.now() - self.start) + "[" + str(datetime.datetime.now()) + "]"
        self.my_logger.info(msg)
        return msg

    def split(self, message="Split started at: "):
        """Start a split timer"""
        self.split_start = datetime.datetime.now()
        return message + str(self.split_start)

    def unsplit(self, message="Unsplit: "):
        """Stops a split. Returns the time elapsed since split was called"""
        return message + str(datetime.datetime.now() - self.split_start)


'''
my_log
'''

import logging
import logging.handlers
import os


class my_log(object):
    """A Log4Python class"""

    def __init__(self, logfile_folder=""):
        # if logfile_folder == "":
        #     logfile_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log")
        # if not logfile_folder.endswith("log"):
        #     logfile_folder = os.path.join(logfile_folder, "log")
        if logfile_folder == "":
            logfile_folder = check_dir_disk("TMP")

        logfile_folder = check_dir(logfile_folder, clear_before=False)
        LOG_FILENAME = os.path.join(logfile_folder, 'mylog.log')
        print("LogFileName:" + LOG_FILENAME)
        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILENAME, maxBytes=3 * 1024 * 1024, backupCount=10)
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)

        self.my_logger = logging.getLogger('MyLogger')
        self.my_logger.setLevel(logging.DEBUG)
        self.my_logger.addHandler(handler)
        self.my_logger.info('creating an instance of MyLogger')
        pass

    def info(self, message):
        self.my_logger.info(message)
