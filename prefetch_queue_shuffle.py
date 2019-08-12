# This code is revised from voxel flow by Ziwei Liu https://github.com/liuziwei7/voxel-flow

from __future__ import print_function
import glob
import numpy as np
import os
import Queue
import random
import threading
import cv2
import pdb

data_dir = './DRV/'

def load_fn_example(train_id):
    gt_path = glob.glob(data_dir + '/long/%s/half0001*.png'%train_id)[0]
    #pdb.set_trace()
    im = cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
    gt_im = np.expand_dims(np.float32(im/65535.0),axis = 0)
    in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/%s/*.png'%train_id))
    #choose two random frames from the same video
    ind_seq = np.random.random_integers(0,len(in_files)-2)
    in_path = in_files[ind_seq]
    im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
    in_im1 = np.expand_dims(np.float32(im/65535.0),axis = 0)
    ind_seq2 = np.random.random_integers(0,len(in_files)-2)
    if ind_seq2 == ind_seq:
        ind_seq2 += 1
    in_path = in_files[ind_seq2]
    im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
    in_im2 = np.expand_dims(np.float32(im/65535.0),axis = 0)
    in_im = np.concatenate([in_im1,in_im2],axis=0)
    return (in_im, gt_im)

class DummpyData(object):
    def __init__(self, data):
        self.data = data
    def __cmp__(self, other):
        return 0

def prefetch_job(load_fn, prefetch_queue, data_list, shuffle, prefetch_size):
    self_data_list = np.copy(data_list)
    data_count = 0
    total_count = len(self_data_list)
    idx = 0
    while True:
        if shuffle:
            if data_count == 0:
                random.shuffle(self_data_list)

            data = load_fn(self_data_list[data_count]) #Load your data here.

            idx = random.randint(0, prefetch_size)
            dummy_data = DummpyData(data)

            prefetch_queue.put((idx, dummy_data), block=True)

        data_count = (data_count + 1) % total_count

class PrefetchQueue(object):
    def __init__(self, load_fn, data_list, batch_size=32, prefetch_size=None, shuffle=True, num_workers=4):
        self.data_list = data_list
        self.shuffle = shuffle
        self.prefetch_size = prefetch_size
        self.load_fn = load_fn
        self.batch_size = batch_size
        if prefetch_size is None:
            self.prefetch_size = 4 * batch_size

        # Start prefetching thread
        # self.prefetch_queue = Queue.Queue(maxsize=prefetch_size)
        self.prefetch_queue = Queue.PriorityQueue(maxsize=prefetch_size)
        for k in range(num_workers):
            t = threading.Thread(target=prefetch_job,
            args=(self.load_fn, self.prefetch_queue, self.data_list,
                  self.shuffle, self.prefetch_size))
            t.daemon = True
            t.start()

    def get_batch(self):
        data_list = []
        #for k in range(0, self.batch_size):
          # if self.prefetch_queue.empty():
          #   print('Prefetch Queue is empty, waiting for data to be read.')
        _, data_dummy = self.prefetch_queue.get(block=True)
        data = data_dummy.data
          #data_list.append(data)
        return data
