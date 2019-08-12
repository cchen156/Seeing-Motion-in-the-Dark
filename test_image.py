from __future__ import division
import os,time
import numpy as np
import pdb
import glob
from network import *
import cv2
from prefetch_queue_shuffle import *
import tensorflow as tf
from vgg import *
import math


data_dir = './DRV/'

train_ids = [line.rstrip('\n') for line in open(data_dir+'train_list.txt')]
val_ids = [line.rstrip('\n') for line in open(data_dir+'val_list.txt')]
test_ids = [line.rstrip('\n') for line in open(data_dir+'test_list.txt')]


in_image=tf.placeholder(tf.float32,[None,None,None,3])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])

out_image=Unet(in_image)

sess=tf.Session()
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("checkpoint")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if not os.path.isdir("result/final"):
    os.makedirs("result/final")

for test_id in test_ids:
    gt_files = glob.glob(data_dir + '/long/%s/half0001*.png'%test_id)
    gt_path = gt_files[0]
    im = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    gt_np = np.float32(im/65535.0)

    in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/%s/*.png'%test_id))

    # run the 5-th frame of each video. Can be revised to run more frames.
    for k in range(4,len(in_files)-3, 500):
        print('running %s-th sequence %d-th frame...'%(test_id,k))
        in_path = in_files[k]
        im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        in_np = np.expand_dims(np.float32(im/65535.0),axis = 0)

        out_np =sess.run(out_image,feed_dict={in_image: in_np})

        cv2.imwrite("result/final/%s_out.png"%(test_id),np.uint8(np.clip(out_np[0,:,:,:]*255,0,255)))
        cv2.imwrite("result/final/%s_gt.png"%(test_id),np.uint8(np.clip(gt_np*255,0,255)))
