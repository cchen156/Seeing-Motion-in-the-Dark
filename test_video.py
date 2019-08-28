from __future__ import division
import os,time
import numpy as np
import pdb
import rawpy
import glob
from network import *
import cv2


data_dir = './DRV/'

train_ids = [line.rstrip('\n') for line in open(data_dir+'train_list.txt')]
val_ids = [line.rstrip('\n') for line in open(data_dir+'val_list.txt')]
test_ids = [line.rstrip('\n') for line in open(data_dir+'test_list.txt')]

sess=tf.Session()

in_image=tf.placeholder(tf.float32,[None,None,None,3])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])

out_image=Unet(in_image)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("checkpoint")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if not os.path.isdir("result/video/"):
    os.makedirs("result/video/")
if not os.path.isdir("result/frames/"):
    os.makedirs("result/frames/")

# test static videos
for test_id in test_ids:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter('result/video/%s.avi'%test_id,fourcc, 20.0, (1374,918))
    if not os.path.isdir("result/frames/%s"%test_id):
        os.makedirs("result/frames/%s"%test_id)

    in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/%s/*.png'%test_id))
    for k in range(4,len(in_files)-3):
        print('running %s-th sequence %d-th frame...'%(test_id,k))
        in_path = in_files[k]
        _, in_fn = os.path.split(in_path)

        im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        in_np = np.expand_dims(np.float32(im/65535.0),axis = 0)

        out_np =sess.run(out_image,feed_dict={in_image: in_np})
        out_np = np.minimum(np.maximum(out_np,0),1)

        out = np.uint8(out_np[0,:,:,:]*255.0)
        cv2.imwrite("result/frames/%s/%04d.png"%(test_id,k+1),out)
        video.write(out)
    video.release()

# test dynamic videos
for test_id in range(23):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter('result/video/M%04d.avi'%test_id,fourcc, 20.0, (1374,918))

    in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/M%04d/*.png'%test_id))
    if not os.path.isdir("result/frames/M%04d"%test_id):
        os.makedirs("result/frames/M%04d"%test_id)
    for k in range(4,len(in_files)-3):

        print('running %s-th sequence %d-th frame...'%(test_id,k))
        in_path = in_files[k]
        _, in_fn = os.path.split(in_path)

        im = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        in_np = np.expand_dims(np.float32(im/65535.0),axis = 0)

        out_np =sess.run(out_image,feed_dict={in_image: in_np})
        out_np = np.minimum(np.maximum(out_np,0),1)

        out = np.uint8(out_np[0,:,:,:]*255.0)
        cv2.imwrite("result/frames/M%04d/%04d.png"%(test_id,k+1),out)
        video.write(out)
    video.release()
