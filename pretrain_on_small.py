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

data_dir = './DRV/'

train_ids = [line.rstrip('\n') for line in open(data_dir+'train_list.txt')]
val_ids = [line.rstrip('\n') for line in open(data_dir+'val_list.txt')]
test_ids = [line.rstrip('\n') for line in open(data_dir+'test_list.txt')]

save_freq = 250

in_image=tf.placeholder(tf.float32,[2,None,None,3])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])

out_image=Unet(in_image)

out_small = tf.image.resize_bilinear(out_image,[918//2,1374//2])
gt_small = tf.image.resize_bilinear(gt_image,[918//2,1374//2])


# convert images to RGB in [0, 255] for pretrained vgg model
vgg_real=build_vgg19(255*gt_small[:,:,:,::-1])
vgg_fake1=build_vgg19(255*out_small[0:1,:,:,::-1],reuse=True)
vgg_fake2=build_vgg19(255*out_small[1:2,:,:,::-1],reuse=True)

alpha = 0.05
G_loss=F_loss(vgg_real,vgg_fake1) + F_loss(vgg_real,vgg_fake2) + alpha*F_loss(vgg_fake1,vgg_fake2)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

sess=tf.Session()
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("checkpoint")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

loss_sum = tf.summary.scalar('loss',G_loss)
sum_writer = tf.summary.FileWriter('./log',sess.graph)


allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

counter = 1

num_workers = 8
load_fn = load_fn_example
p_queue = PrefetchQueue(load_fn, train_ids, 1, 32, num_workers=num_workers)

learning_rate = 1e-4
for epoch in range(lastepoch,1001):
    if os.path.isdir("result/%04d"%epoch):
        continue
    if epoch > 500:
        learning_rate = 1e-5

    for ind in range(len(train_ids)):
        st = time.time()
        X = p_queue.get_batch()  #load a batch for training

        input_np = X[0]
        gt_np = X[1]

        if np.random.randint(2,size=1)[0] == 1:  # random flip
            input_np = np.flip(input_np, axis=1)
            gt_np = np.flip(gt_np, axis=1)
        if np.random.randint(2,size=1)[0] == 1:
            input_np = np.flip(input_np, axis=2)
            gt_np = np.flip(gt_np, axis=2)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose
            input_np = np.transpose(input_np, (0,2,1,3))
            gt_np = np.transpose(gt_np, (0,2,1,3))

        _,G_current,out_np,sum_str=sess.run([G_opt,G_loss,out_image,loss_sum],feed_dict={in_image:input_np,gt_image:gt_np, lr:learning_rate})

        out_np = np.minimum(np.maximum(out_np,0),1)


        sum_writer.add_summary(sum_str,counter)
        counter += 1


        print("%d %s Loss=%.3f Time=%.3f"%(epoch,ind,G_current, time.time()-st))

        if epoch%save_freq==0:
          #save results for visualization
            if not os.path.isdir("result/%04d"%epoch):
                os.makedirs("result/%04d"%epoch)

            temp = np.concatenate((out_np[0,:,:,:],out_np[1,:,:,:],gt_np[0,:,:,:]),axis=1)*255
            temp = np.clip(temp,0,255)
            cv2.imwrite("result/%04d/train_%s.jpg"%(epoch,ind),np.uint8(temp))


    saver.save(sess,"checkpoint/model.ckpt")
