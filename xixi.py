from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2 
import dlib
import sys

detector = dlib.get_frontal_face_detector()
ID = (1511277,1511278,1511279,1511282,1511283,1511286,1511287,1511289,
    1511290,1511291,1511292,1511295,1511296,1511298,1511300,1511301,
    1511302,1511303,1511304,1511305,1511306,1511307,1511308,1511310,
    1511311,1511312,1511313,1511315,1511317,1511318,1511319,1511323,
    1511325,1511328,1511329,1511330,1511332,1511333,1511334,1511337,
    1511339,1511340,1511341,1511342,1511343,1511344,1511345,1511347,
    1511349,1511350,1511351,1511352,1511353,1511355,1511357,1511358,
    1511360,1511361,1511363,1511365)
 
w = 128
h = 128
c= 3

x= tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.float32,shape=[None,],name='y_')

def CNNlayer():
    #第一个卷积层（128——>64)
    conv1=tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
 
    #第二个卷积层(64->32)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
 
    #第三个卷积层(32->16)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
 
    #第四个卷积层(16->8)
    conv4=tf.layers.conv2d(
          inputs=pool3,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
 
    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])
 
    #全连接层
    dense1 = tf.layers.dense(inputs=re1, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits= tf.layers.dense(inputs=dense2, 
                            units=60,
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits
#---------------------------网络结束---------------------------

logits = CNNlayer()
predict = tf.argmax(logits,1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'ckpt1/faces.ckpt-5')

user = input("G OR v")
if user=="G":
    path=input("图片路径名是：")
    img = cv2.imread(path)
    dets = detector(img, 1) 
    print("Number of faces detected: {}".format(len(dets)))
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        img2 = img[..., ::-1]
        io.imsave('temp.png', img2)
        img1=io.imread('temp.png') 
        img1=transform.resize(img1,(w,h,c))
        cv2.imshow('image',img1)
        
        img1 = img[top:bottom,left:right] 
        img1=transform.resize(img1,(w,h,c))
        # cv2.imshow('image1',img)
        res=sess.run(predict, feed_dict={x:[img1]})
        print(ID[res[0]])
    if len(dets)==0:
        img=transform.resize(img,(w,h,c))
        res=sess.run(predict, feed_dict={x:[img]})
        print(ID[res[0]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
	pass
	

