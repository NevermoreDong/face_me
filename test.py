from skimage import io,transform
import glob
import os
import tensorflow as tf 
import numpy as np 
import time 

path = 'E:/Programming/project/face_me/faces'

w = 128
h = 128
c = 3


def read_image(path):
	
	cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
	# cate为path路径下所有文件夹的路径  的列表
	imgs = []
	labels = []
	for idx,folder in enumerate(cate):
	#将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出索引和数据
		for im in glob.glob(folder+'/*.png'): #返回folder下所有png的文件
			#print('reading the image:%s'%(im))
			img = io.imread(im)
			img = transform.resize(img, (w,h,c))
			imgs.append(img)
			labels.append(idx)
	return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

print('have read the images')
data,label = read_image(path)

num_example = data.shape[0] #返回矩阵的行数 1返回矩阵的列数
arr = np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]    #在行的层面上以arr的顺序对data重新排列
label=label[arr]

ratio = 0.8
s = np.int(num_example*ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

x = tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_ = tf.placeholder(tf.int32,shape=[None,],name='y_')

def CNNllayer():
	print('IN CNN LAYERS')
	conv1 = tf.layers.conv2d(
		inputs=x, 
		filters=32, 
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
		)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
	
	conv2 = tf.layers.conv2d(
		inputs=pool1, 
		filters=64, 
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
		)	
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

	conv3 = tf.layers.conv2d(
		inputs=pool2, 
		filters=128, 
		kernel_size=[3,3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
		)
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

	conv4 = tf.layers.conv2d(
		inputs=pool3, 
		filters=128, 
		kernel_size=[3,3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
		)
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)

	re1 = tf.reshape(pool4, [-1,8*8*128])

	dense1 = tf.layers.dense(
		inputs=re1, 
		units=1024,
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
		)
	dense2 = tf.layers.dense(
		inputs=dense1, 
		units=512,
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
		)
	logits = tf.layers.dense(
		inputs=dense2, 
		units=60,
		activation=None,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
		)
	return logits  #logits的shape为1x1x60 60个这个人的特征

logits = CNNllayer()
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False): #四个参数是：训练数据，测试数据，用户输入的每批训练的数据数量，shuffle是洗牌的意思，这里表示是否开始随机。
    assert len(inputs) == len(targets)  #assert断言机制，如果后面的表达式为真，则直接抛出异常。在这里的意思,大概就是:样本和标签数量要对上
    if shuffle:
        indices = np.arange(len(inputs)) #生成一个np.arange可迭代长度是len(训练数据),也就是训练数据第一维数据的数量(就是训练数据的数量，训练图片的数量)。
        np.random.shuffle(indices)  #np.random.shuffle打乱arange中的顺序，使其随机循序化，如果是数组，只打乱第一维。
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size): # 这个range(初始值为0，终止值为[训练图片数-每批训练图片数+1]，步长是[每批训练图片数])：例(0[起始值],80[训练图片数]-20[每批训练图片数],20[每批训练图片数]),也就是(0,60,20)当循环到60时,会加20到达80的训练样本.
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size] # 如果shuffle为真,将indices列表,切片(一批)赋值给excerpt
        else:
            excerpt = slice(start_idx, start_idx + batch_size) # 如果shuffle为假,将slice()函数(切片函数),实例化,初始值为start_idx,结束为止为start_idx + batch_size(也就是根据上一批起始,算出本批结束的位置.),间距为默认.
        yield inputs[excerpt], targets[excerpt]
        #yield常见用法：该关键字用于函数中会input把函数包装为generator。然后可以对该generator进行迭代: for x in fun(param).
        #按照我的理解，可以把yield的功效理解为暂停和播放。
        #在一个函数中，程序执行到yield语句的时候，程序暂停，返回yield后面表达式的值，在下一次调用的时候，从yield语句暂停的地方继续执行，如此循环，直到函数执行完。
        #此处,就是返回每次循环中 从inputs和targets列表中,截取的 经过上面slice()切片函数定义过的 数据.
        #(最后的shuffle变量，决定了样本是否随机化)


saver = tf.train.Saver(max_to_keep=3)
max_acc=0
f=open('ckpt1/acc.txt','w')


n_epoch = 10
batch_size=64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
	print(epoch)
	start_time = time.time()

	train_loss, train_acc, n_batch = 0, 0, 0 
	for x_train_a,y_train_a in minibatches(x_train,y_train,batch_size,shuffle=True):
		_,err,ac = sess.run([train_op,loss,acc],feed_dict={x:x_train_a,y_:y_train_a})
		train_loss += err;train_acc += ac;n_batch += 1
	print("train loss: %f" % (train_loss/ n_batch))
	print("train acc: %f" % (train_acc/ n_batch))


	val_loss, val_acc, n_batch = 0, 0, 0
	for x_val_a,y_val_a in minibatches(x_val,y_val,batch_size,shuffle=False):
		err, ac = sess.run([loss,acc],feed_dict={x:x_val_a,y_:y_val_a})
		val_loss += err;val_acc += ac;n_batch += 1
	print("validation loss: %f" % (val_loss/ n_batch))
	print("validation acc: %f" % (val_acc/ n_batch))

	f.write(str(epoch+1)+', val_acc: '+str(val_acc)+'\n')
	if val_acc>max_acc:
		max_acc = val_acc
		saver.save(sess, 'ckpt1/faces.ckpt',global_step=epoch+1)

f.close()
sess.close()