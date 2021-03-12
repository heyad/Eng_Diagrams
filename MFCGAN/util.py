import numpy as np
import scipy.io
import gzip
import cPickle
import random
#from mnist import MNIST
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from scipy.misc import imresize
import os
#from smrt.balance import smote_balance

random.seed(1024)

def isolate_class(l):
    class_a_idx = []
    for i in range(len(l)):
        if (l[i] == 0 or l[i] == 1 or l[i] == 2 or l[i] == 3 or l[i] == 4 or l[i] == 5 or l[i] == 6 or l[i] == 7 or l[
            i] == 8 or l[i] == 9):
            class_a_idx.append(i)
    return class_a_idx


def vectorized_result(j,n=10):
    '''
    my one-hot encoding
    :param j: label index
    :param n: number of classes
    :return: one-hot encoding
    '''
    e = np.zeros(n)
    e[j] = 1
    return e

def vectorized_result_svhn(j, n=10):
    '''

    my one-hot encoding
    :param j: label index
    :param n: number of classes
    :return: one-hot encoding
    '''
    # e=[]
    # e=[e.append(0) for y in range(n)]
    e = np.zeros(n)
    e[j-1] = 1
    return e

def pickle_mnist():
    f = gzip.open('MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def rap_mnist():
    tr_d, va_d, te_d = pickle_mnist()
    training_data = [np.reshape(x, (28, 28, 1)) for x in tr_d[0]]
    training_labels = [y for y in tr_d[1]]
    validation_data = [np.reshape(x, (28, 28, 1)) for x in va_d[0]]
    test_data = [np.reshape(x, (28, 28, 1)) for x in te_d[0]]
    return training_data, training_labels, validation_data, va_d[1], test_data, te_d[1]


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def plot_svhn(samples):
    '''
    when image is ranged between [-1,1]
    samples =(images+1)*(255/2.0)
    images[images > 255] = 255
    images[images < 0] = 0 
    '''

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(32, 32,3))

    return fig

def plot_fer(samples,image_dim=48):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(7, 7)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(image_dim, image_dim),cmap='Greys_r')

    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 0.02  #0.02#fer2013: 0.2 #emnist: 0.2#mnist:1. / tf.sqrt(in_dim / 2.)#celeba:0.002# 
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def xavier_init2(size):
    in_dim = size[0]
    xavier_stddev = 1.0
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def leaky_relu(x,alpha=0.2):
    return tf.maximum(x, alpha*x)

def huber_loss(labels, predictions, delta=1.0):
    # this function is direct implementation from https://github.com/gitlimlab/SSGAN-Tensorflow/blob/master/ops.py


    residual = tf.abs(predictions - labels)

    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def group_labels(data, num=100):
    # this function is to limit the number of labels used
    # it returns the indexes according the labels
    # data is an array of labels
    # num is the number of labels needed per class

    labels = np.unique(data)
    co_l = []

    for l in labels:
        el_l = [np.where(data == l)]
        co_l.append(np.array(el_l).flatten()[:num])
    return co_l

def group_all_labels(data):
    # this function is to limit the number of labels that are used
    # it returns the indexes according the labels
    # data is an array of labels

    labels = np.unique(data)
    co_l = []

    for l in labels:
        #if l in [0,1]:
        #    el_l = [np.where(data == l)]
       #     co_l.append(np.array(el_l).flatten()[:])
      #  else:
            el_l = np.where(data == l)
            co_l.append(np.array(el_l).flatten()[:4650])
    return co_l


def load_SVHN(file):
    return scipy.io.loadmat(file)


def load_cifar(file):
    my_data= []
    my_label = []
    count =0
    for f in file:
        with open(f,'rb') as inf:
            cifar= cPickle.load(inf)

        data= cifar['data'].reshape((10000, 3, 32, 32))
        data = np.rollaxis(data, 3, 1)
        data = np.rollaxis(data, 3, 1)
        y = cifar['labels']
        my_data.extend(data)
        my_label.extend(y)

    return np.array(my_data), np.array(my_label)

def load_emnist(s):
    emnist = MNIST(s)
    emnist.select_emnist('byclass')
    emnist.gz = False

    images, labels= np.array(emnist.load_training())
    test_images, test_labels = emnist.load_testing()
    images = list([np.reshape(x, (28, 28,1)) for x in images])
    test_images = list([np.reshape(x, (28, 28,1)) for x in test_images])

    return images,list(labels), test_images,list(test_labels)

def get_data(few_idx,training_d,training_l,fs_num):

    fs_test_label=[]
    fs_test_data=[]
    fs_train_label=[]
    fs_train_data=[]

    for l in few_idx:
        fs_labels = []
        fs_data = []
        for k in l:
            fs_data.append(training_d[k])
            fs_labels.append(training_l[k])

        fs_train_data.extend(fs_data[:fs_num])
        fs_train_label.extend(fs_labels[:fs_num])
        fs_test_data.extend(fs_data[fs_num:100+fs_num])
        fs_test_label.extend(fs_labels[fs_num:100+fs_num])

    return fs_train_data,fs_train_label,fs_test_data,fs_test_label

def one_list(my_list):
    flat_list = []
    for i in my_list:
        for j in i:
           flat_list.append(j)
    return flat_list


def binary_labels_generator(mb_labels, label_case):
    train_lab = []
    for x in mb_labels:
        if x == label_case:
            train_lab.append(0)
    else:
        train_lab.append(1)

    return train_lab

def normalize_input(tr_d):
    m = np.array(tr_d).astype(np.float32).mean()
    tr = ((np.array(tr_d)).astype(np.float32) - m) / m
    return tr

def read_attr_by_line(path_txt,t=None):
    training_labels=[]

    with open(path_txt,) as f:
        for line in f:
            attributes = []
            for word in line.split(t):
                attributes.append(word.rstrip('\n'))
            training_labels.append(attributes)

    return training_labels

def process_cleba_labels(tr):

    tr_d=[]
    for att in tr:
        list_att=[]
        for a in att:
            if att.index(a)!=0:
                if a=='-1':
                    list_att.append(0)
                else:
                    list_att.append(1)

        tr_d.append(list_att)
    return tr_d

def resize64(x,r_h=32,r_w=32):
    h, w = x.shape[:2]
    j = int(round((h - 150) / 2.))
    i = int(round((w - 150) / 2.))
    k = imresize(x[j:j + 150, i:i + 150], [r_h, r_w])

    return k

def read_MTFL_textfile(train):
    data=[]

    with open(train) as f:

        for line in f:
            image = []

            for word in line.split():
                image.append(word)

            data.append(image)
    return data

def generate_fake(labels,num=10):
    result=[]
    for x in labels:
        y= [0] * num
        x=x[:num]
        y.extend(x)
        result.append(y)
    return np.array(result)

def svhn_cpickle(file):
    with (open(file, "rb")) as openfile:
        trl = cPickle.load(openfile)
    return trl

def shuffle_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def fer2013_summary(parent):
    no_files = []
    image_classes = []
    for fld in parent:
        image_classes.append(fld)
        no_files.append(len(os.listdir(local_dir + fld)))

    print [(x, y) for x, y in zip(image_classes, no_files)]
    plt.bar(np.arange(len(no_files)), no_files, align='center')
    plt.xticks((np.arange(len(image_classes))), image_classes)
    plt.ylabel('Number of Images')
    plt.xlabel('Image Class')
    return plt, image_classes

def read_fer_images(original_dir,l_dir):
    training_labels = []
    training_data = []
    index=0;
    for fld in original_dir:
        for img in os.listdir(l_dir+fld):
            training_data.append(plt.imread(l_dir+fld+'/'+img))
            training_labels.append(index)
        index+=1
    return training_data,training_labels

def spectral_norm(w, iteration=1):
   w_shape = w.get_shape().as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.Variable(xavier_init([1, w_shape[-1]]), trainable=False, name="u")

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = l2_norm(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = l2_norm(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm


def l2_norm(v, eps=1e-12):
#https://github.com/taki0112/RelativisticGAN-Tensorflow
   return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def savetxt_compact(fname, x, fmt="%s", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

def smote_extend(x_major,y_major,x_minor,y_minor):
    x_temp=[]
    y_temp=[]
    x_temp.extend(x_major)
    x_temp.extend(x_minor)
    y_temp.extend(y_major)
    y_temp.extend(y_minor)
    y_label= y_minor[0]
    x_smote, y_smote = smote_balance(x_temp, y_temp, random_state=40, n_neighbors=25, shuffle=False, balance_ratio=1.0)
    x_output=[]
    y_output=[]

    for x, y in zip(x_smote, y_smote):
        if y == y_label:
            x_output.append(x)
            y_output.append(y)

    return x_output,y_output

def conv_cond_concat(x, y, mb=100,num=10):
  """Concatenate conditioning vector on feature map axis. from https://github.com/jguertl/Conditional-DCGAN/blob/master/ops.py"""
  yb = tf.reshape(y,[mb,1,1,num])
  x_shapes = x.get_shape()
  yb_shapes = yb.get_shape()
  print 'this is the shape of x', x_shapes 
  print 'this is the shape of y', yb_shapes 
  print tf.ones([mb,32, 32, num])
  return tf.concat(axis=3, values=[x, yb*tf.ones([mb, x_shapes[1], x_shapes[2], yb_shapes[3]])])

def conv_concat(x, y, mb=100):
#https://github.com/taki0112/TripleGAN-Tensorflow/blob/master/ops.py
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return tf.concat(values=[x, y * tf.ones([mb, x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def Inner_product(global_pooled, y):
    #global_pooled: B x D,   embeded_y: B x Num_label #https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/ops.py
    H = y.get_shape()[1]
    W = global_pooled.get_shape()[1]
    print 'H ',H
    print 'w', W
    V = tf.Variable(xavier_init([10, 4]))
    V = spectral_norm(V)
    temp = tf.matmul(y, V)
    temp = tf.reduce_sum(temp * global_pooled, axis=1)
    return temp
