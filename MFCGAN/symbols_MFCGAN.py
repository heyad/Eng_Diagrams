import sys
sys.path.insert(0,'../')
from util import *
import numpy as np
#import cv2
import tensorflow as tf
from numpy import *
import matplotlib as mlp
mlp.use('Agg')
from skimage import color
from skimage import io
from collections import  Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

'''
N.B to self but you can give it a go
1. spectral normalization helps
2. gradient penalty not helpful or at least same performance
3. different learning rates for G and D helps (not realy sure?)
4. extra training steps for G is a good decision
5. do not use opencv and plt at the same time (rgb/bgr)
6. best results at 25 epochs upwards
7. batch size plays little role in performance
8. can't do effective G training without batchnorm
9. oversampling works but try to balance fewest among minority
'''


def plot10(samples):
    '''
     this is an auxiliary function to plot generated samples using pyplot
    :param samples: an array of generated images
    :return: a matplotlib figure
    '''
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        # the next 3 lines normalize the image between 0, 255
        # this is because gan uses -1 and 1 norm pixels
        sample =((sample+1)*(255/2.0)).astype(int)
        sample[sample > 255] = 255
        sample[sample < 0] = 0
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64),cmap='Greys_r')

    return fig

np.set_printoptions(threshold=np.inf)

def get_minority(k, dat,lbl):
    '''
     get the minority class of interest k
    :param k: the class label of interest
    :param dat: the set of Images X
    :param lbl:  the class labels y
    :return:  the data, label of class k
    '''
    min_l = []
    min_d = []
    ct =0
    for l in lbl:
        if l==k:
            min_d.append(dat[ct])
            min_l.append(lbl[ct])
        ct+=1


    return min_d, min_l

def get_symbols(dir="symbols29/"):
    '''
    reads symbols images from dir
    :param dir: location of symbols in os path
    :return:
    '''

    data=[]
    labels =[]
    labels_names=[]
    labels_idx=[]
    label_count = 0
    for folder in os.listdir(dir):
        for image in os.listdir(dir+folder+'/'):
            # print dir+folder+'/'+image
            data.append(color.rgb2gray(io.imread(dir+folder+'/'+image)))
            labels.append(label_count)

        labels_idx.append(label_count)
        label_count +=1
        labels_names.append(folder)

    uniq_labels = np.unique(labels)
    label_stat = Counter(labels).values()
    print sorted(zip(label_stat, uniq_labels))
    print zip(labels_idx,labels_names)

    return data, labels, label_count, labels_names


xtrain, ytrain, numclass, label_class = get_symbols()
print 'number of classes : ', numclass

## minority symbols are the minority symbols indexes this might be os dependant
## i,e how the images are ordered and read from the directory/database
## these are my minority indexes
minority_symbols = [1,2,4,5,9,11,13,21,24] # please replace with appropriate indexes

min_data  = []
min_label = []
count = 0

### select minority classes into a set

for l in ytrain:
    if l in minority_symbols:
        min_data.append(xtrain[count])
        min_label.append(l)
    count+=1
####RE-SAMPLING AMONG THE MINORITY####################
## this is done manually for experiemntation and is dependent on the number of samples in the class
for r in range(2):
    d,l=get_minority(11,xtrain,ytrain)
    xtrain.extend(d)
    ytrain.extend(l)

for r in range(2):
    d,l=get_minority(24,xtrain,ytrain)
    xtrain.extend(d)
    ytrain.extend(l)

for r in range(2):
    d,l=get_minority(5,xtrain,ytrain)
    xtrain.extend(d)
    ytrain.extend(l)

d,l=get_minority(4,xtrain,ytrain)
xtrain.extend(d)
ytrain.extend(l)

d,l=get_minority(13,xtrain,ytrain)
xtrain.extend(d)
ytrain.extend(l)

for r in range(2):
    d,l=get_minority(21,xtrain,ytrain)
    xtrain.extend(d)
    ytrain.extend(l)

d,l=get_minority(2,xtrain,ytrain)
xtrain.extend(d)
ytrain.extend(l)

for r in range(5):
    d,l=get_minority(9,xtrain,ytrain)
    xtrain.extend(d)
    ytrain.extend(l)

################### END ###########################
uniq_labels =np.unique(ytrain)

## check symbols distribution
label_stat = Counter(ytrain).values()
print sorted(zip(label_stat,uniq_labels))

mb_size = 64  #batch
X_dim = [64, 64, 1] #image size
y_dim = numclass*2 # double label size to accomodate fake classes
z_dim = 100 # size of noise vector
eps = 1e-8 # a value chosen to avoid NaN error in loss
G_lr = 1e-4 # learning rate for G
D_lr = 1e-4 # learning rate for D
local_dir ='GAN_symbols_complete_rerun_verify/' #lcoation of generated images

#preparing images and labels for training#
xtrain=  np.array([np.reshape(x, (64,64,1)) for x in xtrain])
xtrain = ((xtrain.astype(np.float32) - 127.5) / 127.5) #normalizing pixels values between -1 and 1
ytrain = [vectorized_result(y, y_dim) for y in ytrain] # my one-hot encoding

#preparing minority data
min_data=  np.array([np.reshape(x, (64,64,1)) for x in min_data])
min_data = ((min_data.astype(np.float32) - 127.5) / 127.5)
min_label = np.array([vectorized_result(y, y_dim) for y in min_label])

print 'shape of minority data :', min_data.shape
print 'shape of minority labels :', min_label.shape


X = tf.placeholder(tf.float32, shape=[None, 64, 64, 1]) #input tensor
y = tf.placeholder(tf.float32, shape=[None, y_dim]) #output tensor for real y
fake_y = tf.placeholder(tf.float32, shape=[None, y_dim]) # output tensor for fake y
z = tf.placeholder(tf.float32, shape=[None, z_dim]) #noise vector tensor
condition = tf.placeholder(tf.int32, shape=[], name="condition") # switcher tensor to train with or without labels

#defining G weight and bias sizes for each layer
G_W0 = tf.Variable(xavier_init([z_dim + y_dim, 1024]), name='gw0')
G_b0 = tf.Variable(tf.zeros(shape=[1024]), name='gb0')
G_W1 = tf.Variable(xavier_init([1024, 128 * 8 * 8]), name='gw1')
G_b1 = tf.Variable(tf.zeros(shape=[128 * 8 * 8]), name='gb1')
G_W2 = tf.Variable(xavier_init([5, 5, 256, 128]), name='gw2')
G_b2 = tf.Variable(tf.zeros([256]), name='gb2')
G_W3 = tf.Variable(xavier_init([5, 5, 128, 256]), name='gw3')
G_b3 = tf.Variable(tf.zeros([128]), name='gb3')
G_W4 = tf.Variable(xavier_init([2, 2, 1, 128]), name='gw4')
G_b4 = tf.Variable(tf.zeros(shape=[1]), name='gb4')


def generator(z, c):
    '''
    this is the generator network with leaky relu activation, transpose convolution to increase image size and normal
    matrix multiplication for forst two FC neurons
    :param z: noise vector
    :param c: class label
    :return: generated images
    '''
    inputs = tf.concat(axis=1, values=[z, c])
    G_h0 = lrelu(tf.matmul(inputs, spectral_norm(G_W0)) + G_b0)
    G_h1 = lrelu(tf.matmul(G_h0, spectral_norm(G_W1))+ G_b1)
    print 'shape of G_h1 before reshape:', G_h1.get_shape()
    G_h1 = tf.reshape(G_h1, [-1, 8, 8, 128])
    G_h1 = tf.contrib.layers.batch_norm(G_h1)
    print 'shape of G_h1 after reshape:', G_h1.get_shape()

    G_h2 = lrelu(tf.nn.bias_add( tf.nn.conv2d_transpose(G_h1, spectral_norm(G_W2), output_shape=[mb_size, 16, 16, 256], strides=[1, 2, 2, 1], padding='SAME'), G_b2))
    print 'the shape of G_h2 :', G_h2.get_shape()
    G_h2 = tf.contrib.layers.batch_norm(G_h2)
    G_h3 = lrelu(tf.nn.bias_add(tf.nn.conv2d_transpose(G_h2, spectral_norm(G_W3), output_shape=[mb_size, 32, 32, 128], strides=[1, 2, 2, 1], padding='SAME'), G_b3))
    print 'the shape of G_h3 :', G_h3.get_shape()
    G_h3 = tf.contrib.layers.batch_norm(G_h3)

    G_log_prob = tf.nn.bias_add(tf.nn.conv2d_transpose(G_h3, spectral_norm(G_W4), output_shape=[mb_size, 64, 64, 1], strides=[1, 2, 2, 1], padding='SAME'),G_b4)
    G_prob = tf.nn.tanh(G_log_prob)
    return G_prob


## initializing D weights and biases
D_W0 = tf.Variable(xavier_init([5, 5, 1, 16]), name = 'dw0')
D_b0 = tf.Variable(tf.zeros(shape=[16]), name='db0')
D_W1 = tf.Variable(xavier_init([5, 5, 16, 32]), name = 'dw1')
D_b1 = tf.Variable(tf.zeros(shape=[32]), name = 'db1')
D_W2 = tf.Variable(xavier_init([5, 5, 32, 64]), name = 'dw2')
D_b2 = tf.Variable(tf.zeros(shape=[64]), name = 'db2')

## these are the output parameters of the models
## d_w_gan for normal gan output
### d_w_aux for auxiliary classification
D_W1_gan = tf.Variable(xavier_init([4096, 1]), name = 'dwgan')
D_b1_gan = tf.Variable(tf.zeros(shape=[1]), name = 'dbgan')
D_W1_aux = tf.Variable(xavier_init([4096, y_dim]), name = 'dwaux')
D_b1_aux = tf.Variable(tf.zeros(shape=[y_dim]), name ='dbaux')


def discriminator(X):
    '''
    this is the D network model. uses leaky relu activations and convolution
    :param X: samples of real training images
    :return: gan probability and auxiliary classification
    '''
    D_h0 = lrelu(tf.nn.conv2d(X, spectral_norm(D_W0), strides=[1, 2, 2, 1], padding='SAME') + D_b0)
    print 'shape of D_h0 :', D_h0.get_shape()
    D_h1 = lrelu(tf.nn.conv2d(D_h0, spectral_norm(D_W1), strides=[1, 2, 2, 1], padding='SAME') + D_b1)
    print 'shape of D_h1 :', D_h1.get_shape()
    D_h2 = lrelu(tf.nn.conv2d(D_h1, spectral_norm(D_W2), strides=[1, 2, 2, 1], padding='SAME') + D_b2)
    print 'shape of D_h2 :', D_h2.get_shape()
    D_h3 = tf.reshape(D_h2, [mb_size, -1])

    out_gan = tf.nn.sigmoid(tf.matmul(D_h3, spectral_norm(D_W1_gan)) + D_b1_gan)
    print 'shape of out_gan :', out_gan.get_shape()
    out_aux = tf.matmul(D_h3, spectral_norm(D_W1_aux)) + D_b1_aux
    print 'shape of out_aux :', out_aux.get_shape()
    return out_gan, out_aux

## sets of weights and biases for both D and G. these will be used in training
theta_G = [G_W0, G_W1, G_W2, G_W3, G_W4, G_b0, G_b1, G_b2, G_b3, G_b4]
theta_D = [D_W0, D_W1, D_W2, D_W1_gan, D_W1_aux, D_b0, D_b1, D_b2, D_b1_gan, D_b1_aux]


def sample_z(m, n):
    '''
    these is the random sample method into noise normal distribution
    :param m: batch size
    :param n: size of the noise vector
    :return: a set of noise inputs for G
    '''
    return np.random.uniform(-1., 1., size=[m, n])


def cross_entropy(logit, xy):
    '''

    :param logit: output from D_gan
    :param xy: set of labels for corresponding x inputs
    :return: softmax loss
    '''
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=xy))

G_take = generator(z, y) # g iteration to get generated images
G_sample = G_take

print 'shape of generated images ', G_sample.get_shape()
D_real, C_real = discriminator(X) # d iteration over real images d_real is the gan output, c_real is the classification output
D_fake, C_fake = discriminator(G_sample)  # d iteration over generated images


# GAN D loss
D_loss = tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps))
# the network switcher is used to determine whether to add label loss or not
DC_loss = tf.cond(condition > 0, lambda: -(D_loss +(cross_entropy(C_real, y) + cross_entropy(C_fake, fake_y))), lambda: -D_loss)


# GAN's G loss
G_loss = tf.reduce_mean(tf.log(D_fake + eps))
# network switcher  is used to determine whether to add label loss or not
GC_loss = tf.cond(condition > 0, lambda: -(G_loss +(cross_entropy(C_real, y) + cross_entropy(C_fake, y))), lambda:-G_loss)

# Classification accuracy only if interested in labels classification
correct_prediction = tf.equal(tf.argmax(C_real, 1), tf.argmax(y,1))
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

## defining backprop through D
D_solver = (tf.train.AdamOptimizer(learning_rate=D_lr)
            .minimize(DC_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=G_lr)
            .minimize(GC_loss, var_list=theta_G))

#setting output directory to collect samples
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

#training initiated
sess = tf.Session()
sess.run(tf.global_variables_initializer())


i = 0 #simple count for steps and images
training_labels = np.array(ytrain)
training_data = np.array(xtrain)

for it in range(100000): #make your choice in iteration steps. 100k may not be ideal
    ## creating my own random batching from the number of parameters and batch size
    ind = np.random.choice(training_data.shape[0], mb_size)
    X_mb = np.array(training_data[ind])
    y_mb = np.array(training_labels[ind])#sample_z(mb_size,y_dim)#
    z_mb = sample_z(mb_size, z_dim)
    fake_mb = generate_fake(y_mb, numclass) # generating fake labels from real once

    #trainining step over all samples
    _, DC_loss_curr, acc = sess.run([D_solver, DC_loss, accuracy], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})
    _, GC_loss_curr = sess.run([G_solver, GC_loss], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})

    # extra step for G. this has shown to improve performance
    ind = np.random.choice(training_data.shape[0], mb_size)
    X_mb = np.array(training_data[ind])
    y_mb = np.array(training_labels[ind])
    z_mb = sample_z(mb_size, z_dim)
    fake_mb = generate_fake(y_mb,numclass)
    _, GC_loss_curr = sess.run([G_solver, GC_loss], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})


    if it % 1000 == 0:
        ## some extra training steps on minority classes
        for k in range(10):
                ind = np.random.choice(min_data.shape[0], mb_size)
                X_mb = np.array(min_data[ind])
                y_mb = np.array(min_label[ind])
                z_mb = sample_z(mb_size, z_dim)
                fake_mb = generate_fake(y_mb, numclass)

                _, DC_loss_curr, acc = sess.run([D_solver, DC_loss, accuracy], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})
                _, GC_loss_curr = sess.run([G_solver, GC_loss], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})

                ind = np.random.choice(min_data.shape[0], mb_size)
                X_mb = np.array(min_data[ind])
                y_mb = np.array(min_label[ind])
                z_mb = sample_z(mb_size, z_dim)
                fake_mb = generate_fake(y_mb,numclass)
                _, GC_loss_curr = sess.run([G_solver, GC_loss], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})

        ## generate, save and check samples in the save directory
        samples = []
        for index in minority_symbols:
          s_level = np.zeros([mb_size, y_dim])
          s_level[range(mb_size), index] = 1
          samples.extend(sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim), y: s_level , fake_y:generate_fake(s_level,numclass), condition:1})[:10])

        print('Iter: {}; DC_loss: {:0.4}; GC_loss: {:0.4}; accuracy: {:0.4}; '.format(it,DC_loss_curr, GC_loss_curr,acc))
        fig = plot10(samples[:100])
        plt.savefig(local_dir+'{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

#####save trained samples##############
## this is a post trainin step to generate more symbols for classification
gen_x = []
gen_y = []
for index in minority_symbols:
    for w in range(20):
        s_level = np.zeros([mb_size, y_dim])
        s_level[range(mb_size), index] = 1
        gen_y.extend(s_level)
        gen_x.extend(sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim), y: s_level , fake_y:generate_fake(s_level,numclass), condition:1})[:50])

samples = np.array(gen_x)
np.savez(local_dir + 'generated_samples.npz', samples)

labels = np.array(gen_y)
np.savez(local_dir+'generated_labels.npz',labels)