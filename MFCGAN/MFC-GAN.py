import gzip 
import cPickle
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf

def lrelu(x, alpha=0.2):
    '''
     my leaky-relu activation
    :param x: weight value
    :param alpha: slope
    :return: lrelu smooth
    '''
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def xavier_init(size)
    '''
    weight initialization function
    :param size: filter size
    :return: random normal values
    '''
    in_dim = size[0]
    xavier_stddev = 0.02  
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def pickle_mnist():
    '''
    loading MNIST data
    :return: train validate and test data
    '''
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def rap_mnist():
    '''
    separates data from labels
    :return: train test validate (x,y)
    '''
    tr_d, va_d, te_d = pickle_mnist()
    training_data = [np.reshape(x, (28, 28, 1)) for x in tr_d[0]]
    training_labels = [y for y in tr_d[1]]
    validation_data = [np.reshape(x, (28, 28, 1)) for x in va_d[0]]
    test_data = [np.reshape(x, (28, 28, 1)) for x in te_d[0]]
    return training_data, training_labels, validation_data, va_d[1], test_data, te_d[1]

def vectorized_result(j,n=10):

    '''
    one-hot encoding
    :param j: label
    :param n: number of classes
    :return: one-hot encoding
    '''
    e = np.zeros(n)
    e[j] = 1
    return e

def generate_fake(labels,num=10):
    '''
    generate fake labels from real
    :param labels: real label
    :param num: number of classes
    :return: fake one-hot encoding
    '''
    result=[]
    for x in labels:
        y= [0] * num
        x=x[:num]
        y.extend(x)
        result.append(y)
    return np.array(result)

def plot(samples):
    '''
    just a plot function
    :param samples: list of generated image samples
    :return: matplotlib figure
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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def group_all_labels(data, num=100, minor=[]):
    # this function is to limit the number of labels that are used
    # it returns the indexes according the labels
    # data is an array of labels
    '''

    :param data: array of labels
    :param num: number required
    :param minor: list of minority indexes
    :return: array of labels indexes
    '''

    labels = np.unique(data)
    co_l = []
    if not minor:
        for l in labels:
            el_l = np.where(np.array(data) == l)
            co_l.append(el_l[0])

    else:
        for l in labels:
            if l in minor:
                el_l = np.where(np.array(data) == l)
                co_l.append((el_l[0])[:num])
            else:
                el_l = np.where(np.array(data) == l)
                co_l.append(el_l[0])
    return co_l


training_data1, training_labels1, validation_data, validation_labels, test_data, test_labels = rap_mnist()
training_data1.extend(validation_data)
training_labels1.extend(validation_labels)
test_data = np.array(test_data)

#sample save directory
local_dir = 'gen_samples/'
 ##trimming the number of samples in training classes
grouped_labels = group_all_labels(training_labels1, 50, [0, 1]) #for 50 samples in classes 0 and 1
gr_data = [] #actual training data
gr_labels = [] #actual training labels

## using label indexes to read data selected for training
for q in grouped_labels:
    print(len(q))
    for r in q:
        gr_data.append(training_data1[r])
        gr_labels.append(training_labels1[r])

#preparing data for training
training_data = np.array(gr_data)
training_data = ((training_data.astype(np.float32) - 127.5) / 127.5) ## -1 to 1 pixel normalization
training_labels = np.array([vectorized_result(l,20) for l in gr_labels]) ## one-hot encoding of labels

mb_size = 64 #batch size
X_dim = [28, 28, 1] # image dim
y_dim = 20 # labels  dim 10*2 because of fake classes
z_dim = 110 # size of noise vectorencoding
h_dim = 512  #output size
eps = 1e-8 # value to avoid NaN in loss
G_lr = 1e-3 # G learning rate
D_lr = 1e-3 # D learning rate

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) # input tensor
y = tf.placeholder(tf.float32, shape=[None, y_dim]) # real label tensor
fake_y = tf.placeholder(tf.float32, shape=[None, y_dim]) # fake label tensor
z = tf.placeholder(tf.float32, shape=[None, z_dim]) # noise vector tensor
condition = tf.placeholder(tf.int32, shape=[], name="condition") # network switcher

#G weight initializations
G_W0 = tf.Variable(xavier_init([z_dim + y_dim, 1024]), name='gw0')
G_b0 = tf.Variable(tf.zeros(shape=[1024]), name='gb0')
G_W1 = tf.Variable(xavier_init([1024, 128 * 7 * 7]), name='gw1')
G_b1 = tf.Variable(tf.zeros(shape=[128 * 7 * 7]), name='gb1')
G_W2 = tf.Variable(xavier_init([5, 5, 256, 128]), name='gw2')
G_b2 = tf.Variable(tf.zeros([256]), name='gb2')
G_W3 = tf.Variable(xavier_init([5, 5, 128, 256]), name='gw3')
G_b3 = tf.Variable(tf.zeros([128]), name='gb3')
G_W4 = tf.Variable(xavier_init([2, 2, 1, 128]), name='gw4')
G_b4 = tf.Variable(tf.zeros(shape=[1]), name='gb4')


def generator(z, c):
    '''
    generator network
    :param z: noise vector
    :param c: one-hot encoding class
    :return: generated image
    '''
    inputs = tf.concat(axis=1, values=[z, c])
    G_h0 = lrelu(tf.matmul(inputs, G_W0) + G_b0)
    G_h1 = lrelu(tf.matmul(G_h0, G_W1)+ G_b1)
    print( 'shape of G_h1 before reshape:', G_h1.get_shape())
    G_h1 = tf.reshape(G_h1, [-1, 7, 7, 128])
    G_h1 = tf.contrib.layers.batch_norm(G_h1)
    print( 'shape of G_h1 after reshape:', G_h1.get_shape())

    G_h2 = lrelu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(G_h1, G_W2, output_shape=[mb_size, 7, 7, 256], strides=[1, 1, 1, 1], padding='SAME'),
        G_b2))
    print ('the shape of G_h2 :', G_h2.get_shape())
    G_h2 = tf.contrib.layers.batch_norm(G_h2)
    G_h3 = lrelu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(G_h2, G_W3, output_shape=[mb_size, 14, 14, 128], strides=[1, 2, 2, 1], padding='SAME'),
        G_b3))
    print ('the shape of G_h3 :', G_h3.get_shape())
    G_h3 = tf.contrib.layers.batch_norm(G_h3)

    G_log_prob = tf.nn.bias_add(
        tf.nn.conv2d_transpose(G_h3, G_W4, output_shape=[mb_size, 28, 28, 1], strides=[1, 2, 2, 1], padding='SAME'),
        G_b4)
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_probvector

## discriminaator weight initializers
D_W0 = tf.Variable(xavier_init([5, 5, 1, 32]), name = 'dw0')
D_b0 = tf.Variable(tf.zeros(shape=[32]), name='db0')vector
D_W1 = tf.Variable(xavier_init([5, 5, 32, 64]), name = 'dw1')
D_b1 = tf.Variable(tf.zeros(shape=[64]), name = 'db1')
D_W2 = tf.Variable(xavier_init([5, 5, 64, 128]), name = 'dw2')
D_b2 = tf.Variable(tf.zeros(shape=[128]), name = 'db2')
D_W3 = tf.Variable(xavier_init([5, 5, 128, 256]), name = 'dw3')
D_b3 = tf.Variable(tf.zeros([256]), name = 'db3')

## output weigth initializations d_gan for can output while d_aux for classification
D_W1_gan = tf.Variable(xavier_init([1024, 1]), name = 'dwgan')
D_b1_gan = tf.Variable(tf.zeros(shape=[1]), name = 'dbgan')
D_W1_aux = tf.Variable(xavier_init([1024, y_dim]), name = 'dwaux')
D_b1_aux = tf.Variable(tf.zeros(shape=[y_dim]), name ='dbaux')


def discriminator(X):
    '''
    discriminator network
    :param X: input image batch
    :return: gan logits and classification logits
    '''
    D_h0 = lrelu(tf.nn.conv2d(X, D_W0, strides=[1, 2, 2, 1], padding='SAME') + D_b0)
    print ('shape of D_h0 :', D_h0.get_shape())
    #D_h0 = tf.contrib.layers.batch_norm(D_h0)
    D_h1 = lrelu(tf.nn.conv2d(D_h0, D_W1, strides=[1, 2, 2, 1], padding='SAME') + D_b1)
    print ('shape of D_h1 :', D_h1.get_shape())
    D_h1 = tf.contrib.layers.batch_norm(D_h1)
    D_h2 = lrelu(tf.nn.conv2d(D_h1, D_W2, strides=[1, 2, 2, 1], padding='SAME') + D_b2)
    print ('shape of D_h2 :', D_h2.get_shape())
    D_h2 = tf.contrib.layers.batch_norm(D_h2)
    D_h3 = lrelu(tf.nn.conv2d(D_h2, D_W3, strides=[1, 2, 2, 1], padding='SAME') + D_b3)
    print ('shape of d_h3 :', D_h3.get_shape())
    D_h3 = tf.reshape(D_h3, [mb_size, -1])

    out_gan = tf.nn.sigmoid(tf.matmul(D_h3, D_W1_gan) + D_b1_gan)
    print( 'shape of out_gan :', out_gan.get_shape())
    out_aux = tf.matmul(D_h3, D_W1_aux) + D_b1_aux
    print ('shape of out_aux :', out_aux.get_shape())
    return out_gan, out_aux

## weight sets for G aand D
theta_G = [G_W0, G_W1, G_W2, G_W3, G_W4, G_b0, G_b1, G_b2, G_b3, G_b4]
theta_D = [D_W0, D_W1, D_W2, D_W3, D_W1_gan, D_W1_aux, D_b0, D_b1, D_b2, D_b3, D_b1_gan, D_b1_aux]


def sample_z(m, n):
    '''
    samples noise
    :param m: batch
    :param n: size of vector
    :return: random sample from normal distribution
    '''
    return np.random.uniform(-1., 1., size=[m, n])


def cross_entropy(logit, xy):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=xy))

#generator iteration
G_take = generator(z, y)
G_sample = ((G_take - 127.5) / 127.5) #not necessary ??

print( 'shape of generated images ', G_sample.get_shape())
D_real, C_real = discriminator(X) # discriminator iteration on real images
D_fake, C_fake = discriminator(G_sample) # discriminator iteration on generated images



# GAN D loss
D_loss = tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps)) # gan loss
#using network switcher to determin whether to add classification loss
DC_loss = tf.cond(condition > 0, lambda: -(D_loss +(cross_entropy(C_real, y) + cross_entropy(C_fake, fake_y))), lambda: -D_loss)


# GAN's G loss
G_loss = tf.reduce_mean(tf.log(D_fake + eps)) #gan loss
## using network switcher to determine if classification loss is included or not
GC_loss = tf.cond(condition > 0, lambda: -(G_loss +(cross_entropy(C_real, y) + cross_entropy(C_fake, y))), lambda:-G_loss)


# Classification accuracy if training with labels
correct_prediction = tf.equal(tf.argmax(C_real, 1), tf.argmax(y,1))
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

## backpropagatin loss through d and g
D_solver = (tf.train.AdamOptimizer(learning_rate=D_lr)
            .minimize(DC_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=G_lr)
            .minimize(GC_loss, var_list=theta_G))


#creating sample output directory
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#simple count
i = 0
print (training_data.shape)
for it in range(100000): # you can choose to stop early?
    ### creating my own random batching for each training step
    ind = np.random.choice(training_data.shape[0], mb_size)
    X_mb = np.array(training_data[ind])
    y_mb = np.array(training_labels[ind])
    z_mb = sample_z(mb_size, z_dim)
    fake_mb = generate_fake(y_mb,10)

    # 1 D and G training step
    _, DC_loss_curr, acc= sess.run([D_solver, DC_loss, accuracy], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})
    _, GC_loss_curr = sess.run([G_solver, GC_loss], feed_dict={X: X_mb, y: y_mb, z: z_mb, fake_y:fake_mb, condition:1})

    if it % 500 == 0:
        ### generating sample outputs after each 500 step
        samples = []
        for index in range(10):
          s_level = np.zeros([mb_size, y_dim])
          s_level[range(mb_size), index] = 1
          samples.extend(sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim), y: s_level , fake_y:generate_fake(s_level,10), condition:1})[:10])

        print('Iter: {}; DC_loss: {:0.3}; GC_loss: {:0.3}; accuracy: {:0.3}; '.format(it,DC_loss_curr, GC_loss_curr,acc))
        fig = plot(samples[:100])
        plt.savefig(local_dir+'{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
