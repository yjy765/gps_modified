import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[None,2])
W = tf.Variable(tf.random_uniform([2,1],-1,1))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x,W)+b
x_2 = 2*x
label = tf.placeholder(tf.float32,[1])

weight = tf.Variable(np.asarray([1,2],dtype=np.float32).reshape([1,2]),trainable=False)
loss = 0
loss += tf.reduce_mean(tf.square(y-label))
#loss += tf.reduce_mean(tf.multiply(tf.square(y-label),weight))

opt = tf.train.GradientDescentOptimizer(1)

grads_and_vars = opt.compute_gradients(loss)

new_grads_and_vars = []
gv = grads_and_vars[0]
new_grads_and_vars.append((gv[0]+tf.transpose(x_2),gv[1]))
gv = grads_and_vars[1]
print(y)
import sys
new_grads_and_vars.append((gv[0]+tf.reshape(y,[1]),gv[1]))

optim = opt.apply_gradients(new_grads_and_vars)

x_data = np.asarray([1,-1]).reshape([1,2])
label_data = [1]
'''
optim = opt.minimize(loss)
x_data = np.asarray([1,-1]).reshape([1,2])
label_data= [1,2]
'''

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

WW = tf.Variable(tf.random_uniform([2,2],-1,-1))
sess.run(WW.initializer)

print(sess.run(weight))
print(sess.run([W])[0])
print(sess.run([y],feed_dict={x:x_data,label:label_data})[0])
print(sess.run([W,optim,loss],feed_dict={x:x_data,label:label_data}))
print(sess.run([W,optim,loss],feed_dict={x:x_data,label:label_data}))

print(sess.run(W))
print(sess.run([b])[0])
print(sess.run(weight))

W_return = sess.run(W)
print('W_return')
print(W_return)
print(type(W_return))
x_again = tf.multiply(W,W_return)
print(x_again)
print(W)
print(sess.run(x_again,feed_dict={x:x_data}))
print(sess.run(tf.cumsum(W,axis=0)))
