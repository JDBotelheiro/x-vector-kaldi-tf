import tensorflow as tf
import os
g1 = tf.Graph()
all_vars = tf.trainable_variables()
saver = tf.train.import_meta_graph('model.meta')
sess1 = tf.Session()
saver.restore(sess1, 'model')
values1 = sess1.run(all_vars)
sess2 = tf.Session()
saver.restore(sess2, 'model')
values2 = sess2.run(all_vars)
init_op = tf.global_variables_initializer()
sess3 = tf.Session()
saver.restore(sess3, 'model')
all_assign = []
for var, var1, var2 in zip(all_vars, values1, values2):
    all_assign.append(tf.assign(var, (var1+var2)/2))
values3 = sess3.run(all_assign)
saver.save(sess3, os.path.join('./debug/', 'model'))
