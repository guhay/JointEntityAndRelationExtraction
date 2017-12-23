import tensorflow as tf
import numpy as np
from model1 import JointExtraction
inputs=np.array([1,2,3,4,5,6,7,8,9,10,2,3,4,1,2]).reshape(3,5)
inputs1 = np.array([1, 2, 3, 4, 5]).reshape(1,5)
seq_length=np.array([3,4,5])
seq_length1 = np.array([3])
target=np.array([0,1,2,3,0,1,2,3,0,1,2,3,1,2,3]).reshape(3,5)
begin2end=np.array([0,1,2,3,4])
entity1=np.array([1,2])
entity2=np.array([0])
with tf.Session() as sess:
    a = JointExtraction(10, 100, 100, 50, 5,4,0.01,tf.train.RMSPropOptimizer,3,30,6,0.01)
    writer = tf.summary.FileWriter("graph/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        RC_embeding=sess.run([a.RC_Embeding],feed_dict={a.inputs:inputs1,a.sequence_length:seq_length1,a.dropout_keep_prob:1,a.begin2end:begin2end,a.entity1:entity1,a.entity2:entity2})
