import os
import random as rn
import numpy
import tensorflow as tf
from keras import backend as K

seed = 42


''' This  function fix the seed for reproducibility of problem with Keras '''


def setup(seed_value=seed):

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    rn.seed(seed_value)
    numpy.random.seed(seed_value)
    tf.set_random_seed(seed_value)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    return
