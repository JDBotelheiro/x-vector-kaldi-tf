#!/usr/bin/env python

"""
# Created by feit on 2019-10-29
email: feit@uber.com
customized utils (python 2)
"""

import tensorflow as tf
import os
import shutil
from ze_utils import set_cuda_visible_devices


def get_average_nnet_model(dir, iter, nnets_list, logger, run_opts=None, get_raw_nnet_from_am=False):
    """
    average the models in nnets list
    :param dir: output root model dir
    :param iter: current iteration number
    :param nnets_list: list of model dirs for being averaged (the 1st one is seen as the main one)
    :param logger: logger
    :param run_opts: [fake arguments]
    :param get_raw_nnet_from_am: [fake arguments]
    :return:
    """
    set_cuda_visible_devices(use_gpu=False, logger=logger)
    with tf.Graph().as_default():
        # define the main net
        nnets_to_average = nnets_list.split(" ")
        main_net_dir = nnets_to_average[0]
        # meta file from main net (assuming all nets have exactly same architecture)
        saver = tf.train.import_meta_graph(os.path.join(main_net_dir,'model.meta'))
        all_vars = tf.trainable_variables()
        # read all model params
        model_params = []
        for net_dir in nnets_to_average:
            temp_sess = tf.Session()
            saver.restore(temp_sess, os.path.join(net_dir,'model'))
            temp_param = temp_sess.run(all_vars)
            model_params.append(temp_param)
            temp_sess.close()
        # do averaging
        #init_op = tf.global_variables_initializer()
        sess_final = tf.Session()
        saver.restore(sess_final, os.path.join(main_net_dir,'model'))       # main net as a template
        all_assign = []
        sum_count = 1
        for var_list in zip(all_vars, *model_params):
            var_list = list(var_list)
            var = var_list[0]
            temp_sum = var_list[1]
            if len(var_list)>=3:
                for sub_var in var_list[2:]:
                    temp_sum += sub_var
                    sum_count += 1
            all_assign.append(tf.assign(var, temp_sum/sum_count))
        sess_final.run(all_assign)
        # save averaged model and copy remaining file
        out_dir = '{0}/model_{1}'.format(dir, iter+1)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        saver.save(sess_final, os.path.join(out_dir,'model'))
        shutil.copyfile(os.path.join(main_net_dir,'checkpoint'),os.path.join(out_dir,'checkpoint'))
        shutil.copyfile(os.path.join(main_net_dir,'done'),os.path.join(out_dir,'done'))
        sess_final.close()

#
# if __name__ == "__main__":
#     get_average_nnet_model('/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_clean/xvector_tf_but_test',48,
#                            '/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_clean/xvector_tf_but_test/model_49.1 '
#                            '/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_clean/xvector_tf_but_test/model_49.2')