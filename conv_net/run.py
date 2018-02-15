import logging
import os

import numpy as np
import tensorflow as tf

from config import pathDict, fileNames, myNet, netParams, vars
from conv_net.ops import loss_optimization, summary_builder
from conv_net.vgg import conv_1, conv_2, conv_3, conv_4, fc1, fc2, fc3, softmax
from data_transformation.data_io import getPickleFile
from data_transformation.preprocessing import Preprocessing

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def graph():
    inpX = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['crop_shape'][0], myNet['crop_shape'][1], myNet['crop_shape'][2]],
                          name='X')
    inpY = tf.placeholder(dtype=tf.float32,
                          shape=[None, myNet['num_labels']],
                          name='Y')
    
    X = conv_1(inpX)
    X = conv_2(X)
    X = conv_3(X)
    X = conv_4(X)
    X = tf.contrib.layers.flatten(X, scope='flatten')
    netParams['fc1']['shape'][0] = X.get_shape().as_list()[1]
    
    X = fc1(X)
    X = fc2(X)
    X = fc3(X)
    logging.info('The X till this point is: shape %s', str(X.get_shape().as_list()))
    
    logits, probs = softmax(X)
    logging.info('Logits: shape %s', str(logits.get_shape().as_list()))
    logging.info('Probabilities: shape %s', str(probs.get_shape().as_list()))
    
    lossCE, optimizer, l_rate = loss_optimization(X=logits, y=inpY, learning_rate_decay=True)
    
    return dict(inpX=inpX, inpY=inpY, outProbs=probs,
                loss=lossCE, optimizer=optimizer, l_rate=l_rate)


class Train():
    def __init__(self, params, image_type='aerial'):
        
        params_keys = list(params.keys())

        if 'use_checkpoint' in params_keys:
            self.use_checkpoint = params['use_checkpoint']

        if 'save_checkpoint' in params_keys:
            self.save_checkpoint = params['save_checkpoint']
            
        if 'write_tensorboard_summary' in params_keys:
            self.write_tensorboard_summary = params['write_tensorboard_summary']
        
        if image_type=='aerial':
            self.ckpt_path = pathDict['aerial_ckpt_path']
            self.smry_path = pathDict['aerial_smry_path']
            # aerial images are small in size, so to accomodate it, we override the input shape them here
            myNet['image_shape'] = [224,224,3]         # override the shape
            
        elif image_type=='assessor':
            self.ckpt_path = pathDict['assessor_ckpt_path']
            self.smry_path = pathDict['assessor_smry_path']
        else:
            raise ValueError('Provide a valid imgae type')
            
        
        
    
    def reshape(self, x, y):
        return (x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]),
                y.reshape(y.shape[0] * y.shape[1]))
    
    def to_one_hot(self, y):
        y = np.array(y, dtype=int)
        n_values = int(np.max(y)) + 1
        y = np.eye(n_values)[y]
        return y
    
    def accuracy(self, y, y_hat):
        # Both the predictions and the labels should be in One-Hot vector format.
        return (100.0 * np.sum(np.argmax(y_hat, 1) == np.argmax(y, 1)) / y_hat.shape[0])
    
    def run_preprocessor(self, dataIN, sess):
        # logging.info('INITIATING PREPROCESSING.................')
        out_shape = [dataIN.shape[0]] + myNet['crop_shape']
        pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
        for img_no in np.arange(dataIN.shape[0]):
            feed_dict = {
                self.preprocess_graph['imageIN']: dataIN[img_no, :]
            }
            pp_imgs[img_no, :] = sess.run(
                    self.preprocess_graph['imageOUT'],
                    feed_dict=feed_dict
            )
        # logging.info('Preprocessed data Shape: %s', str(pp_imgs.shape))
        return pp_imgs
    
    def cvalid(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(batchX, sess)
        feed_dict = {
            self.train_graph['inpX']: preprocessed_data
        }
        
        out_prob = sess.run(self.train_graph['outProbs'], feed_dict=feed_dict)
        
        v_acc = self.accuracy(y=batchY, y_hat=out_prob)
        
        print("Fold: " + str(self.foldNUM + 1) +
              ", Step: " + str(self.step + 1) +
              ", Cross Validation Validation Accuracy= " + "{:.5f}".format(v_acc))
        
        return v_acc
    
    def train(self, batchX, batchY, sess):
        preprocessed_data = self.run_preprocessor(batchX, sess)
        feed_dict = {
            self.train_graph['inpX']: preprocessed_data,
            self.train_graph['inpY']: batchY
        }
        
        if self.write_tensorboard_summary:
            out_prob, ls, _, l_rt, smry = sess.run([self.train_graph['outProbs'],
                                                    self.train_graph['loss'],
                                                    self.train_graph['optimizer'],
                                                    self.train_graph['l_rate'],
                                                    self.merged_summary],
                                                   feed_dict=feed_dict)
            self.writer.add_summary(smry, self.step)
        else:
            out_prob, ls, _, l_rt = sess.run([self.train_graph['outProbs'],
                                              self.train_graph['loss'],
                                              self.train_graph['optimizer'],
                                              self.train_graph['l_rate']], feed_dict=feed_dict)
        
        acc = self.accuracy(y=batchY, y_hat=out_prob)
        print("Fold: " + str(self.foldNUM + 1) +
              ", Step: " + str(self.step + 1) +
              ", Loss= " + "{:.6f}".format(ls) +
              ", Training Accuracy= " + "{:.5f}".format(acc))
        
        logging.info("Fold: %s, Step: %s, Loss: %s, Accuracy: %s",
                     str(self.foldNUM + 1),
                     str(self.step + 1),
                     str("{:.6f}".format(ls)),
                     str("{:.5f}".format(acc)))
        
        return acc
    
    # def run_step(self):
    #     saver = tf.train.Saver()
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #
    #         # LOAD CHECKPOINTS (WEIGHTS IF NEEDED)
    #         checkpoints = [ck for ck in os.listdir(self.ckpt_path) if ck != '.DS_Store']
    #
    #         if len(checkpoints) > 0 and self.use_checkpoint:
    #             logging.info('CHECKPOINT SAVER: Fetching model updated parameters')
    #             checkpoint_path = os.path.join(self.ckpt_path,
    #                                            fileNames['checkpoint_file_name'] + '.ckpt'
    #                                            if len(fileNames['checkpoint_file_name'].split('.')) == 1
    #                                            else fileNames['checkpoint_file_name'])
    #             saver.restore(sess, checkpoint_path)
    #
    #         # CREATE TENSOR BOARD SUMMARY WRITER (Writer opens up a file and starts writing summary for every step or
    #         #  so)
    #         if self.write_tensorboard_summary:
    #             logging.info('TENSOR BOARD SUMMARY: Dumping Tensorboard summary')
    #             self.merged_summary, self.writer = summary_builder(sess, self.smry_path)
    #
    #         for step in np.arange(self.num_steps):
    #             self.step = step
    #             offset = (step * vars['batch_size']) % (self.train_size - self.remainder)
    #
    #             if offset == (self.train_size - self.remainder - vars['batch_size']):
    #                 batchX = self.trnX[offset:(offset + vars['batch_size'] + self.remainder), :]
    #                 batchY = self.trnY[offset:(offset + vars['batch_size'] + self.remainder), :]
    #             else:
    #                 batchX = self.trnX[offset:(offset + vars['batch_size']), :]
    #                 batchY = self.trnY[offset:(offset + vars['batch_size']), :]
    #
    #             acc = self.train(batchX, batchY, sess)
    #
    #             # if ((step +1) % 10) == 0:
    #             # v_acc = self.cvalid(batchX=self.cvX, batchY=self.cvY, sess=sess)
    #             if step == 30:
    #                 break
    #
    #         # SAVE CHECKPOINTS TO THE PATH
    #         if self.save_checkpoint:
    #             logging.info('CHECKPOINT SAVER: Saving model updated parameters')
    #             checkpoint_path = os.path.join(self.ckpt_path,
    #                                            fileNames['checkpoint_file_name'] + '.ckpt'
    #                                            if len(fileNames['checkpoint_file_name'].split('.')) == 1
    #                                            else fileNames['checkpoint_file_name'])
    #             saver.save(sess, checkpoint_path)

    def run_step(self):
        saver = tf.train.Saver()
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            # LOAD CHECKPOINTS (WEIGHTS IF NEEDED)
            checkpoints = [ck for ck in os.listdir(self.ckpt_path) if ck != '.DS_Store']
        
            if len(checkpoints) > 0 and self.use_checkpoint:
                logging.info('CHECKPOINT SAVER: Fetching model updated parameters')
                checkpoint_path = os.path.join(self.ckpt_path,
                                               fileNames['checkpoint_file_name'] + '.ckpt'
                                               if len(fileNames['checkpoint_file_name'].split('.')) == 1
                                               else fileNames['checkpoint_file_name'])
                saver.restore(sess, checkpoint_path)
        
            # CREATE TENSOR BOARD SUMMARY WRITER (Writer opens up a file and starts writing summary for every step or
            #  so)
            if self.write_tensorboard_summary:
                logging.info('TENSOR BOARD SUMMARY: Dumping Tensorboard summary')
                self.merged_summary, self.writer = summary_builder(sess, self.smry_path)
        
            for step in np.arange(self.num_steps):
                self.step = step
                offset = (step * vars['batch_size']) % (self.train_size - self.remainder)
            
                if offset == (self.train_size - self.remainder - vars['batch_size']):
                    batchX = self.trnX[offset:(offset + vars['batch_size'] + self.remainder), :]
                    batchY = self.trnY[offset:(offset + vars['batch_size'] + self.remainder), :]
                else:
                    batchX = self.trnX[offset:(offset + vars['batch_size']), :]
                    batchY = self.trnY[offset:(offset + vars['batch_size']), :]
            
                acc = self.train(batchX, batchY, sess)
            
                # if ((step +1) % 10) == 0:
                # v_acc = self.cvalid(batchX=self.cvX, batchY=self.cvY, sess=sess)
                if step == 30:
                    break
        
            # SAVE CHECKPOINTS TO THE PATH
            if self.save_checkpoint:
                logging.info('CHECKPOINT SAVER: Saving model updated parameters')
                checkpoint_path = os.path.join(self.ckpt_path,
                                               fileNames['checkpoint_file_name'] + '.ckpt'
                                               if len(fileNames['checkpoint_file_name'].split('.')) == 1
                                               else fileNames['checkpoint_file_name'])
                saver.save(sess, checkpoint_path)

    def run(self, trX, trY, cvX, cvY, label_dict):
        logging.info('INITIATING RUN ........')
        self.foldNUM = 0
        self.trnX, self.trnY = self.reshape(trX, trY)

        print (self.trnX.shape, self.trnY.shape, cvX.shape, cvY.shape)
        self.cvX, self.cvY = cvX, cvY
        self.trnY = self.to_one_hot(self.trnY)
        self.cvY = self.to_one_hot(self.cvY)
        print('trnX shape: ', self.trnX.shape)
        print('trnY shape: ', self.trnY.shape)
        print('cvX shape: ', self.cvX.shape)
        print('cvY shape: ', self.cvY.shape)
        #
        # # print (trnY)
        # self.preprocess_graph = Preprocessing().preprocessImageGraph(myNet['image_shape'])
        # self.train_graph = graph()
        #
        # self.train_size = self.trnX.shape[0]
        # self.num_steps = int(vars["epochs"] * self.train_size) // vars['batch_size']
        # self.remainder = self.train_size % vars['batch_size']
        #
        # print('Total number of step: ', self.num_steps)
        # print(self.remainder)
        #
        # ########   RUN THE SESSION
        # self.run_step()


# dataX, dataY, label_dict = getPickleFile(pathDict['aerial_batch_path'],
#                                        fileNames['batch_img_file'])



def load_data(image_type, image_shape, tr_batch_size=64, cv_batch_size=200):

    if image_type == 'aerial':
        tr_batch_file_names = [path for path in os.listdir(pathDict['aerial_batch_path']) if
                               path not in ['cv.pickle', '.DS_Store']]
        data_path = pathDict['aerial_batch_path']
    elif image_type == 'assessor':
        tr_batch_file_names = [path for path in os.listdir(pathDict['assessor_batch_path']) if
                               path not in ['cv.pickle', '.DS_Store']]
        data_path = pathDict['assessor_batch_path']
    else:
        raise ValueError('The image type doesnt match the type handled')

    
    # LOAD ALL THE TRAINING DATA INTO ONE ND_ARRAY
    trX = np.ndarray((len(tr_batch_file_names),
                      tr_batch_size,
                      image_shape[0],
                      image_shape[1],
                      image_shape[2]))

    trY = np.ndarray((len(tr_batch_file_names),
                      tr_batch_size))
    for num, flname in enumerate(tr_batch_file_names):
        trX[num, :], trY[num, :], _ = getPickleFile(data_path, flname)

    # LOAD ALL THE VALIDATION DATA INTO ONE ND_ARRAY
    cvX, cvY, label_dict = getPickleFile(data_path, 'cv.pickle')

    logging.info('Data Featched from %s', str(data_path))

    return trX, trY, cvX, cvY, label_dict





# image_type = 'aerial'
# image_shape = [224,224,3]
image_type = 'assessor'
image_shape = [260,260,3]

run = True
if run :
    
    trX, trY, cvX, cvY, label_dict = load_data(image_type=image_type,
                                               image_shape=image_shape,
                                               tr_batch_size=64, cv_batch_size=200)
    
    Train(dict(use_checkpoints=False,
            save_checkpoints=True,
        write_tensorboard_summary=False), image_type=image_type).run(trX, trY, cvX, cvY, label_dict)