import os

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import argparse

from datasets import imagenet
from custom_nets import inception_v3
from preprocessing import inception_preprocessing
import helper_function
from slice_function import *

import time
import math

def show_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_plot = plt.imshow(img)
    # Set up the plot and hide axes
    plt.title('test')
    img_plot.axes.get_yaxis().set_ticks([])
    img_plot.axes.get_xaxis().set_ticks([])
    plt.show()


def load_image(img_path):
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ImageClassifier():
    def __init__(self):
        self.slim = tf.contrib.slim
        self.image_size = inception_v3.inception_v3.default_image_size
        self.checkpoints_dir = 'checkpoints'
        self.names = imagenet.create_readable_names_for_imagenet_labels()
        self.arg_scope = inception_v3.inception_v3_arg_scope()

        self.image = tf.placeholder(tf.uint8, [480, 640, 3])
        #self.label = tf.placeholder(tf.int64, (1,))
        self.type = None
        self.type_pred = None
        self.threshold = None
        self.ratio = None

        self.processed_image = inception_preprocessing.preprocess_image(self.image,
                                                                        self.image_size, self.image_size,
                                                                        is_training=False)
        self.processed_images = tf.expand_dims(self.processed_image, 0)

        # processed_images will be a 1x299x299x3 tensor of float32

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with self.slim.arg_scope(self.arg_scope):
            self.cnet, self.middle, self.logits, self.end_points = inception_v3.inception_v3(self.processed_images, num_classes=1001,
                                                                     is_training=False)
            self.probs = tf.nn.softmax(self.logits)

        self.init_fn = self.slim.assign_from_checkpoint_fn(
            os.path.join(self.checkpoints_dir, 'inception_v3.ckpt'),
            self.slim.get_model_variables('InceptionV3'))
        
        self.slice_logits = tf.placeholder(tf.float32, (1, 1001))
        self.slice_aux_logits = tf.placeholder(tf.float32, (1, 1001))
        self.slice_labels = tf.placeholder(tf.int32, (1,1))      
        self.loss = helper_function.inception_loss(self.slice_logits, self.slice_aux_logits, self.slice_labels)
        #self.mimage = tf.placeholder("float", self.cnet.get_shape())
        print('##################', self.middle)
        print('@@@@@@@@@@@@@@@@@@', self.logits)
        self.session = tf.Session()
        self.init_fn(self.session)
        self.writer = tf.summary.FileWriter("output", self.session.graph)

    def writes(self):

        self.writer = tf.summary.FileWriter("output", self.session.graph)

        
    def build_labels(self, label):
        self.type = label

    def classify_middle(self, middle):
        feed_dict_m = {self.middle: middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))
        #self.type_pred = sorted_inds[0]

    def classify(self, img):
        height, width = img.shape[:2]

        feed_dict = {self.image: img}
        middle = self.session.run(self.cnet, feed_dict=feed_dict)
        feed_dict_m = {self.middle: middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        self.type_pred = sorted_inds[0]

    def classify_loud(self, img):
        height, width = img.shape[:2]
        self.human_img = img
        feed_dict = {self.image: img}
        middle = self.session.run(self.cnet, feed_dict=feed_dict)
        feed_dict_m = {self.middle: middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        self.type_pred = sorted_inds[0]

        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))
    
    def classify_sliced (self, img, slice_map):
        height, width = img.shape[:2]
        self.human_img = img
        feed_dict = {self.image: img}
        middle = self.session.run(self.cnet, feed_dict=feed_dict)
        masked_middle = middle.copy()
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    masked_middle[0,i,j,:] = 0
                    #for k in range(masked_middle.shape[3]):
                        #del_middle(masked_middle[0,:,:,k], i, j, 0)

        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        

        for i in range(1):
            index = sorted_inds[i]
            #print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))

        return sorted_inds[0]

    def classify_middle_sliced (self, slice_map):
        """
        masked_middle = np.zeros(self.m.shape)
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] != 0:
                    for k in range(masked_middle.shape[3]):
                        masked_middle[0,i,j,k] = self.m[0,i,j,k]
                        #del_middle(masked_middle[0,:,:,k], i, j, 0)
        

        """
        masked_middle = self.m.copy()
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    masked_middle[0,i,j,:] = 0
                    #for k in range(masked_middle.shape[3]):
                        #masked_middle[0,i,j,k] = 0
                        #del_middle(masked_middle[0,:,:,k], i, j, 0)
        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        

        #for i in range(1):
        #    index = sorted_inds[i]
            #print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))

        return sorted_inds[0]


    
    def classify_sliced_loud (self, img, slice_map):
        height, width = img.shape[:2]
        self.human_img = img
        feed_dict = {self.image: img}
        middle = self.session.run(self.cnet, feed_dict=feed_dict)
        masked_middle = middle.copy()
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    #del_middle(masked_middle, i, j, 0)
                    for k in range(masked_middle.shape[3]):
                        del_middle(masked_middle[0,:,:,k], i, j, 0)

        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        

        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))

        #return sorted_inds[0]

    def build_middle(self, img):
        self.m = self.session.run(self.cnet, feed_dict={self.image: img})
        return self.m

    def middle_loss(self, middle, label=None):
        auxiliary_logits = self.end_points['AuxLogits']
        logits, auxs = self.session.run([self.logits, auxiliary_logits], feed_dict={self.middle:middle})
        #print(logits)
        if label is None:
            type = self.type_pred
        else:
            type = label

        #feed_dict = {self.label: [type]}
        #print(type)
        #logits = tf.cast([logits, auxs], tf.float32)
        # logits = tf.cast(logits, tf.float32)
        # aux_logits = tf.cast(auxs, tf.float32)
        # labels = tf.cast([[type]], tf.int32)
        # print(logits)
        # print(aux_logits)
        # print(labels)
        

        return self.session.run(self.loss, feed_dict={self.slice_logits:logits,
                                                    self.slice_aux_logits:auxs, 
                                                    self.slice_labels: [[type]]})

    def middle_loss_sliced (self, middle, slice_map, label=None):
        masked_middle = middle.copy()
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    #del_middle(masked_middle, i, j, 0)
                    for k in range(masked_middle.shape[3]):
                        del_middle(masked_middle[0,:,:,k], i, j, 0)

        return self.middle_loss(masked_middle, label)
        #cost, n_, top_indices = self.session.run ([self.cost, self._, self.top_k_pred], feed_dict = {self.mimages: masked_middle, self.label:[self.type]})
        #return cost, n_, top_indices

    def criterion (self, infer_map, threshold=0.0, ratio=0.1):
        if self.threshold is not None:
            threshold = self.threshold
        if self.ratio is not None:
            ratio = self.ratio
        
        am = np.amin(infer_map)
        num = 0
        for i in range(infer_map.shape[0]):
            for j in range(infer_map.shape[1]):
                if abs(infer_map[i,j]) > self.cost_val*threshold:
                    num += 1
        if num/(infer_map.shape[0]*infer_map.shape[1] - len(self.pixel_history)) > ratio:
            return True
        else:
            return False

    def infer_map_build (self, middle, slice_map, dynamic_window=True):
        #masked_middle = middle.copy()
        self.cost_val = self.middle_loss(middle)

        a = middle.shape
        infer_map = np.zeros(a[1:3])

        if dynamic_window:
            flag_expand = False
            flag_shrink = False
            old_infer_map = infer_map.copy()
        else:
            self.window_size = 0
        
        while(True):
            for i in range(a[1]):
                for j in range(a[2]):
                    if slice_map[i,j] == 0:
                        infer_map[i,j] = 0
                        continue
                    temp_middle = middle.copy()
                    for k in range(a[3]):
                        del_middle(temp_middle[0,:,:,k],i,j,self.window_size)
                    delta = self.middle_loss_sliced(temp_middle, slice_map) - self.cost_val

                    #delta = self.session.run(self.cost, feed_dict={self.mimages:temp_middle, self.label:[self.type]}) - self.cost_val
                    if delta != 0:
                        infer_map[i,j] = delta
            
            if not dynamic_window:
                break
            
            if self.criterion(infer_map):
                flag_shrink = True
                if self.window_size == 0:
                    break
                elif flag_expand:
                    break
                else:
                    self.window_size -= 1
            else:
                self.window_size += 1
                if self.window_size > 5:
                    print('amazing!')
                    self.window_size = 5
                    break
                if flag_shrink:
                    infer_map = old_infer_map
                    break
                flag_expand = True
            old_infer_map = infer_map.copy()
        return infer_map

    def slice_minimum (self, middle, dynamic_window=True):
        
        self.pixel_history = []
        self.cost_history = []
        self.masked_history = []
        self.window_history = []
        self.time_stamp = []
        self.window_size = 0
        a = middle.shape
        #infer_map = np.zeros(a[1:3])
        slice_map = np.ones(a[1:3])
        
        for pix in range(a[1]*a[2]):
            t1 = time.time()
            infer_map = self.infer_map_build(middle, slice_map)
            min_i, min_j = min_infer (infer_map, slice_map)
            if (min_i == -1) or (min_j == -1):
                print('#### Can\'t slice anymore')
                break
            slice_map[min_i, min_j] = 0
            
            new_middle = middle.copy()
            for k in range(a[3]):
                del_middle(new_middle[0,:,:,k], min_i, min_j, 0)
            

            probabilities = self.session.run(self.probs, feed_dict={self.middle:new_middle})
            new_cost = self.middle_loss(new_middle)
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]           
            #top_indices, new_cost = self.session.run([self.top_k_pred, self.cost], feed_dict = {self.mimages: new_middle, self.label:[self.type]})
            if sorted_inds[0] != self.type_pred:
                slice_map[min_i, min_j] = 1
                #print('#### this is not [' + self.categories[self.type]+'] anymore...')
                break
            if pix % 50 == 0:
                self.masked_history.append (mask_save(self.human_img, slice_map, pix))
                #slice_save(self.human_img, slice_map, pix)
            self.pixel_history.append((min_i,min_j))
            self.cost_history.append(new_cost)
            self.window_history.append(self.window_size)
            self.time_stamp.append(time.time() - t1)

            if (pix+1) % 1 == 0:
                print('sliced  '+str(pix+1)+'pixels...')
                print('window currently:    ', self.window_size)
                print('time_stamp:          %.4f'% self.time_stamp[pix])

            middle = new_middle
        
        self.masked_history.append (mask_save(self.human_img, slice_map, pix))
        #slice_save(self.human_img, slice_map, pix)
        print('total time:  ', sum(self.time_stamp))
        self.classify_middle(new_middle)
    
    def result_show(self, tag=None):
        with open('./history_new2.bin', 'wb') as f:
            pickle.dump(self.pixel_history, f)
            pickle.dump(self.cost_history, f)
            pickle.dump(self.masked_history, f)
            pickle.dump(self.window_history, f)
            pickle.dump(self.time_stamp,f)
        fig = plt.figure()
        plot_size = math.ceil(math.sqrt(len(masked_history)))
        for i, mimg in enumerate(self.masked_history):
            fig.add_subplot(plot_size,plot_size,i+1)
            plt.imshow(mimg.astype(np.uint8))
        fig.savefig('./result_new/'+str(self.type_pred)+'_.png')
        plt.close(fig)
        # with open('./result/history'+str(tag)+'.bin', 'wb') as f:
        #     pickle.dump(self.pixel_history, f)
        #     pickle.dump(self.cost_history, f)


def k_largest_index_argsort(a_ori, k, slice_map=None): 
    a = a_ori.copy()
    if slice_map is not None:
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    a[i,j] = -float("inf")
    idx = np.argsort(a.ravel())[:-k-1:-1] 
    return np.column_stack(np.unravel_index(idx, a.shape)) 

def k_smallest_index_argsort(a_ori, k, slice_map=None):
    a = a_ori.copy()
    if slice_map is not None:
        for i in range(slice_map.shape[0]):
            for j in range(slice_map.shape[1]):
                if slice_map[i,j] == 0:
                    a[i,j] = float("inf")
    idx = np.argsort(a.ravel())[:k] 
    return np.column_stack(np.unravel_index(idx, a.shape)) 



def main():
    imgs_dir = "./img/"
    tf.reset_default_graph()
    # image_classifier = ImageClassifier()
    # for img_name in os.listdir(imgs_dir):
    #     img = load_image(os.path.join(imgs_dir, img_name))
    #     img = cv2.resize(img, (640, 480))
    #     print(img_name)
    #     image_classifier.classify(img)
    
    img_name = "plane2.jpeg"
    it = 12
    image_classifier = ImageClassifier()
    img = load_image(imgs_dir+img_name)
    img = cv2.resize(img, (640, 480))
    print(img.shape)
    
    image_classifier.classify_loud(img)
    image_classifier.build_middle(img)
    
    #0117A
    """
    print('delete first row')
    slice_test = np.ones(image_classifier.m.shape[1:3])
    slice_test[0, :] = 0
    
    image_classifier.classify_sliced_loud (img, slice_test) 
    infer_test = image_classifier.infer_map_build(image_classifier.m, slice_test, dynamic_window=False)

    fig2 = plt.figure()
    fig2.add_subplot(3, 1,1)
    infer_test = cv2.resize(infer_test, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(infer_test, cmap='gray')

    #fig.add_subplot(3, 1, 2)
    #plt.imshow(thresholding(infer_test), cmap='gray')

    fig2.add_subplot(3,1,3)
    slice_test = cv2.resize(slice_test, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(slice_test, cmap='gray')
    
    plt.show()
    exit(0)
    #0117Ae
    """
    def iterate (img, times=None):
        infer_map_list = []
        slice_map_list = []
        slice_map = np.ones(image_classifier.m.shape[1:3])
        
        #0117A
        """
        slice_map[0,:] = 0 
        delta = 35
        new_cnt =35
        """
        total = slice_map.shape[0] * slice_map.shape[1]
        delta = 0
        new_cnt = 0 
        ti = 0
        second = False
        while True:
        #for tm in range(times):
            ti += 1
            print('########## %d iteration is ongoing  ########' % ti)
            if second:
                infer_map = slice_map.copy()
                print('     ##### It\'s second chance      #####    ') 
            else:
                infer_map = image_classifier.infer_map_build(image_classifier.m, slice_map, dynamic_window=False)
            
            min_indices = k_smallest_index_argsort(infer_map, infer_map.shape[0]*infer_map.shape[1]-new_cnt, slice_map)
            if not second:
                gmin = infer_map[min_indices[0][0], min_indices[0][1]]
                print('min val of infer_2 is', gmin)
                for i in range(infer_map.shape[0]):
                    for j in range(infer_map.shape[1]):
                        if slice_map[i,j] != 0:
                            infer_map[i,j] -= gmin

                gmax = np.amax(infer_map)
                infer_map /= gmax/255
            else:
                infer_map /= 1/255
                np.random.shuffle(min_indices)
            cnt = new_cnt
            new_cnt = 0
            ind = 0
            cri = 0
            escape = False
            tt = time.time()
            for i,j in min_indices:
                #image_classifier.writes()
                slice_map[i,j] = 0
                cnt += 1
                print('checking....', ind, time.time() - tt)
                tt = time.time()
                # if infer_map[i,j] != infer_map[min_indices[cri][0], min_indices[cri][1]]:
                #     cri = ind
                #     if escape:
                #         cnt -=1
                #         new_cnt = cnt
                #         slice_map[i,j] =1
                #         slice_map_list.append(slice_map.copy())
                #         print('EMRGENCY ESCAPE')
                #         break
                # if cnt % 100 == 0:
                #     print('####### for %d pixels .... ' % cnt)
                ind += 1
                if image_classifier.classify_middle_sliced(slice_map) != image_classifier.type_pred:
                    
                    #print('delta-cnt:   ', cnt-delta)
                    #print('ind and len: ', ind, len(min_indices))
                    #print(cri, infer_map[min_indices[cri][0], min_indices[cri][1]])
                    #if (ind < len(min_indices)):
                    #     print(ind, infer_map[min_indices[ind][0], min_indices[ind][1]])
                    if (cnt-delta == 1) and ind < len(min_indices) and infer_map[min_indices[0][0], min_indices[0][1]] == infer_map[min_indices[ind][0], min_indices[ind][1]]:
                        cnt -= 1
                        #new_cnt = 0
                        slice_map[i,j] =1
                        print('ONE MORE TIME')
                        escape = True
                        print('delta-cnt:   ', cnt-delta)
                        print('ind and len: ', ind, len(min_indices))
                        print('min_indices are: ', i, j)
                        continue
                    if new_cnt == 0:
                        new_cnt = cnt-1
                        
                        #branch_map = slice_map.copy()
                        #draw1 = slice_map1.copy()
                        slice_map[i,j] =1
                    print('IM FREE')
                    break
                    #print('feels bad man')
                    # break
            escape=False
            if new_cnt - delta >= 0:
                slice_map_list.append(slice_map.copy())
                infer_map_list.append(infer_map.copy())
            print('sliced_over      '+str(new_cnt)+' / '+str(total)+' pixels')
            print('delta =  ', new_cnt - delta)
            print('   @@@@@@@@@@@ stats @@@@@@@@@')
            _ = 0
            for i in range(slice_map.shape[0]):
                for j in range(slice_map.shape[1]):
                    if slice_map[i,j] == 0:
                        _ += 1
            print('   @   min_indices    ', min_indices[0][0], min_indices[0][1], '@')
            print('   @   new_cnt        ', new_cnt, '  @')
            print('   @   sliced_pix     ', _)
            print('   @@@@@@@@@@@@@@@@@@@@@@@@@@')
            if times is None:
                if new_cnt-delta == 0:
                    if not second:
                    #if len(min_indices) > 1 and (infer_map[min_indices[0][0], min_indices[0][1]] == infer_map[min_indices[1][0], min_indices[1][1]]):
                        print('+++++++++++++ Second chance  +++++++++')
                        second = True
                        continue
                        #break
                    else:
                        slice_map_list.append(slice_map.copy())
                        infer_map_list.append(infer_map.copy())
                        break
            else:
                if ti >= times:
                    slice_map_list.append(slice_map.copy())
                    infer_map_list.append(infer_map.copy())
                    break
            delta = new_cnt
            second = False

            
            if new_cnt > 1250:
                print('simai')
                slice_map_list.append(slice_map.copy())
                infer_map_list.append(infer_map.copy())
                break
            # for i in range(infer.shape[0]):
            #     for j in range(infer.shape[1]):
            #         if infer[i,j] <0.05 :
            #             slice_map1[i,j] = 0
        return infer_map_list, slice_map_list
    
    inf, draw = iterate(img)
    #it = ti
    fig = plt.figure()
    it = len(inf)
    
    print('FINAL INFER: ')
    image_classifier.classify_sliced_loud(img, draw[-1])
    #new_map = np.zeros((640,480))

    def thresholding (gray, threshold=0):
        new_map = np.zeros(gray.shape)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i,j] > threshold:
                    new_map[i,j] = 1
        return new_map
    _ = 0
    for i in range(draw[-1].shape[0]):
        for j in range(draw[-1].shape[1]):
            if draw[-1][i,j] == 0:
                _ += 1
    print(len(inf), len(draw), _)
    for i in range(it):
        fig.add_subplot(10, (it//10)+1, i+1)
        tempshot = cv2.resize(inf[i], (640,480), interpolation=cv2.INTER_AREA)
        plt.imshow(tempshot, cmap='gray')
    plt.show()




    fig = plt.figure()
    
    row = 5
    if row > it:
        row = it
    fig.add_subplot(3,row+2, 1)
    plt.imshow(img.astype(np.uint8))
    for i in range(row):
        fig. add_subplot(3, row+2, i+2)
        inf[(it//row)*(i)] = cv2.resize(inf[(it//row)*(i)], (640,480), interpolation=cv2.INTER_AREA)
        plt.imshow(inf[(it//row)*(i)], cmap='gray')
        
        fig.add_subplot(3, row+2, row+i+4)
        plt.imshow(thresholding(inf[(it//row)*(i)]), cmap='gray')

        fig. add_subplot(3, row+2, row+row+i+6)
        draw[(it//row)*(i)] = cv2.resize(draw[(it//row)*(i)], (640,480), interpolation=cv2.INTER_AREA)
        plt.imshow(draw[(it//row)*(i)], cmap='gray')
    
    fig.add_subplot(3, row+2, row+2)
    inf[-1] = cv2.resize(inf[-1], (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(inf[-1], cmap='gray')

    fig.add_subplot(3, row+2, 2*(row+2))
    plt.imshow(thresholding(inf[-1]), cmap='gray')

    fig. add_subplot(3, row+2, 3*(row+2))
    draw[-1] = cv2.resize(draw[-1], (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(draw[-1], cmap='gray')
    #fig.add_subplot(2,3,3)
    #slice_map1 = cv2.resize(slice_map1, (640,480), interpolation=cv2.INTER_AREA)
    #plt.imshow(slice_map1, cmap='gray')

    plt.show()
    #cv2.imwrite('lenagray.jpeg', infer)
    #image_classifier.slice_minimum(image_classifier.m)
    #image_classifier.result_show()    
    #print(image_classifier.middle_loss(image_classifier.m, 283))
    #image_classifier.slice(img)
    image_classifier.writer.close()

if __name__== "__main__":
    main()
