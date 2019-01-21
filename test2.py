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
                    del_middle(masked_middle, i, j, 0)

        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        

        for i in range(1):
            index = sorted_inds[i]
            #print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))

        return sorted_inds[0]

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
                    del_middle(masked_middle, i, j, 0)

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

def iterate (img, times):
    infer_map_list = []
    slice_map_list = []
    slice_map = np.ones(image_classifier.m.shape[1:3])
    new_cnt = 0
    for tm in range(times):

        infer_map = image_classifier.infer_map_build(image_classifier.m, slice_map, dynamic_window=False)
        infer_map_list.append(infer_map.copy())

        min_indices = k_smallest_index_argsort(infer_map, infer.shape[0]*infer.shape[1]-new_cnt, slice_map)

        gmin = infer_map[min_indices[0][0], min_indices[0][1]]
        print('min val of infer_2 is', gmin)
        for i in range(infer_map.shape[0]):
            for j in range(infer_map.shape[1]):
                if slice_map[i,j] != 0:
                    infer_map[i,j] -= gmin

        gmax = np.amax(infer_map)
        infer_map /= gmax/255

        cnt = new_cnt
        new_cnt = 0
        for i,j in min_indices:
            slice_map[i,j] = 0
            cnt += 1
            if cnt % 50 == 0:
                print('####### for %d pixels .... ' % cnt)
            if image_classifier.classify_sliced(img, slice_map) != image_classifier.type_pred:
                if new_cnt == 0:
                    new_cnt = cnt-1
                    slice_map_list.append(slice_map.copy())
                    #branch_map = slice_map.copy()
                    #draw1 = slice_map1.copy()
                    slice_map[i,j] =1
                break
                #print('feels bad man')
                # break
        print('sliced_over      '+str(new_cnt)+' / '+str(cnt)+' pixels')

        
        # for i in range(infer.shape[0]):
        #     for j in range(infer.shape[1]):
        #         if infer[i,j] <0.05 :
        #             slice_map1[i,j] = 0
    return infer_map_list, slice_map_list

def main():
    imgs_dir = "./img"
    #tf.reset_default_graph()
    # image_classifier = ImageClassifier()
    # for img_name in os.listdir(imgs_dir):
    #     img = load_image(os.path.join(imgs_dir, img_name))
    #     img = cv2.resize(img, (640, 480))
    #     print(img_name)
    #     image_classifier.classify(img)

    img_dir = "./img/catcut.jpeg"
    image_classifier = ImageClassifier()
    img = load_image(img_dir)
    img = cv2.resize(img, (640, 480))
    print(img.shape)
    image_classifier.classify_loud(img)
    image_classifier.build_middle(img)
    infer = image_classifier.infer_map_build(image_classifier.m, np.ones(image_classifier.m.shape[1:3]), dynamic_window=False)

    gmin = np.amin(infer)
    infer -= gmin

    gmax = np.amax(infer)
    infer /= gmax/255
    slice_map1 = np.ones(infer.shape)
    # for i in range(infer.shape[0]):
    #     for j in range(infer.shape[1]):
    #         if infer[i,j] <0.05 :
    #             slice_map1[i,j] = 0


    min_indices = k_smallest_index_argsort(infer, infer.shape[0]*infer.shape[1])
#    print(top10indexes)
    cnt = 0
    new_cnt = None
    for i,j in min_indices:
        slice_map1[i,j] = 0
        cnt += 1
        if cnt % 50 == 0:
            print('####### for %d pixels .... ' % cnt)
        if image_classifier.classify_sliced(img, slice_map1) != image_classifier.type_pred:
            if new_cnt is None:
                new_cnt = cnt-1
                branch_map = slice_map1.copy()
                draw1 = slice_map1.copy()
                branch_map[i,j] =1
            break
            #print('feels bad man')
            # break
    print('sliced_over      '+str(new_cnt)+' / '+str(cnt)+' pixels')
    infer_2 = image_classifier.infer_map_build(image_classifier.m, branch_map, dynamic_window=False)

    
    # for i in range(infer.shape[0]):
    #     for j in range(infer.shape[1]):
    #         if infer[i,j] <0.05 :
    #             slice_map1[i,j] = 0


    min_indices = k_smallest_index_argsort(infer_2, infer.shape[0]*infer.shape[1]-new_cnt, branch_map)

    gmin = infer_2[min_indices[0][0], min_indices[0][1]]
    print('min val of infer_2 is', gmin)
    for i in range(infer.shape[0]):
        for j in range(infer.shape[1]):
            if branch_map[i,j] != 0:
                infer_2[i,j] -= gmin

    gmax = np.amax(infer_2)
    infer_2 /= gmax/255
#    print(top10indexes)
    cnt = new_cnt
    new_cnt = None
    for i,j in min_indices:
        if branch_map[i,j] ==0:
            print('what?')
            break
        branch_map[i,j] = 0
        cnt += 1
        if cnt % 50 == 0:
            print('####### for %d pixels .... ' % cnt)
            
        if image_classifier.classify_sliced(img, branch_map) != image_classifier.type_pred:
            if new_cnt is None:
                new_cnt = cnt -1
                branch_map2 = branch_map.copy()
                draw2 = branch_map.copy()
                branch_map2[i,j] =1
            #print('faulted')
            break
    print('sliced_over      '+str(new_cnt)+' / '+str(cnt)+' pixels')
    infer_3 = image_classifier.infer_map_build(image_classifier.m, branch_map2, dynamic_window=False)


    min_indices = k_smallest_index_argsort(infer_3, infer.shape[0]*infer.shape[1]-new_cnt, branch_map2)

    gmin = infer_3[min_indices[0][0], min_indices[0][1]]
    for i in range(infer.shape[0]):
        for j in range(infer.shape[1]):
            if branch_map2[i,j] != 0:
                infer_3[i,j] -= gmin
    
    print('min val of infer_3 is', gmin)
    gmax = np.amax(infer_3)
    infer_3 /= gmax/255

    cnt = new_cnt
    new_cnt = None
    for i,j in min_indices:
        if branch_map2[i,j] ==0:
            print('what??')
            break
        branch_map2[i,j] = 0
        cnt += 1
        if cnt % 50 == 0:
            print('####### for %d pixels .... ' % cnt)
        if image_classifier.classify_sliced(img, branch_map2) != image_classifier.type_pred:
            if new_cnt is None:
                new_cnt = cnt-1
                draw3 = branch_map2.copy()
                branch_map3 = branch_map2.copy()
                branch_map3[i,j] = 1
            #print('faulted')
            break
    print('sliced_over      '+str(new_cnt)+' / '+str(cnt)+' pixels')
    infer_4 = image_classifier.infer_map_build(image_classifier.m, branch_map3, dynamic_window=False)


    min_indices = k_smallest_index_argsort(infer_4, infer.shape[0]*infer.shape[1]-new_cnt, branch_map3)

    gmin = infer_4[min_indices[0][0], min_indices[0][1]]
    for i in range(infer.shape[0]):
        for j in range(infer.shape[1]):
            if branch_map3[i,j] != 0:
                infer_4[i,j] -= gmin
    print('min val of infer_4 is', gmin)

    gmax = np.amax(infer_4)
    infer_4 /= gmax/255

    cnt = new_cnt
    new_cnt = None
    for i,j in min_indices:
        if branch_map3[i,j] ==0:
            print('what???')
            break
        branch_map3[i,j] = 0
        cnt += 1
        if cnt % 50 == 0:
            print('####### for %d pixels .... ' % cnt)
        if image_classifier.classify_sliced(img, branch_map3) != image_classifier.type_pred:
            if new_cnt is None:
                new_cnt = cnt -1
                draw4 = branch_map3.copy()
            #print('faulted')
            break
    if new_cnt is None:
        draw4 = branch_map3.copy()
    #image_classifier.classify_sliced(img, slice_map1)
    fig = plt.figure()
    fig.add_subplot(2,5,1)
    plt.imshow(img.astype(np.uint8))

    fig.add_subplot(2,5,7)
    draw1 = cv2.resize(draw1, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(draw1, cmap='gray')
    fig.add_subplot(2,5,2)
    infer = cv2.resize(infer, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(infer, cmap='gray')

    fig.add_subplot(2,5,3)
    infer_2 = cv2.resize(infer_2, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(infer_2, cmap='gray')

    fig.add_subplot(2,5,8)
    draw2 = cv2.resize(draw2, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(draw2, cmap='gray')

    fig.add_subplot(2,5,4)
    infer_3 = cv2.resize(infer_3, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(infer_3, cmap='gray')

    fig.add_subplot(2,5,9)
    draw3 = cv2.resize(draw3, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(draw3, cmap='gray')

    fig.add_subplot(2,5,5)
    infer_4 = cv2.resize(infer_4, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(infer_3, cmap='gray')

    fig.add_subplot(2,5,10)
    draw4 = cv2.resize(draw4, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(draw4, cmap='gray')
    #fig.add_subplot(2,3,3)
    #slice_map1 = cv2.resize(slice_map1, (640,480), interpolation=cv2.INTER_AREA)
    #plt.imshow(slice_map1, cmap='gray')

    plt.show()
    cv2.imwrite('lenagray.jpeg', infer)
    #image_classifier.slice_minimum(image_classifier.m)
    #image_classifier.result_show()    
    #print(image_classifier.middle_loss(image_classifier.m, 283))
    #image_classifier.slice(img)


if __name__== "__main__":
    main()
