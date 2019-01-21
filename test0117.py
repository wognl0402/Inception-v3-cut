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
from slice_function_0117 import *

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
                    masked_middle[0,i,j,:] = 0
                    #for k in range(masked_middle.shape[3]):
                        #del_middle(masked_middle[0,:,:,k], i, j, 0) 
                    #del_middle(masked_middle, i, j, 0)

        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        

        for i in range(1):
            index = sorted_inds[i]
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
                    masked_middle[0,i,j,:] = 0
                    #for k in range(masked_middle.shape[3]):
                        #del_middle(masked_middle[0,:,:,k], i, j, 0) 
                    #del_middle(masked_middle, i, j, 0)

        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        

        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))

        #return sorted_inds[0]

    def classify_sliced_loud_0117 (self, img, slice_map):
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
                    #del_middle(masked_middle, i, j, 0)

        feed_dict_m = {self.middle: masked_middle}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict_m)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
        
        return probabilities[self.type_pred]

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
                    masked_middle[0,i,j,:] = 0
                    #for k in range(masked_middle.shape[3]):
                        #del_middle(masked_middle[0,:,:,k], i, j, 0) 
                    #del_middle(masked_middle, i, j, 0)

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

    def infer_map_build_single (self, middle, slice_map):
        self.cost_val = self.middle_loss(middle)

        a = middle.shape
        infer_map = np.zeros(a[1:3])
       
        for i in range(a[1]):
            for j in range(a[2]):
                if slice_map[i,j] == 0:
                    infer_map[i,j] = 0
                    continue
                temp_middle = middle.copy()
                temp_middle[0,i,j,:] = self.del_map[0,i,j,:] 
                #for k in range(a[3]):
                    #del_middle(temp_middle[0,:,:,k],i,j,self.window_size)
                delta = self.middle_loss_sliced(temp_middle, slice_map) - self.cost_val
                infer_map[i,j] = delta
        return infer_map
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
            return self.infer_map_build(middle, slice_map)
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

    def build_del_map (self):
        img = np.zeros((480,640,3))
        self.del_map = self.session.run(self.cnet, feed_dict={self.image: img})
        f = open("del_map.txt", 'w')
        for i in range(self.del_map.shape[0]):
            for j in range(self.del_map.shape[1]):
                f.write("|")
                for k in range(self.del_map.shape[2]):
                    data = self.del_map[i,j,k]
                    f.write(str(data))
                f.write("| ")
            f.write("\n")
        f.close

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
    #tf.reset_default_graph()
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
    image_classifier.build_del_map()
    print('lets see black')
    testimg = np.zeros((480,640,3))
    image_classifier.classify_loud(testimg)
    print('lets see the del_map')
    image_classifier.classify_middle(image_classifier.del_map)
    blind = np.zeros(image_classifier.del_map.shape)
    print('lets see blind')
    image_classifier.classify_middle(blind)


    print('lets see white')
    testimg = np.zeros((480,640,3))
    for i in range(480):
        for j in range(640):
            testimg[i,j] = [0,255,255]
    image_classifier.classify_loud(testimg)
    print('lets see the white del_map')
    image_classifier.build_middle(testimg)
    image_classifier.classify_middle(image_classifier.m)
    print('lets see the light')
    light = np.ones (image_classifier.del_map.shape)
    image_classifier.classify_middle(light)

    print('lets see the noise')
    a = image_classifier.m.shape
    noise = np.random.rand(a[0], a[1],a[2],a[3])
    image_classifier.classify_middle(noise)
    exit(0)
    def del_things (middle, x, y, height, width):
        #side = window//2
        w = width//2
        h = height//2
        xmax = middle.shape[0]
        ymax = middle.shape[1]
        xlh = max(x-h, 0)
        xrh = min(x+h+1, xmax)

        ylh = max(y-w, 0)
        yrh = min(y+w+1, ymax)

        middle[xlh:xrh, ylh:yrh] = 0 
        #image_classifier.del_map[xlh:xrh, ylh:yrh] 

    image_classifier.build_middle(img)
    slice_map = np.ones(image_classifier.m.shape[1:3])
    slice_map_del = slice_map.copy()
    slice_map_del[0, :] = 0
    #del_things(slice_map_del, 0,15, 0, 70 )
    image_classifier.classify_sliced_loud (img, slice_map_del)
    #exit(0)
    param_x = 0 
    param_y = 10 

    score_map = np.zeros(slice_map.shape)
    loss_map = np.zeros(slice_map.shape)

    ori_loss = image_classifier.middle_loss (image_classifier.m)

    for i in range(slice_map.shape[0]):
        for j in range(slice_map.shape[1]):
            slice_map_del = slice_map.copy()
            del_things (slice_map_del, i, j, param_x, param_y)
            print ("deleting ", i, j, "pixels")
            score_map[i,j] = image_classifier.classify_sliced_loud_0117 (img, slice_map_del)
            loss_map[i,j] = image_classifier.middle_loss_sliced (image_classifier.m, slice_map_del) - ori_loss
            print(score_map[i,j])
    slice_map_del = slice_map.copy()
    slice_map_del[0, :] = 0
    #del_things(slice_map_del, 0,15, 0, 70 )
    image_classifier.classify_sliced_loud (img, slice_map_del)
    
    gmax = np.amax(score_map)
    score_map /= gmax/255

    gmin = np.amin(loss_map)
    loss_map -= gmin
    gmax = np.amax(loss_map)
    loss_map /= gmax/255

    fig = plt.figure()
    fig.add_subplot(3,1,1)
    plt.imshow(img.astype(np.uint8))

    fig.add_subplot(3,1,2)
    #score_map = cv2.resize(score_map, (640,480), interpolation=cv2.INTER_AREA)
    plt.imshow(score_map, cmap='gray')

    fig.add_subplot(3,1,3)
    plt.imshow(loss_map, cmap='gray')
    plt.show()
if __name__== "__main__":
    main()
