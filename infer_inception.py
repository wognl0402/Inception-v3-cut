import os

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import imagenet
from nets import inception_v4
from preprocessing import inception_preprocessing


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
        self.image_size = inception_v4.inception_v4.default_image_size
        self.checkpoints_dir = 'checkpoints'
        self.names = imagenet.create_readable_names_for_imagenet_labels()
        self.arg_scope = inception_v4.inception_v4_arg_scope()

        self.image = tf.placeholder(tf.uint8, [480, 640, 3])

        self.processed_image = inception_preprocessing.preprocess_image(self.image,
                                                                        self.image_size, self.image_size,
                                                                        is_training=False)
        self.processed_images = tf.expand_dims(self.processed_image, 0)

        # processed_images will be a 1x299x299x3 tensor of float32

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with self.slim.arg_scope(self.arg_scope):
            self.logits, self.end_points = inception_v4.inception_v4(self.processed_images, num_classes=1001,
                                                                     is_training=False)
            self.probs = tf.nn.softmax(self.logits)

        self.init_fn = self.slim.assign_from_checkpoint_fn(
            os.path.join(self.checkpoints_dir, 'inception_v4.ckpt'),
            self.slim.get_model_variables('InceptionV4'))

        self.session = tf.Session()
        self.init_fn(self.session)

    def classify(self, img):
        height, width = img.shape[:2]

        feed_dict = {self.image: img}
        probabilities = self.session.run(self.probs, feed_dict=feed_dict)
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))


def main():
    imgs_dir = "./cat.jpeg"
    image_classifier = ImageClassifier()
    for img_name in os.listdir(imgs_dir):
        img = load_image(os.path.join(imgs_dir, img_name))
        img = cv2.resize(img, (640, 480))
        print(img_name)
        image_classifier.classify(img)

if __name__== "__main__":
    main()