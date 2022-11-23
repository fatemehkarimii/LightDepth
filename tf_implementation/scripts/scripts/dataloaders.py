import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import MaxPooling2D
import os

class OrdinaryDataloader(object):
    def __init__(self, config,is_training=True,debug=False):
        self.do_flip = config.do_flip
        self.do_augment = config.do_augment
        self.do_rotate = config.do_rotate 
        self.do_kb_crop = config.do_kb_crop
        self.use_normalized_image = config.use_normalized_image
        self.is_training = is_training
        self.debug = debug

        if self.is_training:
            self.multiple_strategy = config.multiple_strategy
        else:
            self.multiple_strategy = False
        self.current_strategy = 0
        self._early_stopping_patience = config._early_stopping_patience
        self.degree = np.deg2rad(tf.constant(config.rotation_degree,dtype=tf.float16)) 
        self.height = config.input_height
        self.width = config.input_width
        self.input_data_path = config.input_data_path
        self.groundtruth_data_path = config.groundtruth_data_path
        self.strategies = config.strategies
        
        if self.is_training:
            filenames = list(open(config.train_filenames_file))
            self.batch_size = config.train_batch_size
        else:
            filenames = list(open(config.test_filenames_file))
            self.batch_size = config.test_batch_size

        if config.train_only_on_the_first_image:
            filenames = [tf.identity(filenames[0]) for i in range(8)]
        self.num_elements = len(filenames)

        self.loader = tf.data.Dataset.from_tensor_slices(filenames)

        if self.is_training:
            if not self.debug:
                self.loader = self.loader.shuffle(self.num_elements,
                                                reshuffle_each_iteration=True)
            self.loader = self.loader.repeat()
            self.loader = self.loader.map(self.parse_function)
            self.loader = self.loader.map(self.train_preprocess)
            self.loader = self.loader.map(
                    lambda x,y: tf.py_function(
                        self.lazy_preprocess,
                        [x,y],
                        [tf.float32,
                        tf.float32]))
        else:
            self.loader = self.loader.map(self.parse_function)
            self.loader = self.loader.map(self.test_preprocess)

        self.loader = self.loader.batch(self.batch_size).prefetch(2)
    
    @property
    def early_stopping_patience(self):
        if self.multiple_strategy:
            return self._early_stopping_patience[self.current_strategy]
        else:
            return self._early_stopping_patience[-1]

    @property
    def num_strategies(self):
        return len(self.strategies)

    def parse_function(self, line):
        paths = tf.strings.split(line) 

        image = tf.image.decode_png(tf.io.read_file(self.input_data_path+paths[0]))
        image = tf.image.convert_image_dtype(image, tf.float32)
        depth_gt = tf.image.decode_png(tf.io.read_file(self.groundtruth_data_path+paths[1]),
                                                            channels=0,
                                                            dtype=tf.uint16)
        depth_gt = tf.cast(depth_gt, tf.float32) / 256.0

        if self.do_kb_crop:
            print('Cropping training images as kitti benchmark images')
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            top_margin = tf.cast(height - 352,dtype=tf.int32)
            left_margin = tf.cast((width - 1216) / 2,dtype=tf.int32)
            depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
        
        return image, depth_gt

    def train_preprocess(self, image,depth_gt):
        if self.do_rotate:
            print('Rotating training images')
            random_angle = tf.random.uniform([], - self.degree, self.degree)
            image = tfa.image.rotate(image, random_angle, interpolation='nearest')
            depth_gt = tfa.image.rotate(depth_gt, random_angle, interpolation='nearest')

        image, depth_gt = self.crop_fixed_size(image, depth_gt)

        if self.do_flip:
            do_flip = tf.random.uniform([], 0, 1)
            image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
            depth_gt = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth_gt), lambda: depth_gt)
        
        if self.do_augment:
            do_augment = tf.random.uniform([], 0, 1)
            image = tf.cond(do_augment > 0.5, lambda: self.augment_image(image), lambda: image)

        image.set_shape([self.height, self.width, 3])
        depth_gt.set_shape([self.height, self.width, 1])

        if self.use_normalized_image:
            image *= 255.0 
            image = self.mean_image_subtraction(image, 
                                                [123.68, 116.78, 103.94])

        return image, depth_gt

    def test_preprocess(self,image,depth_gt):
        image.set_shape([None, None, 3])
        depth_gt.set_shape([None, None, 1])

        if self.use_normalized_image:
            image *= 255.0 
            image = self.mean_image_subtraction(image, 
                                                [123.68, 116.78, 103.94])
        return image, depth_gt

    def lazy_preprocess(self, image, depth_gt):
        if self.multiple_strategy:
            strategy = self.strategies[self.current_strategy]
        else:
            strategy = self.strategies[-1]
        depth_gt = self.dilation(depth_gt,**strategy)
        return image, depth_gt

    def dilation(self, depth_gt, pool_size=(2,2), iterations=1):
        if iterations > 0:
            depth_gt = tf.expand_dims(depth_gt, axis=0)
            for _ in range(iterations):
                depth_gt = MaxPooling2D(pool_size=pool_size)(depth_gt)
            depth_gt = tf.squeeze(depth_gt,axis=0)
        depth_gt = tf.image.resize(depth_gt,
                                   (self.height,self.width),
                                   method='nearest')

        return depth_gt

    def crop_fixed_size(self, image, depth_gt):
        image_depth = tf.concat([image, depth_gt], 2)
        if not self.debug:
            image_depth_cropped = tf.image.random_crop(image_depth, [self.height, self.width, 4])
        else:
            image_depth_cropped = image_depth[100:100+self.height, 365:365+self.width, :]
        image_cropped = image_depth_cropped[:, :, 0:3]
        depth_gt_cropped = tf.expand_dims(image_depth_cropped[:, :, 3], 2)

        return image_cropped, depth_gt_cropped

    def augment_image(self, image):
        # gamma augmentation
        gamma = tf.random.uniform([], 0.9, 1.1)
        image_aug = image ** gamma

        
        brightness = tf.random.uniform([], 0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = tf.random.uniform([3], 0.9, 1.1)
        white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        color_image = tf.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image

        return image_aug

    @staticmethod
    def mean_image_subtraction(image, means):
        """Subtracts the given means from each image channel.
        For example:
          means = [123.68, 116.779, 103.939]
          image = mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.
        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.
        Returns:
          the centered image.
        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """

        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError(f'len(means)==3 must match the number of channels == {num_channels}')

        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

        