from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2D, Concatenate, LeakyReLU, UpSampling2D
import keras.backend as K
import keras.utils.conv_utils as conv_utils
from tensorflow.keras.models import Model
import numpy as np

def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class EfficientUNet():
    def __init__(self,config):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.initial_learning_rate,
            decay_steps=config.decay_steps,
            decay_rate=config.decay_rate)
        self.optimizer = getattr(tf.optimizers,config.optimizer)(learning_rate=lr_schedule)
        self.max_depth = tf.constant(config.max_depth)
        self.min_depth = config.min_depth
        self.model_loss = getattr(self,config.loss_fn)
        self.garg_crop = config.garg_crop
        self.eigen_crop = config.eigen_crop
        self.do_flip_predict = config.do_flip_predict
        self.eps = 1e-5
        def UpConv2D(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, encoder.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i
        
        encoder = getattr(applications.efficientnet,config.encoder)(input_shape=(None, None, 3), include_top=False)
        encoder_output_shape = encoder.output.shape
        decode_filters = int(encoder_output_shape[-1])
        decoder = Conv2D(filters=decode_filters,
                            kernel_size=1, padding='same',
                            input_shape=encoder_output_shape,
                            name='conv2')(encoder.output)
        decoder = UpConv2D(decoder, int(decode_filters/2),
                           'up1', concat_with='block4a_dwconv')
        decoder = UpConv2D(decoder, int(decode_filters/4),
                           'up2', concat_with='block3a_dwconv')
        decoder = UpConv2D(decoder, int(decode_filters/8),
                           'up3', concat_with='block2a_dwconv')
        decoder = UpConv2D(decoder, int(decode_filters/16),
                           'up4', concat_with='block1c_activation')
        # decoder = UpConv2D(decoder, int(decode_filters/32),
        #                    'up5', concat_with=encoder.input.name)
        outputs = Conv2D(filters=1, 
                         kernel_size=3, 
                         strides=1,
                         padding='same', 
                         name='conv3',
                         activation=config.decoder_last_layer_activation_fn)(decoder)
        outputs = UpSampling2D()(outputs)
        if config.decoder_last_layer_activation_fn == 'sigmoid':
            outputs=outputs*self.max_depth + self.eps
        else:
            outputs = outputs - tf.reduce_min(outputs)
            outputs = outputs / tf.reduce_max(outputs)
            outputs = (outputs*(self.max_depth-self.min_depth))+self.min_depth
        self.model = Model(inputs=encoder.input, outputs=outputs)

    @tf.function
    def test_step(self,image,depth_gt):
        depth_est = self.model(image, training=False)
        loss_value = self.model_loss(depth_est, depth_gt)
        return loss_value,depth_est

    @tf.function
    def train_step(self, image, depth_gt):
        with tf.GradientTape() as tape:
            depth_est = self.model(image, training=True)
            loss_value = self.model_loss(depth_est, depth_gt)

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value,tf.reduce_max(depth_est),tf.reduce_min(depth_est)

    def compute_metrics(self,image,depth_gt):
        valid_mask = np.logical_and(depth_gt > self.min_depth,
                              depth_gt < self.max_depth)

        if self.garg_crop or self.eigen_crop:
            batches, gt_height, gt_width, channels = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if self.garg_crop:
                eval_mask[:,int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width),:] = 1
            elif self.eigen_crop:
                # if self.dataset == 'kitti':
                eval_mask[:,int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                int(0.0359477 * gt_width):int(0.96405229 * gt_width),:] = 1
                # else:
                #     eval_mask[:,45:471, 41:601,:] = 1

        depth_est = self.model(image, training=False)
        if self.do_flip_predict:
            depth_est_lr = self.model(image[...,::-1,:], training=False)
            depth_est_final = (0.5*(depth_est + depth_est_lr))[valid_mask]
        else:
            depth_est_final = depth_est[valid_mask]
        depth_gt = depth_gt[valid_mask]

        thresh = np.maximum((depth_gt / depth_est_final), (depth_est_final / depth_gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(depth_gt - depth_est_final) / depth_gt)
        sq_rel = np.mean(((depth_gt - depth_est_final) ** 2) / depth_gt)

        rmse = (depth_gt - depth_est_final) ** 2
        rmse = np.sqrt(np.mean(rmse))

        rmse_log = (np.log(depth_gt) - np.log(depth_est_final)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(depth_est_final) - np.log(depth_gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        log_10 = (np.abs(np.log10(depth_gt) - np.log10(depth_est_final))).mean()
        return dict(a1=a1, a2=a2, a3=a3,
                    abs_rel=abs_rel, rmse=rmse, log_10=log_10,
                    rmse_log=rmse_log, silog=silog, sq_rel=sq_rel)

    @tf.function
    def bts_loss(self,depth_est,depth_gt):
        mask = depth_gt > self.min_depth

        depth_gt_masked = tf.boolean_mask(depth_gt, mask)
        depth_est_masked = tf.boolean_mask(depth_est, mask)
        
        d = tf.math.log(depth_est_masked) - tf.math.log(depth_gt_masked)
        return tf.sqrt(tf.reduce_mean(d ** 2) - 0.85 * (tf.reduce_mean(d) ** 2)) * 10.0
    
    @tf.function
    def kitti_loss(self,depth_est,depth_gt):
        mask = depth_gt > self.min_depth

        depth_gt_masked = tf.boolean_mask(depth_gt, mask)
        depth_est_masked = tf.boolean_mask(depth_est, mask)

        d = tf.math.log(depth_est_masked) - tf.math.log(depth_gt_masked)
        return tf.reduce_mean(d ** 2) - (tf.reduce_mean(d) ** 2)

    @tf.function
    def densedepth_loss(self,depth_est, depth_gt, theta=0.1, maxDepthVal=1000.0/10.0): 
    
        l_depth = K.mean(K.abs(depth_est - depth_gt), axis=-1)

        dy_true, dx_true = tf.image.image_gradients(depth_gt)
        dy_pred, dx_pred = tf.image.image_gradients(depth_est)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        l_ssim = K.clip((1 - tf.image.ssim(depth_gt, depth_est, maxDepthVal)) * 0.5, 0, 1)

        w1 = 1.0
        w2 = 1.0
        w3 = theta

        return tf.reduce_mean((w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)))
