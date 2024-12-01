import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import gradients, sum as K_sum, square, mean
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from PIL import Image

class PConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super(PConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding, use_bias=False)
        self.mask_conv = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding, use_bias=False, trainable=False)
        self.bias = self.add_weight(shape=(self.filters,), initializer='zeros', trainable=True, name='bias')
        self.built = True

    def call(self, inputs):
        image, mask = inputs
        if image.shape[-1] != mask.shape[-1]:
            mask = tf.tile(mask, [1, 1, 1, image.shape[-1] // mask.shape[-1]])

        mask_output = self.mask_conv(mask)
        mask_output = tf.clip_by_value(mask_output, 0.0, 1.0)

        conv_output = self.conv(image * mask)
        normalized_output = tf.math.divide_no_nan(conv_output, mask_output + 1e-8)
        return normalized_output + self.bias, mask_output


def build_pconv_unet(input_shape=(720, 1280, 1), enable_batch_norm=True):
    inputs_image = Input(shape=input_shape, name='image_input')
    inputs_mask = Input(shape=input_shape, name='mask_input')

    def encoder_block(filters, kernel_size, inputs, enable_bn):
        conv, mask = PConv2D(filters, kernel_size)(inputs)
        if enable_bn:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv, mask
    def decoder_block(filters, kernel_size, inputs, skip_connection):
        conv, mask = PConv2D(filters, kernel_size)(inputs)
        skip_connection_adjusted = Conv2D(filters, kernel_size=1, padding='same')(skip_connection)
        conv = Concatenate()([conv, skip_connection_adjusted])
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv, mask

    e1, m1 = encoder_block(64, 7, [inputs_image, inputs_mask], enable_batch_norm)
    e2, m2 = encoder_block(128, 5, [e1, m1], enable_batch_norm)
    e3, m3 = encoder_block(256, 5, [e2, m2], enable_batch_norm)

    d3, m3 = decoder_block(128, 5, [e3, m3], e2)
    d2, m2 = decoder_block(64, 5, [d3, m3], e1)

    output, _ = PConv2D(1, 3)([d2, m2])

    model = Model(inputs=[inputs_image, inputs_mask], outputs=output)
    return model

def total_variation_loss(image):
    pixel_dif1 = image[:, 1:, :, :] - image[:, :-1, :, :]
    pixel_dif2 = image[:, :, 1:, :] - image[:, :, :-1, :]
    return tf.reduce_mean(tf.abs(pixel_dif1)) + tf.reduce_mean(tf.abs(pixel_dif2))

def perceptual_loss(vgg, predicted, ground_truth):
    predicted_3ch = tf.tile(predicted, [1, 1, 1, 3])
    ground_truth_3ch = tf.tile(ground_truth, [1, 1, 1, 3])
    
    features_pred = vgg(predicted_3ch)
    features_gt = vgg(ground_truth_3ch)
    return MeanAbsoluteError()(features_pred, features_gt)

def style_loss(vgg, predicted, ground_truth):
    def gram_matrix(features):
        shape = tf.shape(features)
        reshaped = tf.reshape(features, (shape[0], -1, shape[-1]))
        return tf.matmul(reshaped, reshaped, transpose_a=True)

    predicted_3ch = tf.tile(predicted, [1, 1, 1, 3])
    ground_truth_3ch = tf.tile(ground_truth, [1, 1, 1, 3])
    
    gram_pred = gram_matrix(vgg(predicted_3ch))
    gram_gt = gram_matrix(vgg(ground_truth_3ch))
    return MeanAbsoluteError()(gram_pred, gram_gt)

def combined_loss(vgg, alpha=1.0, beta=0.1, gamma=0.01):
    def loss_fn(ground_truth, predicted):
        pixel_loss = MeanAbsoluteError()(ground_truth, predicted)
        p_loss = perceptual_loss(vgg, predicted, ground_truth)
        s_loss = style_loss(vgg, predicted, ground_truth)
        tv_loss = total_variation_loss(predicted)

        tf.print("Pixel Loss:", pixel_loss, "Perceptual Loss:", p_loss, "Style Loss:", s_loss, "TV Loss:", tv_loss)
        
        return alpha * pixel_loss + beta * p_loss + gamma * (s_loss + tv_loss)
    
    return loss_fn

def combined_loss(vgg, alpha=1.0, beta=0.1, gamma=0.01):
    def loss_fn(ground_truth, predicted):
        pixel_loss = MeanAbsoluteError()(ground_truth, predicted)
        p_loss = perceptual_loss(vgg, predicted, ground_truth)
        s_loss = style_loss(vgg, predicted, ground_truth)
        tv_loss = total_variation_loss(predicted)
        return alpha * pixel_loss + beta * p_loss + gamma * (s_loss + tv_loss)
    return loss_fn

def train_model(yuv_dir, diff_map_dir):
    model_path = "/Artifact Removal/Models/pconv_model.h5"
    
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(720, 1280, 3))
    for layer in vgg.layers:
        layer.trainable = False
    
    train_dataset = YUVImageDataset(yuv_dir, diff_map_dir, batch_size=1, mode="train")
    test_dataset = YUVImageDataset(yuv_dir, diff_map_dir, batch_size=1, mode="test")
    
    model = build_pconv_unet(input_shape=(720, 1280, 1), enable_batch_norm=True)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=combined_loss(vgg))
    
#     model.summary()
    
    model.fit(x=train_dataset, epochs=50, steps_per_epoch=min(10000, len(train_dataset)), 
              validation_data=test_dataset, validation_steps=min(1000, len(test_dataset)), verbose=1)
    
    # Second training phase without batch normalization
    model = build_pconv_unet(input_shape=(720, 1280, 1), enable_batch_norm=False)
    model.compile( optimizer=Adam(learning_rate=0.00005), loss=combined_loss(vgg))
    
    model.fit(x=train_dataset, epochs=50, steps_per_epoch=min(10000, len(train_dataset)), 
              validation_data=test_dataset, validation_steps=min(1000, len(test_dataset)), verbose=1)
    
    model.save(model_path)
    return model

class YUVImageDataset(Sequence):
    def __init__(self, yuv_dir, diff_map_dir, batch_size, image_size=(720, 1280), mask_threshold=0.1, mode='train', train_ratio=0.7):
        self.yuv_dir = yuv_dir
        self.diff_map_dir = diff_map_dir
        self.batch_size = batch_size
        self.image_size = image_size  # (height, width)
        self.mask_threshold = mask_threshold
        self.mode = mode

        self.image_triplets = self._load_image_triplets()
        train_triplets, test_triplets = train_test_split(self.image_triplets, test_size=1 - train_ratio, random_state=42)
        self.image_triplets = train_triplets if self.mode == 'train' else test_triplets

    def _load_image_triplets(self):
        image_triplets = []
        for filename in os.listdir(self.yuv_dir):
            if filename.startswith('original_') and filename.endswith('.raw'):
                base_name = filename[len('original_'):-len('.raw')]
                original_path = os.path.join(self.yuv_dir, filename)
                denoised_filename = f'denoised_{base_name}.raw'
                denoised_path = os.path.join(self.yuv_dir, denoised_filename)
                diff_map_filename = f'difference_{base_name}.png'
                diff_map_path = os.path.join(self.diff_map_dir, diff_map_filename)
                if os.path.exists(denoised_path) and os.path.exists(diff_map_path):
                    image_triplets.append((original_path, denoised_path, diff_map_path))
        return image_triplets

    def __len__(self):
        return len(self.image_triplets) // self.batch_size

    def _load_y_channel(self, yuv_path):
        height, width = self.image_size
        y_size = width * height
        with open(yuv_path, 'rb') as f:
            y_data = np.frombuffer(f.read(y_size), dtype=np.uint8)
            y_channel = y_data.reshape((height, width))
        return y_channel / 255.0

    def _generate_mask(self, diff_map_path=None):
        height, width = self.image_size
        if diff_map_path is None:
            mask = np.ones((height, width), dtype=np.uint8)
            min_size = 10
            max_size = 200
            
            for _ in range(np.random.randint(5, 15)):
                w = np.random.randint(min_size, max_size)
                h = np.random.randint(min_size, max_size)
                x = np.random.randint(0, max(1, width - w))
                y = np.random.randint(0, max(1, height - h))
                mask[y:y+h, x:x+w] = 0
            return mask
        else:
            diff_map = Image.open(diff_map_path).convert('L')
            diff_map = diff_map.resize((width, height))
            mask = np.array(diff_map) / 255.0
            return np.where(mask > self.mask_threshold, 0, 1)

    def __getitem__(self, idx):
        batch_triplets = self.image_triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks, ground_truths = [], [], []
        
        for original_path, denoised_path, diff_map_path in batch_triplets:
            image = self._load_y_channel(original_path if self.mode == 'train' else denoised_path)
            ground_truth = self._load_y_channel(original_path)
            mask = self._generate_mask(diff_map_path if self.mode == 'test' else None)

            assert image.shape == mask.shape, f"Shape mismatch: {image.shape} vs {mask.shape}"

            images.append(image * mask)
            masks.append(mask)
            ground_truths.append(ground_truth)

        images = tf.convert_to_tensor(np.stack(images)[:, :, :, np.newaxis], dtype=tf.float32)
        masks = tf.convert_to_tensor(np.stack(masks)[:, :, :, np.newaxis], dtype=tf.float32)
        ground_truths = tf.convert_to_tensor(np.stack(ground_truths)[:, :, :, np.newaxis], dtype=tf.float32)

        return {"image_input": images, "mask_input": masks}, ground_truths

train_model("/Artifact Removal/Dataset/dataset_full", "/Artifact Removal/Dataset/detection_map_full")
