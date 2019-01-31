from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import numpy as np
from keras import backend as K


def preprocess(image, imgh, imgw):
    img = load_img(image, target_size=(imgh, imgw))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess(img, imgh, imgw):
    img = img.reshape((imgh, imgw, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def inputs(content, style, imgh, imgw):
    content_arr = K.variable(preprocess(content, imgh, imgw))
    style_arr = K.variable(preprocess(style, imgh, imgw))
    generated_arr = K.placeholder(content_arr.shape)
    return content_arr, style_arr, generated_arr
