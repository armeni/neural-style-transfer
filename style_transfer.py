import time
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import save_img
from losses import calc_content_loss, calc_style_loss, calc_variation_loss
from utils import *

content = 'stata'
style = 'wave'
content_img = 'img/' + content + '.jpg'
style_img = 'img/' + style + '.jpg'

iterations = 15
style_weight = 1.0
content_weight = 0.5
variation_weight = 0.2
width, height = load_img(content_img).size
imgh = 400
imgw = int(width * imgh / height)

content_input, style_input, generated_input = inputs(content_img, style_img, imgh, imgw)
input_tensor = K.concatenate([content_input, style_input, generated_input], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, include_top=False)

outputs_dict = {layer.name: layer for layer in model.layers}
content_loss = calc_content_loss(outputs_dict, ['block5_conv2'])
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
style_loss = calc_style_loss(outputs_dict, style_layers, imgh, imgw)
variation_loss = calc_variation_loss(generated_input)
loss = content_weight * content_loss + style_weight * style_loss + variation_weight * variation_loss

grads = K.gradients(loss, generated_input)[0]
f_outputs = K.function([generated_input], [loss, grads])


def eval_loss_and_grads(x):
    x = x.reshape((1, imgh, imgw, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()
x = preprocess(content_img, imgh, imgw)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxiter=20)
    print('Current loss value: ', min_val)
    img = deprocess(x.copy(), imgh, imgw)
    fname = 'results/' + content + '_' + style + '_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))