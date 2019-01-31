from keras import backend as K


def calc_content_loss(layer_dict, content_layer_names):
    loss = 0
    for name in content_layer_names:
        layer = layer_dict[name]
        content_features = layer.output[0, :, :, :]
        generated_features = layer.output[2, :, :, :]
        loss += K.sum(K.square(generated_features - content_features))
    return loss / len(content_layer_names)


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features))


def style_loss_f(style, generated, imgh, imgw):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    channels = 3
    return K.sum(K.square(S - G)) / 4.0 * (channels ** 2) * ((imgh * imgw) ** 2)


def calc_style_loss(layer_dict, style_layer_names, imgh, imgw):
    loss = 0
    for name in style_layer_names:
        layer = layer_dict[name]
        style_features = layer.output[1, :, :, :]
        generated_features = layer.output[2, :, :, :]
        loss += style_loss_f(style_features, generated_features, imgh, imgw)
    return loss / len(style_layer_names)


def calc_variation_loss(x):
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
