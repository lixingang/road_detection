# -*- coding: utf-8 -*-
import numpy as np

from keras.models import Model
from keras import backend as K

import utils_vis
from utils import *
from segmentation_models import Unet
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import yaml
def conv_output(model, layer_name, img):
    """Get the output of conv layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.
    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]


def conv_filter(model, layer_name, img):
    """Get the filter of conv layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.
    Returns:
           filters.
    """
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    try:
        layer_output = layer_dict[layer_name].output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    kept_filters = []
    for i in range(layer_output.shape[-1]):
        loss = K.mean(layer_output[:, :, :, i])
        # compute the gradient of the input picture with this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = utils_vis.normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.
        # run gradient ascent for 20 steps
        fimg = img.copy()

        for j in range(40):
            loss_value, grads_value = iterate([fimg])
            fimg += grads_value * step

        # decode the resulting input image
        fimg = utils_vis.deprocess_image(fimg[0])
        kept_filters.append((fimg, loss_value))

        # sort filter result
        kept_filters.sort(key=lambda x: x[1], reverse=True)

    return np.array([f[0] for f in kept_filters])


def output_heatmap(model, last_conv_layer, img):
    """Get the heatmap for image.
    Args:
           model: keras model.
           last_conv_layer: name of last conv layer in the model.
           img: processed input image.
    Returns:
           heatmap: heatmap.
    """
    # predict the image class
    preds = model.predict(img)
    # find the class index
    index = np.argmax(preds[0])
    # This is the entry in the prediction vector
    target_output = model.output[:, index]

    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

    # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads
    # given the input picture
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


if __name__ == '__main__':


    img_path = '/home/lxg/data/road/spacenet/vegas/images_rgb/RGB-PanSharpen_AOI_2_Vegas_img1201.tif'
    # layer_name = 'concatenate_5' # edge
    # last_conv_layer = 'concatenate_5' # edge

    
    layer_name_list = [
        'e_output_0_16', 'e_output_1_16','e_output_2_16','e_output_3_16','e_fuse_sig','concatenate_5',
        'r_output_0_16', 'r_output_1_16','r_output_2_16','r_output_3_16','r_fuse_sig','concatenate_6',
        'u_outputs_sig','f_outputs_sig','concatenate_7',

        'decoder_stage1_relu2','decoder_stage2_relu2','decoder_stage3_relu2','decoder_stage4_relu2',
        'u_outputs_fea',
        'fuse_dir'
                      ]  # region
    col_list = [
        4,4,4,4,1,2,
        4,4,4,4,1,2,
        1,1,4,
        4,4,4,4,
        4,
        1,
    ]
    # layer_name_list = [
    #     'f_outputs_sig',
    #
    #                   ]  # region
    # col_list = [
    #     1
    # ]
    # last_conv_layer = 'conv2d_15'  # region

    with open("config.yaml") as pfile:
        cfgs = yaml.load(pfile)
    model = Unet(backbone_name='resnet50')
    model.load_weights(os.path.join(cfgs["SAVE_DIR"], "weights", "epoch55.h5"))
    print("RETORE SUCCESSFULLY!")

    img, pimg = utils_vis.read_img(img_path, (512, 512))
    for  i, layer_name in enumerate(layer_name_list):
        cout = conv_output(model, layer_name, pimg)
        utils_vis.vis_conv(cout, col_list[i], layer_name, 'conv')

    # pimg = np.random.random((1, 512, 512, 3)) * 20 + 128.
    # fout = conv_filter(model, layer_name, pimg)
    # utils_vis.vis_conv(fout, 2, layer_name, 'filter')

    # heatmap = output_heatmap(model, last_conv_layer, pimg)
    # utils_vis.vis_heatmap(img, heatmap)