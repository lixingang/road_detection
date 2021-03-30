from keras.layers import Conv2D,Conv2DTranspose,Reshape,UpSampling2D
from keras.layers import Activation,Lambda,Dense,concatenate,GlobalAveragePooling2D,multiply
from keras.models import Model
import keras.backend as K
from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..utils import get_layer_number, to_tuple
from math import pow

def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])
    

    # 
    # pixel-level
    # 

    e_outputs = [None] * n_upsample_blocks
    e_outputs_sig = [None] * n_upsample_blocks
    r_outputs = [None] * n_upsample_blocks
    r_outputs_sig = [None] * n_upsample_blocks
    r_outputs_ex = [None] *n_upsample_blocks
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):

            skip_connection = backbone.layers[skip_connection_idx[i]].output
            
            
            # edge-level
            stride = [16,8,4,2]

            e_outputs[i] = Conv2D(16, 2, padding='same',activation='relu')(skip_connection)
            e_outputs[i] = UpSampling2D(stride[i])(e_outputs[i])
            e_outputs[i] = Conv2D(8, 2, padding='same',activation='relu', name='e_output_{}_16'.format(i))(e_outputs[i])
            e_outputs[i] = Conv2D(1, 2, padding='same', activation='relu',
                                  name='e_outputs_{}'.format(i))(e_outputs[i])
            #e_outputs[i] = Conv2DTranspose(1, 3,strides=(stride[i],stride[i]),padding='same',name='e_outputs_{}'.format(i))(e_outputs[i])

            #print('e_outputs:',e_outputs[i])
            
            #region-level
#             r_outputs[i] = Conv2D(4,1,padding='same',activation='relu')(skip_connection)
#             r_outputs[i] = Conv2DTranspose(1, 3,strides=(stride[i],stride[i]),padding='same',name='r_outputs_{}'.format(i))(r_outputs[i])
            
            stride = [1,2,4,8]
            r_outputs[i] = Conv2D(16, stride[i],strides=(stride[i],stride[i]),padding='valid',activation='relu')(skip_connection)
            r_outputs[i] = Conv2D(8, 2, strides=(1,1), padding='same',activation='relu', name='r_output_{}_16'.format(i))(r_outputs[i])
            r_outputs[i] = Conv2D(1, 2,strides=(1,1),padding='same',activation='relu',name='r_outputs_{}'.format(i))(r_outputs[i])
            r_outputs_ex[i] = Lambda(lambda img: K.resize_images(img,16,16,'channels_last'))(r_outputs[i])
            
        upsample_rate = to_tuple(upsample_rates[i])
        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)



    u_outputs_fea = Conv2D(8, (3,3), padding='same', name='u_outputs_fea',activation='relu')(x)
    u_outputs = Conv2D(1, (1, 1), padding='same', name='final_conv',activation='relu')(u_outputs_fea)
    u_outputs_sig = Activation(activation, name='u_outputs_sig')(u_outputs)
    for i in range(len(skip_connection_idx)):
        e_outputs_sig[i] = Activation('sigmoid', name='e_outputs_{}_sig'.format(i))(e_outputs[i])
        r_outputs_sig[i] = Activation('sigmoid', name='r_outputs_{}_sig'.format(i))(r_outputs[i])

    e_fuse = concatenate([e_outputs[0],e_outputs[1],e_outputs[2],e_outputs[3]])
    e_fuse = Conv2D(1,2,padding='same',activation='relu',name='e_fuse')(e_fuse)
    e_fuse_sig = Activation('sigmoid', name='e_fuse_sig')(e_fuse)
    
    r_fuse = concatenate([r_outputs[0],r_outputs[1],r_outputs[2],r_outputs[3]])
    r_fuse = Conv2D(1,2,padding='same',activation='relu',name='r_fuse')(r_fuse)
    r_fuse_sig = Activation('sigmoid', name='r_fuse_sig')(r_fuse)
    r_fuse_ex = Lambda(lambda img: K.resize_images(img, 16, 16, 'channels_last'))(r_fuse)
    fuse = concatenate([
                u_outputs_fea,
                e_outputs[0],e_outputs[1],e_outputs[2],e_outputs[3],e_fuse,
                r_outputs_ex[0],r_outputs_ex[1],r_outputs_ex[2],r_outputs_ex[3],r_fuse_ex
                       ])
    # fuse_se = squeeze_excitation_layer(fuse,1)
    fuse_dir = direction_layer(fuse, "fuse_dir")
    f_outputs_sig = Conv2D(1,2,padding='same',activation='softmax',name='f_outputs_sig')(fuse)



    model = Model(
        inputs=input,
        outputs=[
            u_outputs_sig,
            e_outputs_sig[0],e_outputs_sig[1],e_outputs_sig[2],e_outputs_sig[3],e_fuse_sig,
            r_outputs_sig[0],r_outputs_sig[1],r_outputs_sig[2],r_outputs_sig[3],r_fuse_sig,
            f_outputs_sig,fuse_dir,
        ]
    )
    
    return model


def direction_layer(x, name):
    '''
    Direction net from connNet
    '''

    dir_layer = Dense(units=8)(x)

    squeeze = GlobalAveragePooling2D()(dir_layer)
    excitation = Dense(units=8)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=8)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 8))(excitation)

    scale = multiply([dir_layer, excitation])
    scale = Dense(units=8)(scale)
    # scale = Dense(units=8)(scale)
    scale = Activation('relu')(scale)
    # scale = Activation('sigmoid')(scale)
    sum_layer = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), name=name)(scale)
    return sum_layer

def squeeze_excitation_layer(x, out_dim):
        '''
        SE module performs inter-channel weighting.
        x: 21 channels
        '''
        # x1 = Conv2D(1, 2, padding='same', activation='relu', name='se_1')(x)
        # x2 = Conv2D(1, 2, padding='same', activation='relu', name='se_2')(x)
        # x3 = Conv2D(1, 2, padding='same', activation='relu', name='se_3')(x)
        # # x4 = Conv2D(1, 3, padding='same', activation='relu', name='se_4')(x)
        # # x5 = Conv2D(1, 3, padding='same', activation='relu', name='se_5')(x)
        # # x6 = Conv2D(1, 3, padding='same', activation='relu', name='se_6')(x)
        # # x7 = Conv2D(1, 3, padding='same', activation='relu', name='se_7')(x)
        # x = concatenate([x1,x2,x3])

        squeeze = GlobalAveragePooling2D()(x)
        
        excitation = Dense(units=out_dim)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1,1,out_dim))(excitation)
        
        scale = multiply([x,excitation])
        
        return scale