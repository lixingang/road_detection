import os,sys
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from utils import *
from tqdm import tqdm
import yaml
import cv2
import numpy as np
# keras
from keras.callbacks import TensorBoard
import keras.backend as K
from keras import optimizers
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.tensorflow_backend.set_session(session)

# model
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing


class trainer:
    def __init__(self,epoch):
        LOGS = []
        self.epoch = epoch
    def setup(self, cfg_path="config.yaml"):
        with open(cfg_path) as pfile:
            self.cfgs = yaml.load(pfile)
        self.dl = DataLoader(self.cfgs)

    def data_generator(self, type="train"):
        # type = "train" or "test"
        if type == 'train':
            images_batch, labels_batch = self.dl.next_batch(type)
            return images_batch, labels_batch
        if type == 'test':
            images_all, labels_all = self.dl.get_test_data()
            return images_all, labels_all

    def define_train_y(self, *par):
        dict_train_y = {
            "u_outputs_sig": par[0],
            "e_outputs_0_sig": par[1],
            "e_outputs_1_sig": par[1],
            "e_outputs_2_sig": par[1],
            "e_outputs_3_sig": par[1],
            "e_fuse_sig": par[1],
            "r_outputs_0_sig": par[2],
            "r_outputs_1_sig": par[2],
            "r_outputs_2_sig": par[2],
            "r_outputs_3_sig": par[2],
            "r_fuse_sig": par[2],
            'f_outputs_sig': par[0],
            'fuse_dir': par[3]
        }
        # dict_train_y = [par[0],par[1],par[1],par[1],par[1],par[2],par[2],par[2],par[2],par[0]]
        return dict_train_y

    def workflow(self):
        # define model
        model = Unet(backbone_name='resnet50', encoder_weights='imagenet')
        #model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
        model.load_weights(os.path.join(self.cfgs["SAVE_DIR"], "epoch"+str(self.epoch)+".h5"))
        print("RETORE SUCCESSFULLY!")
        test_images, test_ulabels, test_elabels, test_rlabels, filelist = self.dl.get_test_data()
        # TEST:
        print('start')
        start = time.clock()
        results = model.predict(test_images, batch_size=5, verbose=1)
        stop = time.clock()
        print('程序运行时间：',str(stop-start),' 秒')
        pmlabels = results[0]

        print(len(results))

        mkdirs(self.cfgs["SAVE_DIR"],['images', 'labels_e', 'labels_r', 'preds','preds_threshold'])
        for ii in range(results[0].shape[0]):
            cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'images/{}'.format(filelist[ii][0])),
                        test_images[ii, :] * 255)

            cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'labels_e/{}'.format(filelist[ii][1])),
                        test_elabels[ii, :] * 255)
            cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'labels_r/{}'.format(filelist[ii][1])),
                        test_rlabels[ii, :] * 255)

            cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'preds/{}'.format(filelist[ii][1])),
                        results[-1][ii, :])
            pred_threshold = threshold(results[-1][ii, :])
            cv2.imwrite(
                os.path.join(self.cfgs["SAVE_DIR"], 'preds_threshold/{}'.format(filelist[ii][1])),
                pred_threshold * 255)

#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e1/{}'.format(filelist[ii][1])),
#                         threshold(results[1][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e2/{}'.format(filelist[ii][1])),
#                         threshold(results[2][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e3/{}'.format(filelist[ii][1])),
#                         threshold(results[3][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e4/{}'.format(filelist[ii][1])),
#                         threshold(results[4][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e5/{}'.format(filelist[ii][1])),
#                         threshold(results[5][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r1/{}'.format(filelist[ii][1])),
#                         threshold(results[6][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r2/{}'.format(filelist[ii][1])),
#                         threshold(results[7][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r3/{}'.format(filelist[ii][1])),
#                         threshold(results[8][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r4/{}'.format(filelist[ii][1])),
#                         threshold(results[9][ii, :]) * 255)
#             cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r5/{}'.format(filelist[ii][1])),
#                         threshold(results[10][ii, :]) * 255)


if __name__ == "__main__":
    epoch = sys.argv[1]
    worker = trainer(epoch)
    worker.setup()
    worker.workflow()
