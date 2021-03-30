# project_ours_vegas
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from utils import *
from tqdm import tqdm
import yaml
import cv2

# kerasi
import keras
import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard
# modeli
from segmentation_models import Unet

class trainer:
    def __init__(self):
        pass

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

    def define_loss(self, *par):
        dict_train_y = {
            "u_outputs_sig": cross_entropy_balanced,
             "e_outputs_0_sig": cross_entropy_balanced,
             "e_outputs_1_sig": cross_entropy_balanced,
             "e_outputs_2_sig": cross_entropy_balanced,
             "e_outputs_3_sig": cross_entropy_balanced,
            "e_fuse_sig": cross_entropy_balanced,
             "r_outputs_0_sig": cross_entropy_balanced,
             "r_outputs_1_sig": cross_entropy_balanced,
             "r_outputs_2_sig": cross_entropy_balanced,
             "r_outputs_3_sig": cross_entropy_balanced,
            "r_fuse_sig": cross_entropy_balanced,
            'f_outputs_sig':cross_entropy_balanced,
            'fuse_dir': 'mse'

        }
        # dict_train_y = [par[0],par[1],par[1],par[1],par[1],par[2],par[2],par[2],par[2],par[0]]
        return dict_train_y

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
        adam = keras.optimizers.Adam(lr=self.cfgs["LEARNING_RATE"])
        model.summary()
        # model.compile('Adam', sigmoid_cross_entropy_balanced)
        model.compile('Adam',
                      # cross_entropy_balanced
                      loss=self.define_loss(),
                      # 'binary_crossentropy'
                      )

        test_images, test_ulabels, test_elabels, test_rlabels, filelist = self.dl.get_test_data()

        if self.cfgs["RESTORE"]:
            model.load_weights(os.path.join(self.cfgs["SAVE_DIR"], "weights", "epoch150.h5"))
            print("RETORE SUCCESSFULLY!")

        callback = TensorBoard('./graph')
        callback.set_model(model)
        train_names = ['loss', 'u_outputs_sig_loss', 'e_fuse_sig_loss', 'r_fuse_sig_loss', 'fuse_dir_loss']

        current_learning_rate = self.cfgs["LEARNING_RATE"]
        K.set_value(model.optimizer.lr, current_learning_rate)
        for i in range(self.cfgs["EPOCH"]):
            print("[I] EPOCH {}".format(i))
            # TRAIN
            for j in tqdm(range(self.cfgs["STEP"])):
                images_batch, ulabels_batch, elabels_batch, rlabels_batch, d_labels_batch = self.dl.next_batch("train")
                Logs = model.train_on_batch(images_batch,
                                            self.define_train_y(ulabels_batch, elabels_batch, rlabels_batch,
                                                                d_labels_batch),
                                            )

            write_log(callback, train_names, Logs, i)
            if i % self.cfgs["INTERVAL"] == 0 and i >= 0:

                # TEST:
                results = model.predict(test_images, batch_size=10, verbose=0)
                logits = results[-1]
                r_logits = results[-2]

                # result analyse and show
                rlt_worker = ResultManager(i, logits, test_ulabels)
                # r_analyst.compute_roc(savename='roc_vegas_{}.csv'.format(i))
                # rlt_worker_r = ResultManager(i, r_logits, test_rlabels)

                rlt_worker.run()
                # rlt_worker_r.run()

                for ii in range(results[0].shape[0]):
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/images/{}'.format(filelist[ii][0])),
#                                 test_images[ii, :] * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/labels/{}'.format(filelist[ii][1])),
#                                 test_ulabels[ii, :] * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/labels_e/{}'.format(filelist[ii][1])),
#                                 test_elabels[ii, :] * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/labels_r/{}'.format(filelist[ii][1])),
#                                 test_rlabels[ii, :] * 255)

                    #cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/preds/{}'.format(filelist[ii][1])),
                     #           results[-1][ii, :])
                    pred_threshold = threshold(results[-1][ii, :])
                    cv2.imwrite(
                        os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/preds_threshold/{}'.format(filelist[ii][1])),
                        pred_threshold * 255)

#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e1/{}'.format(filelist[ii][1])),
#                                 threshold(results[1][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e2/{}'.format(filelist[ii][1])),
#                                 threshold(results[2][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e3/{}'.format(filelist[ii][1])),
#                                 threshold(results[3][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e4/{}'.format(filelist[ii][1])),
#                                 threshold(results[4][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_e5/{}'.format(filelist[ii][1])),
#                                 threshold(results[5][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r1/{}'.format(filelist[ii][1])),
#                                 threshold(results[6][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r2/{}'.format(filelist[ii][1])),
#                                 threshold(results[7][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r3/{}'.format(filelist[ii][1])),
#                                 threshold(results[8][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r4/{}'.format(filelist[ii][1])),
#                                 threshold(results[9][ii, :]) * 255)
#                     cv2.imwrite(os.path.join(self.cfgs["SAVE_DIR"], 'main_outputs/out_r5/{}'.format(filelist[ii][1])),
#                                 threshold(results[10][ii, :]) * 255)


                # SAVE WEIGHTS
                current_learning_rate = current_learning_rate * self.cfgs["LEARNING_RATE_DECAY"]
                K.set_value(model.optimizer.lr, current_learning_rate)
                print('[I] Current Learning Rate: ', current_learning_rate)
                model_json = model.to_json()
                with open("model.json", "w") as json_file:
                    json_file.write(model_json)
                model.save_weights(os.path.join(self.cfgs["SAVE_DIR"], "epoch{}.h5".format(i)))


if __name__ == '__main__':
    worker = trainer()
    worker.setup()
    worker.workflow()


