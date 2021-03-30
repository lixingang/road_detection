import numpy as np
import sklearn.metrics
import pandas as pd
import keras.backend as K
import cv2
import os
import tensorflow as tf
# from albumentations import (
#     HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
#     IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
# )


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def to_categorical_reverse(arr):
    return np.argmax(arr, axis=-1)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)*10


def _categorical_crossentropy(y_true, y_pred):
    pass


# def strong_aug(p=0.95, flag='train'):
#     if flag=='train':
#         return Compose([
#             OneOf([
#                 GaussNoise(),
#             ], p=0.2),
#             OneOf([
#                 MotionBlur(p=0.2),
#                 MedianBlur(blur_limit=3, p=0.1),
#             ], p=0.2),
#             OneOf([
#                 RandomContrast(),
#                 RandomBrightness(),
#             ], p=0.3),
#             HueSaturationValue(p=0.3),
#         ], p=p)
#     else:
#         return Compose([],p=p)


class DataLoader:
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def imread(self, im_path, islabel=False, RESIZE=True, THRESHOLD=True):
        if not islabel:
            img = cv2.imread(im_path)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if RESIZE:
                img = cv2.resize(img, (self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"]))
        elif islabel:
            img = cv2.imread(im_path)
            
            if RESIZE:
                img = cv2.resize(img, (self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"]))
            w = img.shape[0]
            img = img[:, :, 0].reshape([w, w, 1])
            
            if THRESHOLD:
                img = threshold(img)
        #             cv2.imwrite('img.png',img*200)
        #             print(img)
      
        return img

    def next_batch(self, type='train'):
        # perm = select_id(DATA, batch_size)
        if type == "train":
            filelist = parse_list(self.cfgs["TRAIN_LIST"])
        elif type == "valid":
            filelist = parse_list(self.cfgs["VALID_LIST"])
        else:
            print("[E] The Type must be *train or *test .")
            exit(-1)
        perm = np.arange(len(filelist))
        np.random.shuffle(perm)
        images = np.zeros([self.cfgs["BATCH_SIZE"], self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 3])
        ulabels = np.zeros([self.cfgs["BATCH_SIZE"], self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        elabels = np.zeros([self.cfgs["BATCH_SIZE"], self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        rlabels = np.zeros([self.cfgs["BATCH_SIZE"], self.cfgs["REGION_SIZE"], self.cfgs["REGION_SIZE"], 1])
        dlabels = np.zeros([self.cfgs["BATCH_SIZE"], self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        count = 0
        for i in perm[:self.cfgs["BATCH_SIZE"]]:
            fname = self.imread(os.path.join(self.cfgs["IMAGE_PATH"], filelist[i][0]), islabel=False) * 1.0 / 255
            ulabel = self.imread(os.path.join(self.cfgs["LABEL_PATH"], filelist[i][1]), islabel=True).reshape([512,512,1])
            elabel = self.imread(os.path.join(self.cfgs["EDGE_PATH"], filelist[i][1]), islabel=True).reshape([512,512,1])
            rlabel = self.imread(os.path.join(self.cfgs["REGION_PATH"], filelist[i][1]), islabel=True, RESIZE=False).reshape([32,32,1])
            dlabel = self.imread(os.path.join(self.cfgs["DIR_PATH"], filelist[i][1]), islabel=True, THRESHOLD=False)

            images[count, :, :, :] = fname
            ulabels[count, :, :, :] = ulabel
            elabels[count, :, :, :] = elabel
            rlabels[count, :, :, :] = rlabel
            dlabels[count, :, :, :] = dlabel
            count += 1
        return images, ulabels, elabels, rlabels, dlabels

    def get_test_data(self):
        filelist = parse_list(self.cfgs["TEST_LIST"])
        perm = np.arange(len(filelist))
        images = np.zeros([len(filelist), self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 3])
        ulabels = np.zeros([len(filelist), self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        elabels = np.zeros([len(filelist), self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        rlabels = np.zeros([len(filelist), self.cfgs["REGION_SIZE"], self.cfgs["REGION_SIZE"], 1])
        count = 0
        for i in perm:
#             print(os.path.join(self.cfgs["IMAGE_PATH"], filelist[i][0]))
            fname = self.imread(os.path.join(self.cfgs["IMAGE_PATH"], filelist[i][0]), islabel=False) * 1.0 / 255
            ulabel = self.imread(os.path.join(self.cfgs["LABEL_PATH"], filelist[i][1]), islabel=True).reshape([512,512,1])
            elabel = self.imread(os.path.join(self.cfgs["EDGE_PATH"], filelist[i][1]), islabel=True).reshape([512,512,1])
            rlabel = self.imread(os.path.join(self.cfgs["REGION_PATH"], filelist[i][1]), islabel=True, RESIZE=False).reshape([32,32,1])
#             print(os.path.join(self.cfgs["REGION_PATH"], filelist[i][1]), rlabel.shape, rlabels.shape)
            images[count, :, :, :] = fname
            ulabels[count, :, :, :] = ulabel
            elabels[count, :, :, :] = elabel
            rlabels[count, :, :, :] = rlabel
            count += 1
        return images, ulabels, elabels, rlabels, filelist
    def get_valid_data(self):
        filelist = parse_list(self.cfgs["VALID_LIST"])
        perm = np.arange(len(filelist))
        images = np.zeros([len(filelist), self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 3])
        ulabels = np.zeros([len(filelist), self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        elabels = np.zeros([len(filelist), self.cfgs["IMG_SIZE"], self.cfgs["IMG_SIZE"], 1])
        rlabels = np.zeros([len(filelist), self.cfgs["REGION_SIZE"], self.cfgs["REGION_SIZE"], 1])
        count = 0
        for i in perm:
            fname = self.imread(os.path.join(self.cfgs["IMAGE_PATH"], filelist[i][0]), islabel=False) * 1.0 / 255
            ulabel = self.imread(os.path.join(self.cfgs["LABEL_PATH"], filelist[i][1]), islabel=True)
            elabel = self.imread(os.path.join(self.cfgs["EDGE_PATH"], filelist[i][1]), islabel=True)
            rlabel = self.imread(os.path.join(self.cfgs["REGION_PATH"], filelist[i][1]), islabel=True, RESIZE=False)

            images[count, :, :, :] = fname
            ulabels[count, :, :, :] = ulabel
            elabels[count, :, :, :] = elabel
            rlabels[count, :, :, :] = rlabel
            count += 1
        return images, ulabels, elabels, rlabels, filelist

def threshold(im):
    max = np.max(im)
    min = np.min(im)
    if max == min:
        result = np.eye(im.shape[0]) * 255
        return result
    # average = (max+min)/2.0
    result = np.where(im > 0.01, 1, 0).astype(np.uint8)
    return result


def parse_list(txt):
    arr = None
    with open(txt, 'r') as f:
        arr = f.readlines()
    arr = [a.strip().split(' ') for a in arr]
    return arr


def randnorepeat(m, n):
    p = list(range(n))
    d = random.sample(p, m)
    return d


# def write_log(callback, names, logs, batch_no):
#     for name, value in zip(names, logs):
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         callback.writer.add_summary(summary, batch_no)
#         callback.writer.flush()

def mkdirs(dir, dir_list):
    for i in range(len(dir_list)):
        if os.path.exists(os.path.join(dir, dir_list[i])):
            continue
        else:
            os.mkdir(os.path.join(dir, dir_list[i]))


def sparse_Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):  # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:, :, 0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1) > 0
        ious = K.sum(inter, axis=1) / K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches))))  # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


class ResultManager:
    def __init__(self, epoch, logits, labels):
        self._epoch = epoch
        self._labels = threshold(labels)
        self._logits = logits
        self._pred = threshold(self._logits)

    def run(self, savename=None):
        self._auc = self.compute_auc()
        self._f1, self._P, self._R = self.compute_f1()
        self._iou = self.compute_iou()
        log_line = "epoch:{}, iou:{}, P:{},R:{},F1:{},AUC:{}\n".format(self._epoch, self._iou, self._P, self._R,
                                                                       self._f1, self._auc)
        print(log_line)
        # write logs
        if savename:
            with open(savename, 'a') as f:
                f.write(log_line)

    def compute_roc(self, savename):
        labels = np.reshape(self._labels, [-1, 1])
        logits = np.reshape(self._logits, [-1, 1])
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, logits)
        print(fpr.shape, tpr.shape, thresholds.shape)
        df = np.hstack([fpr, tpr, thresholds]).reshape(-1, 3)
        roc = pd.DataFrame(df, columns=['fpr', 'tpr', 'thresholds'])
        roc.to_csv(savename, sep=',', encoding='utf-8')

    def compute_auc(self):
        labels = np.reshape(self._labels, [-1, 1])
        logits = np.reshape(self._logits, [-1, 1])
        return sklearn.metrics.roc_auc_score(labels, logits)

    def compute_f1(self):
        labels = np.reshape(self._labels, [-1, 1])
        preds = np.reshape(self._logits, [-1, 1])
        labels = threshold(labels)
        preds = threshold(preds)
        Precision = sklearn.metrics.precision_score(preds, labels)
        Recall = sklearn.metrics.recall_score(preds, labels)
        F1 = sklearn.metrics.f1_score(preds, labels)
        return F1, Precision, Recall

    def compute_iou(self):
        gt = self._labels
        preds = self._pred
        gt = np.where(gt > 0.5, 1., 0.)
        preds = np.where(preds > 0.5, 1., 0.)

        intersection = gt * preds
        union = gt + preds

        union = np.where(union > 0, 1., 0.)
        intersection = np.sum(intersection)
        union = np.sum(union)
        if union == 0:
            union = 1e-09
        return intersection / union

    def adjust(self, num, mode='01'):
        if mode == "01":
            if num > 0 and num < 1:
                return num
            else:
                return 0


if __name__ == '__main__':
    dl = DataLoader(3)
    arr = dl.parse_list('1.txt')
    print(len(arr))