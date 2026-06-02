import re
import os
import numpy as np
import  cv2
from config import *
from random import shuffle
import  tensorflow as tf

class DataManager(object):
    def __init__(self, dataList,param,shuffle=True):
        """
        """
        self.shuffle=shuffle
        self.data_list=dataList
        self.data_size=len(dataList)
        self.data_dir=param["data_dir"]
        self.epochs_num=param["epochs_num"]
        self.batch_size = param["batch_size"]
        self.number_batch = int(np.floor(len(self.data_list) /self.batch_size))
        self.next_batch=self.get_next()

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3+200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        for index in range(len(self.data_list)):
            file_basename_image,file_basename_label = self.data_list[index]
            image_path = os.path.join(self.data_dir, file_basename_image)
            label_path= os.path.join(self.data_dir, file_basename_label)
            image= self.read_data(image_path)
            label = self.read_data(label_path, is_label=True)
            label_pixel,label=self.label_preprocess(label)
            image = (np.array(image[:, :, np.newaxis]))
            label_pixel = (np.array(label_pixel[:, :, np.newaxis]))
            yield image, label_pixel,label, file_basename_image

    def read_data(self, data_name, is_label=False):
        """
        Read and preprocess an image.

        When is_label=True, reads raw without enhancement (label pixels
        should stay 0/255).  When is_label=False, applies bilateral filter
        + CLAHE for defect-preserving contrast enhancement.
        """
        img = cv2.imread(data_name, 0)  # read as grayscale

        if not is_label:
            # ---- Pain Point 1: eliminate lighting artifacts ----
            # Bilateral filter: smooths background noise + highlight speckles
            # while keeping defect boundary edges sharp.
            img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

            # CLAHE: Contrast Limited Adaptive Histogram Equalization
            # Suppresses large-area highlights, boosts weak scratch/pit contrast.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        return img


    def label_preprocess(self, label):
        """
        Downsample label from 1280x512 to 160x64 using max-pool logic.

        Pain Point 2 fix: cv2.resize with interpolation DILUTES single-pixel
        defects (burrs, hairline scratches < 3px wide) into oblivion,
        producing all-zero labels.  Max-pool guarantees: "if any pixel in the
        8x8 block is a defect, the downsampled pixel stays 1."

        Formula: reshape (M,K,N,K) -> max over axes 1,3 -> (M//K, N//K)
        """
        label = np.array(label)
        M, N = label.shape
        K = 8  # downsample factor (3x max_pool_2x2 = 8x)

        # Trim to multiple of K for clean reshape
        label_pixel = label[:M - M % K, :N - N % K]
        label_pixel = label_pixel.reshape(M // K, K, N // K, K).max(axis=(1, 3))

        # Binarize: any defect pixel -> 1
        label_pixel = np.where(label_pixel > 0, 1, 0)

        # Global label: at least one defect pixel -> 1 (NG)
        label_global = 1 if label_pixel.sum() > 0 else 0

        return label_pixel, label_global

    def ImageBinarization(self,img, threshold=1):
        img = np.array(img)
        image = np.where(img > threshold, 1, 0)
        return image

    def label2int(self,label):  # label shape (num,len)
        # seq_len=[]
        target_input = np.ones((MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
        target_out = np.ones(( MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
        target_input[0] = 0  # 第一个为GO
        for j in range(len(label)):
            target_input[j + 1] = VOCAB[label[j]]
            target_out[j] = VOCAB[label[j]]
            target_out[len(label)] = 1
        return target_input, target_out

    def int2label(self,decode_label):
        label = []
        for i in range(decode_label.shape[0]):
            temp = ''
            for j in range(decode_label.shape[1]):
                if VOC_IND[decode_label[i][j]] == '<EOS>':
                    break
                elif decode_label[i][j] == 3:
                    continue
                else:
                    temp += VOC_IND[decode_label[i][j]]
            label.append(temp)
        return label


    def get_label(self,f):
        return  f.split('.')[-2].split('_')[1]



