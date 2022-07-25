import pickle
import os
from turtle import colormode
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    save_img,
    img_to_array,
)

import pandas as pd
from PIL import Image
import numpy as np
from os import walk, getcwd
import h5py

import imageio
from glob import glob

from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

import pdb


class ImageLabelLoader:
    def __init__(self, image_folder, target_size):
        self.image_folder = image_folder
        self.target_size = target_size

    def build(self, att, batch_size, label=None):

        data_gen = ImageDataGenerator(rescale=1.0 / 255)
        if label:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col="image_id",
                y_col=label,
                target_size=self.target_size,
                class_mode="other",
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col="image_id",
                target_size=self.target_size,
                class_mode="input",
                batch_size=batch_size,
                shuffle=True,
            )

        return data_flow


class DataLoader:
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob("./data/%s/%s/*" % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = np.array(Image.fromarray(img).resize(self.img_res))

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = np.array(Image.fromarray(img).resize(self.img_res))
            imgs.append(img)

        imgs = np.array(imgs) / 127.5 - 1.0

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob("./data/%s/%sA/*" % (self.dataset_name, data_type))
        path_B = glob("./data/%s/%sB/*" % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_A = path_A[i * batch_size : (i + 1) * batch_size]
            batch_B = path_B[i * batch_size : (i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = np.array(Image.fromarray(img_A).resize(self.img_res))
                img_B = np.array(Image.fromarray(img_B).resize(self.img_res))

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.0
            imgs_B = np.array(imgs_B) / 127.5 - 1.0

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = np.array(Image.fromarray(img).resize(self.img_res))
        img = img / 127.5 - 1.0
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return imageio.imread(path, pilmode="RGB").astype(np.uint8)


def load_model(model_class, folder):

    with open(os.path.join(folder, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, "weights/weights.h5"))

    return model



def load_dataset(data_name, image_size, batch_size):
    data_folder = os.path.join("./data", data_name)

    data_gen = ImageDataGenerator(
        preprocessing_function=lambda x: (x.astype("float32") - 127.5) / 127.5
    )

    x_train = data_gen.flow_from_directory(
        data_folder,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode="input",
        subset="training",
        color_mode="rgb",
    )
    print(image_size)

    return x_train
