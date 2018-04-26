import pickle
import numpy as np
import cv2
import random

from .base_provider import ImagesDataSet, DataProvider


def shiftxy(image, xoffset, yoffset):
    rows, cols, depth = image.shape
    M = np.float32([[1, 0, xoffset], [0, 1, yoffset]])
    res = cv2.warpAffine(np.copy(image), M, (cols, rows))
    assert (res.shape[0] == 32)
    assert (res.shape[1] == 32)
    return res


# function to rotate images by given degrees
def rotate(image, degree):
    rows, cols, depth = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    res = cv2.warpAffine(image, M, (cols, rows))
    assert (res.shape[0] == 32)
    assert (res.shape[1] == 32)
    return res


# function to resize the image
def scale(image, ratio):
    rows, cols, depth = image.shape
    newrows = int(ratio * rows)
    newcols = int(ratio * cols)
    res = cv2.resize(image, (newrows, newcols), interpolation=cv2.INTER_AREA)
    if newrows * newcols > 1024:
        # image is larger than 32x32, randomly crop the image back to 32x32
        xoffset = (newcols - 32) - int(random.random() * float(newcols - 32))
        yoffset = (newrows - 32) - int(random.random() * float(newrows - 32))
        cropped = res[xoffset:xoffset + 32, yoffset:yoffset + 32]
        res = cropped
    else:
        # image is smaller than before, randomly insert it into a 32x32 canvas
        if newrows * newcols < 1024:
            tmpimage = np.copy(image) * 0
            xoffset = (32 - newcols) - int(random.random() * float(32 - newcols))
            yoffset = (32 - newrows) - int(random.random() * float(32 - newrows))
            tmpimage[xoffset:newrows + xoffset, yoffset:newcols + yoffset] = res
            res = tmpimage
    assert (res.shape[0] == 32)
    assert (res.shape[1] == 32)
    return res


def gaussian_blur(img, kernel_size):
    # Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def augment_image(simage, jitter=3, rotate_degree=15, scaling=0.15):
    # set up the random jitter
    x = int(random.random() * jitter * 2) - jitter
    y = int(random.random() * jitter * 2) - jitter
    degree = int(random.random() * rotate_degree * 2) - rotate_degree
    ratio = random.random() * scaling * 2 + (1 - scaling)
    image = scale(rotate(shiftxy(simage, x, y), degree), ratio)
    return image


def augment_all_images(initial_images):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i])
    return new_images


class GTSRDataSet(ImagesDataSet):
    def __init__(self, images, labels, n_classes, shuffle, normalization,
                 augmentation, mean=None, std=None):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            n_classes: `int`, number of classes
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        self._means = mean
        self._stds = std
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        if self.augmentation:
            images = augment_all_images(images)
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class GTSRDataProvider(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, save_path=None, validation_combined=False,
                 shuffle=None, normalization=None,
                 one_hot=True, _n_classes=43, data_augmentation=False, use_Y=False, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return labels one hot encoded
        """
        self._n_classes = _n_classes
        self.data_augmentation = data_augmentation
        self._save_path = save_path
        self.one_hot = one_hot
        self.use_Y = use_Y

        # add train and validations datasets
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.read_data_and_label()
        if normalization == 'by_chanels':
            self._measure_mean_and_std(X_train)

        if not validation_combined:
            self.train = GTSRDataSet(
                images=X_train, labels=y_train,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=False,
                mean=self._means, std=self._stds)
            self.validation = GTSRDataSet(
                images=X_valid, labels=y_valid,
                n_classes=self.n_classes, shuffle=None,
                normalization=normalization,
                augmentation=self.data_augmentation,
                mean=self._means, std=self._stds)
        else:
            self.train = GTSRDataSet(
                images=np.concatenate((X_train, X_valid), axis=0),
                labels=np.concatenate((y_train, y_valid), axis=0),
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation,
                mean=self._means, std=self._stds)

        # add test set
        self.test = GTSRDataSet(
            images=X_test, labels=y_test,
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False,
            mean=self._means, std=self._stds)

    @property
    def data_shape(self):
        if self.use_Y:
            return (32, 32, 1)
        else:
            return (32, 32, 3)

    @property
    def n_classes(self):
        return self._n_classes

    def read_data_and_label(self):

        training_file = './data/train.p'
        validation_file = './data/valid.p'
        testing_file = './data/test.p'

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_test, y_test = test['features'], test['labels']
        if self.one_hot:
            y_train = self.labels_to_one_hot(y_train)
            y_valid = self.labels_to_one_hot(y_valid)
            y_test = self.labels_to_one_hot(y_test)
        if self.use_Y:
            X_train = np.expand_dims(X_train[:, :, :, 0], axis=3)
            X_valid = np.expand_dims(X_valid[:, :, :, 0], axis=3)
            X_test = np.expand_dims(X_test[:, :, :, 0], axis=3)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def _measure_mean_and_std(self, images):
        # for every channel in image
        means = []
        stds = []
        # for every channel in image(assume this is last dimension)
        for ch in range(images.shape[-1]):
            means.append(np.mean(images[:, :, :, ch]))
            stds.append(np.std(images[:, :, :, ch]))
        self._means = means
        self._stds = stds
