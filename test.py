# The followings are the DenseNets module, the training was actually taken place in the `run_dense_net.py` file.
# Sorry, I really like Pycharm (and to be fair, Pytorch is so much an easier language to debug)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from models import DenseNet
from data_providers.utils import get_data_provider_by_name
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import random
import time
from matplotlib import pyplot as plt
# Visualizations will be shown in the notebook.
# % matplotlib inline
from matplotlib import gridspec

# Load pickled data
import pickle
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
X_test, y_test_origin = test['features'], test['labels']


train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 500,
    'initial_learning_rate': 0.05,
    'reduce_lr_epoch_1': 50,  # epochs * 0.5
    'reduce_lr_epoch_2': 75,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
    'use_YUV': True,
    'use_Y': True,  # use only Y channel
    'data_augmentation': 0,  # [0, 1]
}


# We save this model params.json from the trained model
with open('model_params.json', 'r') as fp:
    model_params = json.load(fp)

# some default params dataset/architecture related
train_params = train_params_cifar
print("Params:")
for k, v in model_params.items():
    print("\t%s: %s" % (k, v))
print("Train params:")
for k, v in train_params.items():
    print("\t%s: %s" % (k, v))


model_params['use_Y'] = True
print("Prepare training data...")
data_provider = get_data_provider_by_name(model_params['dataset'], train_params)
print("Initialize the model..")
tf.reset_default_graph()
model = DenseNet(data_provider=data_provider, **model_params)
print("Loading trained model")
model.load_model()

print("Data provider test images: ", data_provider.test.num_examples)
print("Testing...")
loss, accuracy = model.test(data_provider.test, batch_size=30)
print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
total_prediction, y_test = model.predictions_test(data_provider.test, batch_size=100)


# Plotting incorrect examples
incorrectlist = []
for i in range(len(total_prediction)):
    #if not correctness(y_test[i],total_prediction[i]):
    for j in range(len(y_test[i])):
        if not np.argmax(y_test[i][j]) == np.argmax(total_prediction[i][j]):
            correct_classId = np.argmax(y_test[i][j])
            predict_classId = np.argmax(total_prediction[i][j])
            incorrectlist.append({'index':i*100+j, 'correct':correct_classId, 'predicted':predict_classId})


incorrectmatrix = {}
modeCount = 0
# get the label description from the CSV file.
classLabelList = pd.read_csv('signnames.csv')
for i in range(len(incorrectlist)):
    predicted = incorrectlist[i]['predicted']
    correct = incorrectlist[i]['correct']
    index = incorrectlist[i]['index']
    bucket = str(correct) + "+" + str(predicted)
    incorrectinstance = incorrectmatrix.get(bucket, {'count': 0, 'samples': []})

    # add to the count
    count = incorrectinstance['count'] + 1

    # add to samples of this correct to predicted condition
    samples = incorrectinstance['samples']
    samples.append(index)

    # put back in the list
    incorrectmatrix[bucket] = {'count': count, 'correct': correct, 'predicted': predicted, 'samples': samples}

    # update most common error
    if count > modeCount:
        modeCount = count
        modeBucket = bucket


# get the list of buckets and sort them
def compare_bucket_count(bucket):
    return modeCount - incorrectmatrix[bucket]['count']


sortedBuckets = list(incorrectmatrix.keys())
sortedBuckets.sort(key=compare_bucket_count)

# get the unique number of original picture sizes and the min and max last instance
n_buckets = len(sortedBuckets)

# print the stats
print("\nNumber of unique buckets in incorrect set: ", n_buckets, "\n")
print("Mode Bucket: ", modeBucket, "with count: ", modeCount)

print("\nTop Twenty Distribution of buckets with incorrect predicted test dataset labels:")
for n in range(20):
    bucket = sortedBuckets[n]
    cclassId = incorrectmatrix[bucket]['correct']
    pclassId = incorrectmatrix[bucket]['predicted']
    count = incorrectmatrix[bucket]['count']
    cdescription = classLabelList[classLabelList.ClassId == cclassId].SignName.to_string(header=False, index=False)
    pdescription = classLabelList[classLabelList.ClassId == pclassId].SignName.to_string(header=False, index=False)
    print(
        "incorrect set count: {0:4d}  CClassId: {1:02d} Description: {2}\n                           PClassId: {3:02d} Description: {4}".format(
            count, cclassId, cdescription, pclassId, pdescription))


def draw_sample_incorrectmatrix(datasettxt, sortedBuckets, incorrectmatix, dataset, cmap=None):
    n_samples = 11
    n_labels = 10

    # size of each sample
    fig = plt.figure(figsize=(n_samples * 1.8, n_labels))
    w_ratios = [1 for n in range(n_samples)]
    w_ratios[:0] = [int(n_samples * 0.8)]
    h_ratios = [1 for n in range(n_labels)]

    # gridspec
    time.sleep(1)  # wait for 1 second for the previous print to appear!
    grid = gridspec.GridSpec(n_labels, n_samples + 1, wspace=0.0, hspace=0.0, width_ratios=w_ratios,
                             height_ratios=h_ratios)
    labelset_pbar = tqdm(range(n_labels), desc=datasettxt, unit='labels')
    for a in labelset_pbar:
        cclassId = incorrectmatrix[sortedBuckets[n_labels - a - 1]]['correct']
        pclassId = incorrectmatrix[sortedBuckets[n_labels - a - 1]]['predicted']
        cdescription = classLabelList[classLabelList.ClassId == cclassId].SignName.to_string(header=False, index=False)
        pdescription = classLabelList[classLabelList.ClassId == pclassId].SignName.to_string(header=False, index=False)
        count = incorrectmatrix[sortedBuckets[n_labels - a - 1]]['count']
        for b in range(n_samples):
            i = a * (n_samples + 1) + b
            ax = plt.Subplot(fig, grid[i])
            if b == 0:
                ax.annotate(
                    'CClassId %d (%d): %s\nPClassId %d: %s' % (cclassId, count, cdescription, pclassId, pdescription),
                    xy=(0, 0), xytext=(0.0, 0.3))
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
            else:
                random_i = random.choice(incorrectmatrix[sortedBuckets[n_labels - a - 1]]['samples'])
                image = dataset[random_i]
                if cmap == None:
                    ax.imshow(image)
                else:
                    # yuv = cv2.split(image)
                    # ax.imshow(yuv[0], cmap=cmap)
                    ax.imshow(image, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

        # We also plot the GT image on the right
        i = a * (n_samples + 1) + n_samples
        ax = plt.Subplot(fig, grid[i])

        img_idx = np.where(y_train == pclassId)
        random_i = random.choice(img_idx[0])
        image = X_train[random_i]
        if cmap == None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

        # hide the borders\
        if a == (n_labels - 1):
            all_axes = fig.get_axes()
            for ax in all_axes:
                for sp in ax.spines.values():
                    sp.set_visible(False)

    plt.show()

draw_sample_incorrectmatrix('Test set 10 ten incorrect sample images  using RGB as input, right most is the predicted image in the training set', sortedBuckets, incorrectmatrix, test['features'])
