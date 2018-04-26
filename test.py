

# The followings are the DenseNets module, the training was actually taken place in the `run_dense_net.py` file.
# Sorry, I really like Pycharm (and to be fair, Pytorch is so much an easier language to debug)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from models import DenseNet
from data_providers.utils import get_data_provider_by_name

train_params_cifar = {
    'batch_size': 128,
    'n_epochs': 100,
    'initial_learning_rate': 0.01,
    'reduce_lr_epoch_1': 50,  # epochs * 0.5
    'reduce_lr_epoch_2': 75,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
    'use_Y': False,  # use only Y channel
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+', 'GTSR']:
        return train_params_cifar

import json
# We save this model params.json from the trained model
with open('model_params.json', 'r') as fp:
    model_params = json.load(fp)

# some default params dataset/architecture related
train_params = get_train_params_by_name(model_params['dataset'])
print("Params:")
for k, v in model_params.items():
    print("\t%s: %s" % (k, v))
print("Train params:")
for k, v in train_params.items():
    print("\t%s: %s" % (k, v))

print("Prepare training data...")
model_params['data_augmentation'] = False
data_provider = get_data_provider_by_name(model_params['dataset'], train_params)
print("Initialize the model..")
model = DenseNet(data_provider=data_provider, **model_params)
model.load_model()
print("Data provider test images: ", data_provider.test.num_examples)
print("Testing...")
loss, accuracy = model.test(data_provider.test, batch_size=200)
print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))