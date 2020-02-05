import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger('tensorflow')
logger.disabled = True

import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils
from tqdm import tqdm
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-const", type = float, default = 0.5)
parser.add_argument("-seed", type = int, default = 0)
parser.add_argument("-dataset", type = int, required = True, choices = [0, 1, 2, 3, 4])
parser.add_argument("-inspect", default = False, action = "store_true")
parser.add_argument("-hidden", type = int, default = 128)
parser.add_argument("-c", default = 0.1, type = float)
parser.add_argument("-xi", default = 0.1, type = float)
parser.add_argument("-lr", default = 0.0001, type = float)
parser.add_argument("-nepochs", default = 20, type = int)
args = parser.parse_args()

config = {
    'overwrite_name': 'si-h%d-lr%g-c%g-xi%g-dataset%d' % (args.hidden, args.lr, args.c, args.xi, args.dataset),
}

utils.setup_logging(args.seed, config['overwrite_name'])
print("Seed: %d" % args.seed)
session_config = utils.set_seed(args.seed, args.dataset)
n_permute_tasks, it, layer_sizes = utils.setup_dataset(args.dataset, args.inspect)

config = {
    **config,
    'c': args.c,
    'xi': args.xi,
    'lr': args.lr,
}

if args.hidden != None:
    layer_sizes = layer_sizes[:1] + [args.hidden for ln in range(len(layer_sizes)-2)] + layer_sizes[-1:]
else:
    print('hidden unset')
config['layer_sizes'] = layer_sizes
print(config)

net = utils.SINetwork(
    layer_sizes = config['layer_sizes'],
    reshape_dims = it.reshape_dims,
    seed = args.seed,
    session_config = session_config,
    it = it,
    c = config['c'],
    xi = config['xi'],
    lr = config['lr'],
)


net.setup()

for n_task in range(n_permute_tasks):

    it.switch_task(n_task)
    it.i = 0
    n_labels = len(np.unique(it.test_y))
    division = 255.0 if args.dataset in [0, 1, 2] else 1.0
    net.preprocessed_(n_task, {
        'train_x': it.train_x.astype('float32') / division,
        'test_x': it.test_x.astype('float32') / division,
        'train_y': np.eye(n_labels)[it.train_y],
        'test_y': np.eye(n_labels)[it.test_y],
    })


for n_task in range(n_permute_tasks):

    print('Training for task %d' % n_task)
    it.switch_task(n_task)
    it.i = 0
    net.oopt.set_nb_data(it.n)

    for epoch in range(args.nepochs):
        net.train_epoch(n_task, epoch)

    net.oopt.update_task_metrics(
        net.all_data[n_task]['train_x'],
        net.all_data[n_task]['train_y'],
        it.batch_size
    )
    net.oopt.update_task_vars()

    print("Training finished for task %d" % n_task)

    net.final_stats(n_task, n_permute_tasks)