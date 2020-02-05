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
parser.add_argument("-seed", type = int, default = 0)
parser.add_argument("-dataset", type = int, required = True, choices = [0, 1, 2, 3, 4])
parser.add_argument("-inspect", default = False, action = "store_true")
parser.add_argument("-hidden", type = int, default = 128)
parser.add_argument("-lr", type = float, default = 0.0001)
parser.add_argument("-nepochs", type = int, default = 20)
args = parser.parse_args()

overwrite_name = "baseline-h%d-lr%g-dataset%d" % (args.hidden, args.lr, args.dataset)
utils.setup_logging(args.seed, overwrite_name)
print("Seed: %d" % args.seed)
session_config = utils.set_seed(args.seed, args.dataset)
n_permute_tasks, it, layer_sizes = utils.setup_dataset(args.dataset, args.inspect)

if args.hidden != None:
    layer_sizes = layer_sizes[:1] + [args.hidden for ln in range(len(layer_sizes)-2)] + layer_sizes[-1:]
else:
    print('hidden unset')

net = utils.BaselineNetwork(
    layer_sizes = layer_sizes,
    reshape_dims = it.reshape_dims,
    seed = args.seed,
    session_config = session_config,
    it = it,
    lr = args.lr,
    embedding = args.dataset in [3, 4], # cifar100
)

net.setup_phs() # Creates X, Y
net.forward() # Creates fX
net.backward() # Creates loss, train_op, predictions, acc
net.create_session()

for n_task in range(n_permute_tasks):

    print('Training for task %d' % n_task)
    it.switch_task(n_task)
    do_epoch = True
    epoch = 0

    while do_epoch:

        past_val_accs = net.train_epoch(n_task, epoch)
        epoch += 1
        if epoch >= args.nepochs: break

    print("Training finished for task %d" % n_task)

    net.final_stats(n_task, n_permute_tasks)

net.objs['sess'].close()
