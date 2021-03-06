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
parser.add_argument("-lr", default = 0.0001, type = float)
parser.add_argument("-c1", default = -1, type = float)
parser.add_argument("-c2", default = -1, type = float)
parser.add_argument("-case", default = 3, type = int, choices = [1, 2, 3, 4])
parser.add_argument("-nepochs", default = 20, type = int)
parser.add_argument("-norm", default = 'l1', type = str)
args = parser.parse_args()

config = {
    'overwrite_name': 'dm',
}

config["overwrite_name"] += "-const%s-h%d-lr%g-dataset%d" % (
    utils.const_str(args.const), args.hidden, args.lr, args.dataset)
if args.c1 != -1:
    assert(args.c2 != -1)
    config["overwrite_name"] += "-c1=%g-c2=%g" % (args.c1, args.c2)
else:
    config["overwrite_name"] += "-unconstrained"

config['overwrite_name'] += '-case%d' % (args.case)
config['overwrite_name'] += '-norm%s' % (args.norm)

utils.setup_logging(args.seed, config['overwrite_name'])
print("Seed: %d" % args.seed)
session_config = utils.set_seed(args.seed, args.dataset)
n_permute_tasks, it, layer_sizes = utils.setup_dataset(args.dataset, args.inspect)

config = {
    **config,
    'use_latest_theta_star': False, # Use the saved wts instead of instantaneous wts
    'use_orig_loss': False, # Don't just use the initial CE loss, accumulate penalties
    'reset_opt': True, # Per task, reset optimizer
    'c1': None,
    'c2': None,
    'version': 'case%d' % (args.case),
    'norm': args.norm,
}

if args.hidden != None:
    layer_sizes = layer_sizes[:1] + [args.hidden for ln in range(len(layer_sizes)-2)] + layer_sizes[-1:]
else:
    print('hidden unset')
config['layer_sizes'] = layer_sizes

if args.c1 != -1: config['c1'] = args.c1
if args.c2 != -1: config['c2'] = args.c2

print(config)

net = utils.DMNetwork(
    layer_sizes = config['layer_sizes'],
    reshape_dims = it.reshape_dims,
    seed = args.seed,
    session_config = session_config,
    it = it,
    multiplier = args.const,
    use_latest_theta_star = config['use_latest_theta_star'],
    use_orig_loss = config['use_orig_loss'],
    reset_opt = config['reset_opt'],
    lr = args.lr,
    c1 = config['c1'],
    c2 = config['c2'],
    version = config['version'],
    norm = config['norm'],
    embedding = args.dataset in [3, 4], # cifar100
)

net.setup_phs() # Creates X, Y
net.forward() # Creates fX
net.backward() # Creates loss, train_op, predictions, acc
net.create_session()


for n_task in range(n_permute_tasks):

    print('Training for task %d' % n_task)
    it.switch_task(n_task)
    it.i = 0
    do_epoch = True
    epoch = 0

    while do_epoch:

        past_val_accs = net.train_epoch(n_task, epoch)
        saved_i = int(it.i)
        it.switch_task(n_task)
        it.i = saved_i
        epoch += 1
        if epoch >= args.nepochs: break

    print("Training finished for task %d" % n_task)

    net.final_stats(n_task, n_permute_tasks)

    it.switch_task(n_task) # fisher computation
    if n_task == n_permute_tasks-1:
        break
    else:
        net.save_task_weights(n_task)
        net.update_fisher_diag(n_task+1)
        net.update_loss(n_task+1)


net.objs['sess'].close()
