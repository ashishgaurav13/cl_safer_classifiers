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
parser.add_argument("-dataset", type = int, default = 3)
parser.add_argument("-inspect", default = False, action = "store_true")
parser.add_argument("-hidden", type = int, default = 128)
parser.add_argument("-fix", type = float, default = 0.0)
parser.add_argument("-avg", default = False, action = "store_true")
parser.add_argument("-nv", default = False, action = "store_true")
parser.add_argument("-nvfast", default = False, action = "store_true")
parser.add_argument("-correctmask", default = True, action = "store_true") # default True
parser.add_argument("-evaldist", default = False, action = "store_true")
parser.add_argument("-lr", default = 0.0001, type = float)
args = parser.parse_args()

config = {
    'overwrite_name': 'ewcgrad-%.2f' % args.fix,
}

if args.nvfast or args.nv:
    if args.nv:
        config['overwrite_name'] += '-nv'
    else:
        config['overwrite_name'] += '-nvfast'

config['overwrite_name'] += '-h%d-lr%g-const%s-dataset%d' % (args.hidden, args.lr, utils.const_str(args.const), args.dataset)

utils.setup_logging(args.seed, config['overwrite_name'])
print("Seed: %d" % args.seed)
session_config = utils.set_seed(args.seed, args.dataset)
n_permute_tasks, it, layer_sizes = utils.setup_dataset(args.dataset, args.inspect)
# n_permute_tasks = 3

config = {
    **config,
    'reset_fisher': False,
    'use_latest_theta_star': False,
    'use_orig_loss': False,
    'reset_opt': True, # False should work but True if to be compared against growing ewc
    'fix': args.fix,
    'fisher_avg': False,
    'newest_variance': False,
    'less_compute': False,
    'correctmask': False,
    'lr': args.lr
}

if args.avg:
    config['fisher_avg'] = True
elif args.nv or args.nvfast:
    config['newest_variance'] = True
    if args.nvfast:
        config['less_compute'] = True
        config['correctmask'] = args.correctmask


if args.hidden != None:
    layer_sizes = layer_sizes[:1] + [args.hidden for ln in range(len(layer_sizes)-2)] + layer_sizes[-1:]
else:
    print('hidden unset')
config['layer_sizes'] = layer_sizes
print(config)

net = utils.EWCGradientsNetwork(
    layer_sizes = config['layer_sizes'],
    reshape_dims = it.reshape_dims,
    seed = args.seed,
    session_config = session_config,
    it = it,
    ewc_const = args.const,
    reset_fisher = config['reset_fisher'],
    use_latest_theta_star = config['use_latest_theta_star'],
    use_orig_loss = config['use_orig_loss'],
    reset_opt = config['reset_opt'],
    fix = config['fix'],
    fisher_avg = config['fisher_avg'],
    newest_variance = config['newest_variance'],
    less_compute = config['less_compute'],
    correctmask = config['correctmask'],
    lr = config['lr'],
    embedding = args.dataset in [3, 4], # cifar
)

net.setup_phs() # Creates X, Y
net.forward() # Creates fX
net.backward() # Creates loss, train_op, predictions, acc
net.create_session()

for n_task in range(n_permute_tasks):

    if args.evaldist:
        net.eval_distances(n_task, n_permute_tasks, begin = True)

    # net.objs['es'].reset()
    print('Training for task %d' % n_task)
    it.switch_task(n_task)
    it.i = 0
    do_epoch = True
    epoch = 0

    while do_epoch:

        past_val_accs = net.train_epoch(n_task, epoch)
        saved_i = int(it.i)
        if args.evaldist:
            net.eval_distances(n_task, n_permute_tasks, begin = False)
        it.switch_task(n_task)
        it.i = saved_i
        epoch += 1
        if epoch >= 20: break
        # es_ret = net.objs['es'].add_acc(past_val_accs)
        # do_epoch = es_ret == -1

    print("Training finished for task %d" % n_task)

    net.final_stats(n_task, n_permute_tasks)

    it.switch_task(n_task)
    if n_task == n_permute_tasks-1:
        break
    else:
        net.save_task_weights(n_task)
        net.update_fisher_diag(n_task+1)
        net.update_loss(n_task+1)


net.objs['sess'].close()
