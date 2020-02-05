# Datasets, Image Manipulation, Data Handling
from .data_iterators.common import setup_dataset, download_and_unzip, resize_to
from .data_iterators.common import load_pickle, save_pickle, save_variables_to_pkl
from .data_iterators.common import GenericIterator, PermutationGenerator
from .data_iterators.pmnist import PMNIST_DataIterator
from .data_iterators.smnist import SMNIST_DataIterator
from .data_iterators.sim_emnist import SimEMNIST_Iterator
from .data_iterators.cifar100 import CIFAR100_DataIterator

# Logging
from .logging import Tee, setup_logging, timer, timer_done, const_str

# Networks
from .networks.network import Network
from .networks.network_baseline import BaselineNetwork
from .networks.network_ewc import EWCNetwork
from .networks.network_dm import DMNetwork
from .networks.network_si_pathint import SINetwork
from .networks.network_ewc_gradients import EWCGradientsNetwork

# Tensorflow
from .tf.common import set_seed, dense, dense2, average, get_var, say, \
    waverage, softplus_inv, get_activation, conv2d, gradient_clip_minimize_op, sum_up
from .tf.early_stopping import EarlyStopping, get_session
from .tf.vgg16 import vgg16, vgg16_reuse