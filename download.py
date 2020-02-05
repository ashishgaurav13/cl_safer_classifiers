import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger('tensorflow')
logger.disabled = True

import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils

for dataset in [0, 1, 2, 3, 4]:
    _ = utils.setup_dataset(dataset, inspect = False)
    print("Downloaded dataset %d." % dataset)

print("Downloaded all datasets.")