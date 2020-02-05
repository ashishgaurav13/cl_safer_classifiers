from scipy import io as spio
import numpy as np
import matplotlib.pyplot as plt
import pickle
import urllib
import zipfile, tarfile
import sys, os
from google_drive_downloader import GoogleDriveDownloader as gdd
import utils
import tensorflow.compat.v1 as tf
from scipy.misc import imresize
from scipy.io import loadmat
import collections

# Main function to setup datasets
# Supported Datasets:
# (0) P-MNIST
# (1) S-MNIST
# (2) Sim-EMNIST
def setup_dataset(idx, inspect, mixed = False):
    batch_size = 32
    if idx == 0: # p-mnist
        print("Dataset: P-MNIST")
        permutes = utils.PermutationGenerator().permutes
        n_permute_tasks = 5
        layer_sizes = [28*28, 128, 128, 10]
        train_data, test_data = tf.keras.datasets.mnist.load_data()
        it = utils.PMNIST_DataIterator(train_data, test_data, batch_size,
            n_tasks = n_permute_tasks)
        it.set_permutes(permutes)
        if inspect: it.inspect()

    elif idx == 1: # split mnist
        print("Dataset: S-MNIST")
        permutes = utils.PermutationGenerator().permutes
        n_permute_tasks = 5
        layer_sizes = [28*28, 128, 128, 2]
        train_data, test_data = tf.keras.datasets.mnist.load_data()
        it = utils.SMNIST_DataIterator(train_data, test_data, batch_size,
            n_tasks = n_permute_tasks)
        if inspect: it.inspect()

    elif idx == 2: # sim-emnist
        print("Dataset: Sim-EMNIST")
        # https://arxiv.org/pdf/1702.05373v1.pdf, Pg 8
        n_permute_tasks = 4
        layer_sizes = [28*28, 128, 128, 3]
        config = [["2", "O", "U"], ["Z", "8", "V"], ["7", "9", "W"], ["T", "Q", "Y"]]
        it = utils.SimEMNIST_Iterator(config, inspect = inspect)

    elif idx == 3: # cifar100
        print("Dataset: CIFAR100")
        n_permute_tasks = 5
        classes_per_task = 3
        layer_sizes = [64*8*8, 128, 128, classes_per_task]
        train_data, test_data = tf.keras.datasets.cifar100.load_data()
        lr_func = lambda a, b: list(range(a, b))
        create_config = lambda n, c: [lr_func(i*c, i*c+c) for i in range(n//c)]
        config = create_config(15, classes_per_task)
        it = utils.CIFAR100_DataIterator(train_data, test_data,
            task_labels = config)
        if inspect: it.inspect()

    elif idx == 4: # simcifar100
        print("Dataset: Sim-CIFAR100")
        coarse_to_fines = get_cifar100_coarse_to_fines()
        n_permute_tasks = 5
        classes_per_task = 3
        layer_sizes = [64*8*8, 128, 128, classes_per_task]
        train_data, test_data = tf.keras.datasets.cifar100.load_data()
        create_config = lambda d, s, cn: [[d[si][ci] for si in s] \
            for ci in range(cn)]
        s = [5, 6, 9] # see https://www.cs.toronto.edu/~kriz/cifar.html for coarse names
        config = create_config(coarse_to_fines, s, n_permute_tasks)
        it = utils.CIFAR100_DataIterator(train_data, test_data,
            task_labels = config)
        if inspect: it.inspect()

    else:
        print("Dataset not supported.")
        exit(0)

    return n_permute_tasks, it, layer_sizes

# For CIFAR100, find a mapping that produces all fine labels given a coarse label
# There are 20 coarse labels, 5 fine labels per coarse class
def get_cifar100_coarse_to_fines():
    download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz"
    save_file = "utils/data_iterators/cifar100.tar.gz"
    extract_dir = "utils/data_iterators/cifar100"
    if not os.path.exists(extract_dir):
        download_and_unzip(download_url, save_file, extract_dir)
    data_file = "utils/data_iterators/cifar100/cifar-100-matlab/test.mat"
    test_set = loadmat(data_file)
    y_labels = np.hstack((test_set['coarse_labels'], test_set['fine_labels']))
    unique_y_labels = np.unique(y_labels, axis = 0)
    names = loadmat("utils/data_iterators/cifar100/cifar-100-matlab/meta.mat")["fine_label_names"]
    ret = {}
    fine_names = {}
    for y_label in unique_y_labels:
        if y_label[0] not in ret:
            ret[y_label[0]] = []
        if y_label[1] not in fine_names:
            fine_names[y_label[1]] = None
        ret[y_label[0]] += [y_label[1]]
        fine_names[y_label[1]] = names[y_label[1]][0][0]

    fine_names = dict(collections.OrderedDict(sorted(fine_names.items())))
    # print(fine_names)
    return ret

def report(count, blockSize, totalSize):
  	percent = int(count*blockSize*100/totalSize)
  	sys.stdout.write("\r%d%%" % percent + ' complete')
  	sys.stdout.flush()

# Download a zip file from URL, save it as SAVE_FILE and
# unzip it to UNZIP_DIR
def download_and_unzip(url, save_file, unzip_dir):
    if not os.path.exists(save_file):
        print("Downloading %s ..." % url)
        if "drive.google.com" in url:
            file_id = url.split("id=")[-1]
            gdd.download_file_from_google_drive(file_id, "./"+save_file, unzip = False)
            filehandle = save_file
        else:
            filehandle, _ = urllib.request.urlretrieve(url, save_file, reporthook=report)
    else:
        print("%s exists" % save_file)
        filehandle = save_file
    if save_file.split(".")[-1] == "zip":
        zip_file_object = zipfile.ZipFile(filehandle, 'r')
        zip_file_object.extractall(unzip_dir)
        os.remove(save_file)
    elif len(save_file.split(".")) >= 2 and save_file.split(".")[-2:] == ["tar", "gz"]:
        targz_file = tarfile.open(save_file, "r:gz")
        targz_file.extractall(unzip_dir)
        os.remove(save_file)
    else:
        print("\nUnknown format")
        exit(0)

# Save a variable to __main__.__file__ pkl
def save_pickle(x, savefile = None):
    if save_pickle == None:
        import __main__
        logfile = __main__.__file__
        logfile = logfile.rstrip(".py") + ".pkl"
    else:
        logfile = savefile
    with open(logfile, "wb") as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle dumped to: %s" % logfile)

# Load a variable from a pkl file
def load_pickle(f):
    with open(f, "rb") as handle:
        return pickle.load(handle)

# Save tf variables
def save_variables_to_pkl(var_list, sess):
    var_dict = {}
    for var in var_list:
        var_dict[var.name] = sess.run(var)
    save_pickle(var_dict)

# Resize img to new shape
def resize_to(img, new_shape):
    n_channels = img.shape[-1]
    ret = np.zeros((*new_shape, n_channels), dtype = np.uint8)
    assert(n_channels <= 3)
    for i in range(n_channels):
        ret[...,i] = imresize(img[...,i], new_shape)
    return ret

# Generic iterator; Examples can be added through add(...), and once done
# you can call finish(...) to produce a set of train_{x/y}, test_{x/y}
class GenericIterator:

    def __init__(self, batch_size = 32, randomize = True, preprocess_fn = None):
        self.train_x, self.train_y = [], []
        self.test_x, self.test_y = [], []
        self.batch_size = batch_size
        self.randomize = randomize
        self.preprocess_fn = preprocess_fn

    def npadd(self, a, b):
        b = np.array(b)
        if getattr(self, a) == []:
            setattr(self, a, b)
        else:
            setattr(self, a, np.append(getattr(self, a), b, axis = 0))

    def add(self, x, y, tx, ty):
        self.npadd('train_x', x)
        self.npadd('train_y', y)
        self.npadd('test_x', tx)
        self.npadd('test_y', ty)
        # print("add: %s, %s, %s, %s" % (self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape))

    def finish(self):
        # self.train_x = np.reshape(self.train_x, [-1, *reshape_dims])
        # self.test_x = np.reshape(self.test_x, [-1, *reshape_dims])
        if self.preprocess_fn:
            self.test_x = np.array([self.preprocess_fn(item) for item in self.test_x])
        self.n = len(self.train_y)
        self.tn = len(self.test_y)
        print("n = %d, tn = %d" % (self.n, self.tn))
        self.i = 0
        if self.randomize:
            idx = np.random.permutation(self.n)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
            print("Shuffled")

    def __iter__(self):
        return self

    def __next__(self):
        if self.i+self.batch_size > self.n:
            self.i = 0
        ret_data = self.train_x[self.i:self.i+self.batch_size]
        if self.preprocess_fn:
            ret_data = np.array([self.preprocess_fn(item) for item in ret_data])
        ret_labels = self.train_y[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        return ret_data, ret_labels

    def test(self, samples = 32):
        idx = np.random.choice(self.tn, size = samples, replace = False)
        ret_data = self.test_x[idx]
        ret_labels = self.test_y[idx]
        return ret_data, ret_labels

# To generate permutations of pixels, for PMNIST
class PermutationGenerator:

    def load(self):
        with open(self.f, 'rb') as h:
            self.permutes = pickle.load(h)
            print("Loaded %s" % self.f)

    def save(self):
        assert(hasattr(self, 'permutes'))
        with open(self.f, 'wb') as h:
            pickle.dump(self.permutes, h, protocol = pickle.HIGHEST_PROTOCOL)
            print("Saved %s" % self.f)

    def generate(self, n, dim):
        self.permutes = [np.random.permutation(dim) for i in range(n)]
        print("Generated %d permutes of dim %d" % (n, dim))

    def __init__(self, f = 'utils/data_iterators/permute10.pickle', n = 10, dim = 784):
        self.f = f
        if os.path.exists(f):
            self.load()
            self.permutes[0] = np.array(range(dim))
            print("Reset permute 0 to default")
        else:
            print('ERROR: Place permute10.pickle in utils/data_iterators/')
            exit(0)
            self.generate(n, dim)
            self.save()
