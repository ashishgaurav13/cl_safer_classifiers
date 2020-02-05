import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from keras.utils import to_categorical
import tensorflow.keras as keras
from .common import save_pickle, load_pickle
from tqdm import tqdm

# utils/data_iterators/cifar100_ResNet44v1_model.171.h5 => flatten
# utils/data_iterators/cifar100_ResNet44v1_model.171.h5 => activation_42

class CIFAR100_DataIterator:

    def __init__(self, train_data, test_data, batch_size = 32, 
        randomize = True, task_labels = None,
        embedding_save_file = 'utils/data_iterators/cifar100_embedding.pkl',
        embedding_model_file = 'utils/data_iterators/cifar100_ResNet44v1_model.171.h5',
        embedding_model_layer = 'activation_42'): # 'flatten'):

        assert(task_labels != None)
        self.train_x, self.train_y = train_data
        self.n = len(self.train_y)
        print('Training examples = %d' % self.n)
        self.test_x, self.test_y = test_data
        self.tn = len(self.test_y)
        print('Test examples = %d' % self.tn)
        self.i = 0
        self.batch_size = batch_size
        print('Batch size = %d' % self.batch_size)
        self.randomize = randomize
        if randomize:
            idx = np.random.permutation(self.n)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
            print('Shuffled training data')
        self.orig_data = (np.copy(self.train_x), np.copy(self.train_y),
            np.copy(self.test_x), np.copy(self.test_y))
        
        self.embedding_save_file = embedding_save_file
        self.embedding_model_file = embedding_model_file
        self.embedding_model_layer = embedding_model_layer
        self.reshape_dims = (64*8*8,) # (64,)
        self.convert_to_embeddings()

        self.n_tasks = len(task_labels)
        self.task_labels = task_labels
        self.n_labels_per_task = len(task_labels[0])
        for t in self.task_labels: assert(len(t) == self.n_labels_per_task)
        self.get_taskwise_data()
        self.switch_task(0)

        def img_fn_cifar100(img):
            image = np.zeros((32,32,3), dtype=np.uint8)
            image[...,0] = np.reshape(img[:1024], (32,32)) # Red channel
            image[...,1] = np.reshape(img[1024:2048], (32,32)) # Green channel
            image[...,2] = np.reshape(img[2048:], (32,32)) # Blue channel
            return image
        
        self.img_fn = img_fn_cifar100
    
    def iterative_fn(self, fn, dataset, batches = 100):
        ret = []
        n = dataset.shape[0]
        per_batch_size = n // batches
        for i in tqdm(range(batches)):
            if i+1 != batches:
                ret += [fn(dataset[i*per_batch_size:(i+1)*per_batch_size])]
            else:
                ret += [fn(dataset[i*per_batch_size:])]
        ret = np.vstack(ret)
        return ret

    def convert_to_embeddings(self):
        if os.path.isfile(self.embedding_save_file):
            print('Embedding file %s exists, skipping embedding generation.'
                % self.embedding_save_file)
            self.etrain_x, self.etest_x = load_pickle(self.embedding_save_file)
        else:
            assert(os.path.isfile(self.embedding_model_file))
            model = load_model(self.embedding_model_file)
            print("Loaded model: %s" % self.embedding_model_file)
            train_x = self.train_x.astype('float32') / 255
            train_x_mean = np.mean(train_x, axis = 0)
            train_x -= train_x_mean
            test_x = self.test_x.astype('float32') / 255
            test_x -= train_x_mean
            results = model.evaluate(test_x, to_categorical(self.test_y))
            print("Test acc: %s" % results)
            intermediate_layer = model.\
                get_layer(self.embedding_model_layer).output
            embedding_model = keras.Model(
                inputs = model.input, outputs = intermediate_layer)
            assert(len(self.reshape_dims) == 1)
            dim = self.reshape_dims[0]
            fn = lambda x: np.reshape(embedding_model.predict(x), [-1, dim])
            self.etrain_x = self.iterative_fn(fn, train_x)
            self.etest_x = self.iterative_fn(fn, test_x)
            save_pickle([self.etrain_x, self.etest_x],
                savefile = self.embedding_save_file)
            clear_session()
        print('Loaded embeddings.')
    
    # Remap class labels eg. 33,2,4 => 0, 1, 2
    def remap(self, x, classnums):
        # print(x)
        x = np.squeeze(x)
        # curr_labels = np.unique(x)
        # new_labels = {label: i for i, label in enumerate(curr_labels)}
        new_labels = {label: i for i, label in enumerate(classnums)}
        x_remapped = np.copy(x)
        for i in range(x.shape[0]):
            x_remapped[i] = new_labels[x[i]]
        # print(np.unique(x), np.unique(x_remapped))
        return x_remapped, new_labels

    def get_taskwise_data(self):
        self.tasks = {}
        for i in range(self.n_tasks):
            self.tasks[i] = {}
            class_nums = self.task_labels[i]
            tr_indices = np.array([np.where(self.train_y == class_num)[0] for \
                class_num in class_nums]).flatten()
            test_indices = np.array([np.where(self.test_y == class_num)[0] for \
                class_num in class_nums]).flatten()
            self.tasks[i]['train_x'] = self.etrain_x[tr_indices]
            self.tasks[i]['img_train_x'] = self.train_x[tr_indices]
            self.tasks[i]['train_y'], tr_labels = self.remap(self.train_y[tr_indices], class_nums)
            self.tasks[i]['n'] = len(tr_indices)
            if self.randomize:
                idx = np.random.permutation(self.tasks[i]['n'])
                self.tasks[i]['train_x'] = self.tasks[i]['train_x'][idx]
                self.tasks[i]['img_train_x'] = self.tasks[i]['img_train_x'][idx]
                self.tasks[i]['train_y'] = self.tasks[i]['train_y'][idx]
            self.tasks[i]['test_x'] = self.etest_x[test_indices]
            self.tasks[i]['img_test_x'] = self.test_x[test_indices]
            self.tasks[i]['test_y'], test_labels = self.remap(self.test_y[test_indices], class_nums)
            self.tasks[i]['tn'] = len(test_indices)
            if self.randomize:
                idx = np.random.permutation(self.tasks[i]['tn'])
                self.tasks[i]['test_x'] = self.tasks[i]['test_x'][idx]
                self.tasks[i]['img_test_x'] = self.tasks[i]['img_test_x'][idx]
                self.tasks[i]['test_y'] = self.tasks[i]['test_y'][idx]
            assert(tr_labels == test_labels)

    def switch_task(self, new_task_idx):
        assert(0 <= new_task_idx < self.n_tasks)
        self.curr_idx = new_task_idx
        self.n = self.tasks[self.curr_idx]['n']
        self.tn = self.tasks[self.curr_idx]['tn']
        self.train_x = self.tasks[self.curr_idx]['train_x']
        self.img_train_x = self.tasks[self.curr_idx]['img_train_x']
        self.train_y = np.squeeze(self.tasks[self.curr_idx]['train_y'])
        self.test_x = self.tasks[self.curr_idx]['test_x']
        self.img_test_x = self.tasks[self.curr_idx]['img_test_x']
        self.test_y = np.squeeze(self.tasks[self.curr_idx]['test_y'])
        # print('switch to %d: %s' % (new_task_idx, np.unique(self.test_y)))
    
    def inspect(self):

        print('inspect')

        r, c = self.n_tasks, self.n_labels_per_task
        xw = min(15, c)
        yw = max(1.5*r, 10)
        fig = plt.figure(figsize = (xw, yw))
        subplot_i = 0
        
        for task in range(self.n_tasks):
            self.switch_task(task)
            classes_to_show = np.unique(self.test_y)
            all_indices = [np.where(self.test_y == class_num)[0] for class_num in classes_to_show]
            n_ex = [len(item) for item in all_indices]
            example_indices = [np.random.choice(item) for item in all_indices]
            examples = self.img_test_x[example_indices]

            for i, img_idx in enumerate(classes_to_show):
                ax = fig.add_subplot(r, c, subplot_i+1)
                ax.set_xticks(())
                ax.set_yticks(())
                label_human_readable = str(img_idx) # TODO
                img = examples[img_idx]
                ax.set_xlabel(label_human_readable)
                plt.imshow(img, cmap='gray', interpolation='none')
                subplot_i += 1

        # plt.tight_layout(True)
        plt.savefig("inspect.png")
        plt.show()

    def __iter__(self):
        return self

    def __next__(self):
        if self.i+self.batch_size > self.n:
            self.i = 0
        ret_data = self.train_x[self.i:self.i+self.batch_size]
        ret_labels = self.train_y[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        return ret_data, ret_labels
    
    def test(self, samples = 32):
        idx = np.random.choice(self.tn, size = samples, replace = False)
        return self.test_x[idx], self.test_y[idx]