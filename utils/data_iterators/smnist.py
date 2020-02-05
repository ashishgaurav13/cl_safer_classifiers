import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import PMNIST_DataIterator as DataIterator

class SMNIST_DataIterator:

    def __init__(self, train_data, test_data, batch_size = 32,
        randomize = True, n_tasks = 5):

        self.it = DataIterator(train_data, test_data, batch_size,
            randomize, n_tasks = n_tasks)
        
        self.train_x = self.it.train_x
        self.train_y = self.it.train_y
        self.test_x = self.it.test_x
        self.test_y = self.it.test_y

        self.i = 0
        self.batch_size = batch_size

        self.n_tasks = n_tasks

        assert(n_tasks == 5)
        print("labels are 0/1, 2/3, 4/5, 6/7, 8/9")
        self.generate_tasks([
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
        ])
        self.img_fn = self.it.img_fn
        self.reshape_dims = (28*28,)
        self.switch_task(0)

    def append_concat(self, i, el, newdata):
        if not hasattr(self, 'all_tasks'):
            self.all_tasks = {}
        if i not in self.all_tasks:
            self.all_tasks[i] = {}
        if el not in self.all_tasks[i]:
            self.all_tasks[i][el] = np.copy(newdata)
        else:
            prev_shape = self.all_tasks[i][el].shape
            self.all_tasks[i][el] = \
                np.concatenate([
                    self.all_tasks[i][el],
                    np.copy(newdata)
                ], axis = 0)

    def generate_tasks(self, config):

        unique_labels = np.unique(self.test_y)
        n_tasks = len(config)
        for i, task in enumerate(config):
            for replace_label, allowed_label in enumerate(task):
                test_indices = np.where(self.test_y == allowed_label)[0]
                train_indices = np.where(self.train_y == allowed_label)[0]
                self.append_concat(i, 'train_x', self.train_x[train_indices])
                self.append_concat(i, 'train_y', self.train_y[train_indices])
                self.append_concat(i, 'test_x', self.test_x[test_indices])
                self.append_concat(i, 'test_y', self.test_y[test_indices])
                self.all_tasks[i]['train_y'][self.all_tasks[i]['train_y'] == allowed_label] = replace_label
                self.all_tasks[i]['test_y'][self.all_tasks[i]['test_y'] == allowed_label] = replace_label
            self.all_tasks[i]['train_x'] = np.reshape(self.all_tasks[i]['train_x'], [-1, 28*28])
            self.all_tasks[i]['test_x'] = np.reshape(self.all_tasks[i]['test_x'], [-1, 28*28])
            for key in self.all_tasks[i].keys():
                print("all_tasks[%d]['%s']: %s" % 
                    (i, key, self.all_tasks[i][key].shape))
    
    def switch_task(self, new_task_idx):

        assert(hasattr(self, 'all_tasks'))
        self.train_x = np.copy(self.all_tasks[new_task_idx]['train_x'])
        self.train_y = np.copy(self.all_tasks[new_task_idx]['train_y'])
        self.test_x = np.copy(self.all_tasks[new_task_idx]['test_x'])
        self.test_y = np.copy(self.all_tasks[new_task_idx]['test_y'])
        self.n = len(self.train_y)
        self.tn = len(self.test_y)
        self.i = 0
    
    def inspect(self):

        print('inspect split')

        r, c = self.n_tasks, len(np.unique(self.train_y))
        fig = plt.figure() # figsize = (r, c))
        subplot_i = 0
        
        for task in range(self.n_tasks):
            self.switch_task(task)
            classes_to_show = np.unique(self.test_y)
            all_indices = [np.where(self.test_y == class_num)[0] for class_num in classes_to_show]
            n_ex = [len(item) for item in all_indices]
            example_indices = [np.random.choice(item) for item in all_indices]
            examples = self.test_x[example_indices]

            for i, img_idx in enumerate(classes_to_show):
                ax = fig.add_subplot(r, c, subplot_i+1)
                ax.set_xticks(())
                ax.set_yticks(())
                label_human_readable = str(img_idx)
                img = examples[i]
                ax.set_xlabel(label_human_readable)
                plt.imshow(self.img_fn(img))
                subplot_i += 1

        plt.tight_layout(True)
        plt.savefig("inspect.png")
        plt.show()
        exit(0)

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