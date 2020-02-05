import os
from scipy import io as spio
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import GenericIterator

class SimEMNIST_Iterator:

    def __init__(self, config, inspect = False):
        self.apply_fn = None

        download_url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
        save_file = "emnist.zip"
        extract_dir = "emnist"
        main_data_file = "emnist/matlab/emnist-balanced.mat"
        if not os.path.exists(main_data_file):
            utils.download_and_unzip(download_url, save_file, extract_dir)
        extract_fn = lambda fname: spio.loadmat(fname)["dataset"]
        self.dataset = extract_fn(main_data_file)
        config = [[ord(item2) for item2 in item] for item in config]
        train_x_fn = lambda x: x[0][0][0][0][0][0]
        train_y_fn = lambda x: x[0][0][0][0][0][1]
        test_x_fn = lambda x: x[0][0][1][0][0][0]
        test_y_fn = lambda x: x[0][0][1][0][0][1]
        labels_fn = lambda: list(range(0x30, 0x39+1))+list(range(0x41, 0x5a+1))+list(range(0x61, 0x7a+1))
        reshape_dims = (784,)
        self.img_fn = lambda x: np.rot90(np.flipud(x.reshape(28, 28)), 3)

        self.trainx = train_x_fn(self.dataset)
        self.trainy = train_y_fn(self.dataset)
        self.testx = test_x_fn(self.dataset)
        self.testy = test_y_fn(self.dataset)
        self.labels = labels_fn()
        self.reshape_dims = reshape_dims

        self.config = config
        self.n_tasks = len(config)
        print("Custom dataset: n_tasks = %d" % self.n_tasks)
        self.n_classes = len(config[0])
        for task in config: assert(len(task) == self.n_classes)
        print("Custom dataset: n_classes = %d" % self.n_classes)

        self.tasks = []
        self.categorize(all = False, render = inspect)
        self.curr_idx = 0
        self.n = self.tasks[self.curr_idx].n
        self.tn = self.tasks[self.curr_idx].tn
        self.batch_size = self.tasks[self.curr_idx].batch_size
        self.train_x = self.tasks[self.curr_idx].train_x
        self.train_y = self.tasks[self.curr_idx].train_y
        self.test_x = self.tasks[self.curr_idx].test_x
        self.test_y = self.tasks[self.curr_idx].test_y

    def switch_task(self, new_task_idx):
        self.curr_idx = new_task_idx
        self.n = self.tasks[self.curr_idx].n
        self.tn = self.tasks[self.curr_idx].tn
        self.batch_size = self.tasks[self.curr_idx].batch_size
        self.train_x = self.tasks[self.curr_idx].train_x
        self.train_y = self.tasks[self.curr_idx].train_y
        self.test_x = self.tasks[self.curr_idx].test_x
        self.test_y = self.tasks[self.curr_idx].test_y
        # print('switch to %d: %s' % (new_task_idx, np.unique(self.test_y)))

    def __iter__(self):
        return self.tasks[self.curr_idx].__iter__()

    def __next__(self):
        return self.tasks[self.curr_idx].__next__()
    
    def test(self, samples = 32):
        return self.tasks[self.curr_idx].test(samples = samples)

    def categorize(self, all = True, render = False):
        classes_to_show = np.unique(self.testy)
        all_tr_indices = [np.where(self.trainy == class_num)[0] for class_num in classes_to_show]
        all_indices = [np.where(self.testy == class_num)[0] for class_num in classes_to_show]
        n_ex_tr = [len(item) for item in all_tr_indices]
        n_ex = [len(item) for item in all_indices]

        if render:
            example_indices = [np.random.choice(item) for item in all_indices]
            examples = self.testx[example_indices]

        if all:
            r, c = 8, 8
        else:
            r, c = self.n_tasks, self.n_classes
            new_classes = []
            for task in self.config:
                class_i = 0
                if self.apply_fn:
                    self.tasks.append(GenericIterator(preprocess_fn = self.apply_fn))
                else:
                    self.tasks.append(GenericIterator())
                for character in task:
                    labels_idx = self.labels.index(character)
                    new_classes.append(labels_idx)
                    print('%s: %d, %d' % (character, n_ex_tr[labels_idx], n_ex[labels_idx]))
                    x = self.trainx[all_tr_indices[labels_idx]]
                    y = [class_i for item in all_tr_indices[labels_idx]]
                    tx = self.testx[all_indices[labels_idx]]
                    ty = [class_i for item in all_indices[labels_idx]]
                    self.tasks[-1].add(x, y, tx, ty)
                    class_i += 1
                print("Added task data for: %s" % task)
                self.tasks[-1].finish()

            classes_to_show = new_classes

        if render:
            fig = plt.figure(figsize = (r, c))

        for i, img_idx in enumerate(classes_to_show):
            if render:
                ax = fig.add_subplot(r, c, i+1)
                ax.set_xticks(())
                ax.set_yticks(())
            label_human_readable = chr(self.labels[img_idx])
            if render:
                img = examples[img_idx]
            if all:
                print('%s: %d, %d' % (label_human_readable, n_ex_tr[img_idx], n_ex[img_idx]))
            if render:
                ax.set_xlabel(label_human_readable)
                plt.imshow(self.img_fn(img))

        if render:
            plt.tight_layout(True)
            plt.savefig("inspect.png")
            plt.show()
            exit(0)