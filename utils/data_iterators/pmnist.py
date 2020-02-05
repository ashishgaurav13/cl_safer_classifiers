import numpy as np
import matplotlib.pyplot as plt
import cv2

class PMNIST_DataIterator:

    def __init__(self, train_data, test_data, batch_size = 32, 
        randomize = True, n_tasks = 5):
        
        self.train_x, self.train_y = train_data
        self.n = len(self.train_y)
        print('Training examples = %d' % self.n)
        self.test_x, self.test_y = test_data
        self.tn = len(self.test_y)
        print('Test examples = %d' % self.tn)
        self.i = 0
        self.batch_size = batch_size
        self.reshape_dims = (28*28,)
        print('Batch size = %d' % self.batch_size)
        if randomize: 
            idx = np.random.permutation(self.n)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
            print('Shuffled training data')
        self.orig_data = (np.copy(self.train_x), np.copy(self.train_y),
            np.copy(self.test_x), np.copy(self.test_y))
        
        self.n_tasks = n_tasks
        self.img_fn = lambda x: cv2.cvtColor(np.reshape(x, (28, 28)), cv2.COLOR_GRAY2RGB)

    def inspect(self):

        print('inspect')

        r, c = self.n_tasks, 10
        fig = plt.figure(figsize = (r, c))
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
                img = examples[img_idx]
                ax.set_xlabel(label_human_readable)
                plt.imshow(self.img_fn(img), cmap='gray', interpolation='none')
                subplot_i += 1

        plt.tight_layout(True)
        plt.savefig("inspect.png")
        plt.show()
        exit(0)

    def set_permutes(self, permutes):
        self.permutes = permutes
    
    def switch_task(self, new_task_idx):
        assert(hasattr(self, 'permutes'))
        self.apply_permute(self.permutes, new_task_idx)

    def apply_permute(self, permutes, pidx):
        # print('Restoring data')
        self.train_x, self.train_y, self.test_x, self.test_y = \
            [np.copy(item) for item in self.orig_data]
        # print('Applying permute %d' % pidx)
        self.train_x = np.reshape(self.train_x, [-1, 28*28])
        self.test_x = np.reshape(self.test_x, [-1, 28*28])
        self.train_x = np.take(self.train_x, permutes[pidx], axis=-1)
        self.test_x = np.take(self.test_x, permutes[pidx], axis=-1)
        self.train_x = np.reshape(self.train_x, [-1, 28, 28])
        self.test_x = np.reshape(self.test_x, [-1, 28, 28])
        # print('Applied')
        self.train_x = np.reshape(self.train_x, [-1, *self.reshape_dims])
        self.test_x = np.reshape(self.test_x, [-1, *self.reshape_dims])

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