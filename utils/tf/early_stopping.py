import tensorflow.compat.v1 as tf
import os
import shutil
import numpy as np

# https://github.com/tensorflow/tensorflow/issues/8425
def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session

class EarlyStopping:

    def __init__(self, sess, saver, improve_by = 10, min_epoch = 15, save_dir = "saved", mode = "acc"):
        self.sess = sess
        self.save_dir = save_dir
        print('Savedir: %s' % save_dir)
        self.saver = saver
        self.improve_by = improve_by
        self.min_epoch = min_epoch
        assert(mode in ["acc", "loss"])
        self.mode = mode

    def reset(self, mode = "acc"):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            print("shutil.rmtree: %s" % self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            print("os.mkdir: %s" % self.save_dir)
        self.val_accs = []
        self.epoch_num = 0
        print('EarlyStopping: reset, improve_by = %d' % self.improve_by)
        self.saver.save(self.sess, os.path.join(self.save_dir, "%d.ckpt" % self.epoch_num))
        assert(mode in ["acc", "loss"])
        self.mode = mode
        if self.mode == "acc":
            self.best_val_acc = 0.0
            self.best_val_acc_idx = 0
        else:
            self.best_val_acc = np.inf
            self.best_val_acc_idx = 0
        self.since = 0

    def add_acc(self, x):
        self.val_accs.append(x)
        self.epoch_num += 1
        assert(len(self.val_accs) == self.epoch_num)
        if self.epoch_num < self.min_epoch:
            return -1
        elif self.epoch_num == self.min_epoch:
            print("Min epochs completed.")
        check = x >= self.best_val_acc if self.mode == "acc" else x <= self.best_val_acc
        if check:
            self.saver.save(self.sess, os.path.join(self.save_dir, "%d.ckpt" % self.epoch_num))
            self.best_val_acc = x
            self.best_val_acc_idx = self.epoch_num
            self.since = 0
        else:
            self.since += 1
        if self.epoch_num < self.min_epoch:
            return -1
        else:
            if self.since > self.improve_by:
                print("Early stopping; Restoring to epoch %d with %s %.2f" % (
                    self.best_val_acc_idx, self.mode, self.best_val_acc))
                best_path = os.path.join(self.save_dir, "%d.ckpt" % self.best_val_acc_idx)
                self.saver.restore(self.sess, best_path)
                shutil.rmtree(self.save_dir)
                print("shutil.rmtree: %s" % self.save_dir)
                return self.best_val_acc_idx
            else:
                return -1
