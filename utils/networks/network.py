import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils
from tqdm import tqdm
import numpy as np
import argparse
import os

class Network:

    def __init__(self, layer_sizes, feature_extractor_needed = False, use_dropout = False,
        activation = 'relu', dropoutv = 0.5, reshape_dims = None, seed = 0,
        session_config = None, it = None, embedding = False):

        assert(len(layer_sizes) == 4)
        assert(session_config != None)
        assert(it != None)
        self.layer_sizes = layer_sizes
        self.feature_extractor_needed = feature_extractor_needed
        self.use_dropout = use_dropout
        self.dropoutv = dropoutv
        self.reshape_dims = reshape_dims
        self.seed = seed
        self.session_config = session_config
        self.it = it
        self.embedding = embedding # if true, then expect x data to be
            # embeddings

        if self.use_dropout:
            self.glob_training_ph = tf.placeholder_with_default(False, shape=())
            self.training_ph = tf.placeholder_with_default(False, shape = ())

        self.activation = utils.get_activation(activation)

        print("Using feature extractor: %s" % self.feature_extractor_needed)
        print("Using dropout, bn: %s, %f" % (self.use_dropout, self.dropoutv))

        self.phs = {}
        self.vars = {}
        self.objs = {}
        self.all_predictions = []


    def apply_feature_extractor(self, X):

        if self.feature_extractor_needed:
            if not hasattr(self, 'feature_extractor_set'):
                with tf.variable_scope("feature_extractor"):
                    X, created_layers = utils.vgg16(X, self.training_ph)
                self.feature_extractor_set = True
                self.feature_extractor_layers = created_layers
            else:
                print("Reusing feature extractor")
                with tf.variable_scope("feature_extractor", reuse = True):
                    X = utils.vgg16_reuse(X, self.training_ph, self.feature_extractor_layers)
        else:
            X = tf.reshape(X, [-1, self.layer_sizes[0]])

        return X
    
    def get_trainable_vars(self, scope = "", silent = False):

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
        if self.feature_extractor_needed:
            new_var_list = []
            for item in var_list:
                if not item.name.startswith("feature_extractor"):
                    new_var_list += [item]
            var_list = new_var_list
        if not silent:
            print("Trainable vars: %s" % str(var_list))
        return var_list

    def create_session(self, improve_by = 5, min_epoch = 10):
        self.objs['saver'] = tf.train.Saver()
        # self.objs['sess'] = tf.Session(config = self.session_config)
        self.objs['sess'] = tf.InteractiveSession()
        self.objs['sess'].run(tf.global_variables_initializer())
        # (Disabled) Early Stopping
        # self.objs['es'] = utils.EarlyStopping(
        #     self.objs['sess'], 
        #     self.objs['saver'],
        #     save_dir = "saved_seed%d" % self.seed,
        #     improve_by = improve_by,
        #     min_epoch = min_epoch
        # )

        if self.feature_extractor_needed:
            if not os.path.exists("vgg16_cifar100"):
                print("Pretrained model doesnt exist for VGG16")
                print("Run cifar100.py first")
                exit(0)
            else:
                reqd_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "feature_extractor")
                feature_extractor_saver = tf.train.Saver(reqd_variables)
                print("Restoring feature extractor variables")
                feature_extractor_saver.restore(self.objs['sess'], "vgg16_cifar100/saved.ckpt")
                print("Done")

        # self.objs['sess'].graph.finalize()
    
    def print_vars(self, var_list, show_values = False, extra = False):
        for num, var in enumerate(var_list):
            print_strs = []
            if show_values:
                if 'sess' in self.objs:
                    red_sum = self.objs['sess'].run(tf.reduce_sum(var))
                    print_strs += ["mag %f" % (red_sum)]
                else:
                    print_strs += ["init"]
            if extra:
                if 'sess' in self.objs:
                    nonzerovar = tf.boolean_mask(var, tf.greater(var, 0.000001))
                    tmin = self.objs['sess'].run(tf.math.reduce_max(nonzerovar))
                    print_strs += ["tmax %f" % tmin]
                    nz = self.objs['sess'].run(tf.math.count_nonzero(var))
                    print_strs += ["nonzero %d" % nz]
                    num_elements = self.objs['sess'].run(tf.reduce_sum(tf.ones_like(var)))
                    print_strs += ["total %d" % num_elements]
            print_str = "\t(%d) %s" % (num+1, var.name)
            if len(print_strs) > 0:
                print_str += " => %s" % " , ".join(print_strs)
            print(print_str)
        print("Number of vars: %d" % len(var_list))

    def train_epoch(self, n_task, epoch, silent = False):
        
        n_minibatches = self.it.n // self.it.batch_size
        avg_tr_loss, avg_tr_acc = 0.0, 0.0
        for i in range(n_minibatches):
            nX, nY = next(self.it)
            train_data = {self.phs['X']: nX, self.phs['Y']: nY}
            if self.use_dropout:
                train_data[self.glob_training_ph] = True

            _, tr_loss, tr_acc = self.objs['sess'].run([
                self.vars['train_op'], self.vars['loss'],
                self.vars['acc']], feed_dict = train_data)
            avg_tr_loss += tr_loss / n_minibatches
            avg_tr_acc += tr_acc / n_minibatches

        test_data = {self.phs['X']: self.it.test_x, self.phs['Y']: self.it.test_y}
        test_acc = self.objs['sess'].run(self.vars['acc'], feed_dict = test_data)
        task_accs_all = self.accuracies(n_task+1)
        task_accs = np.average(task_accs_all)
        
        pred_n_tasks = self.predictions(n_task+1)
        self.all_predictions += [pred_n_tasks]
        behs = [self.beh(i) for i in range(n_task)] # Check beh on n_task-1 tasks
        beh_str = "Beh: " + str(behs)
        del(self.all_predictions[-1])

        print("Epoch: %d, Acc: %.2f%%, ValAcc: %.2f%%, Loss: %f" % (
            epoch+1, avg_tr_acc * 100.0, test_acc * 100.0, avg_tr_loss))
        if not silent:
            print("PastValAcc(%d): %s => %.2f%%" % (n_task+1,
                " ".join(["%.2f%%" % item for item in task_accs_all]), task_accs))
            print("%s" % beh_str)

        return task_accs     

    def accuracies(self, n):
        task_accuracies = []
        for i in range(n):
            self.it.switch_task(i)
            test_data = {self.phs['X']: self.it.test_x, self.phs['Y']: self.it.test_y}
            test_acc = self.objs['sess'].run(self.vars['acc'], feed_dict = test_data) * 100.0
            task_accuracies.append(test_acc)
        return task_accuracies

    def predictions(self, n):
        pred_n_tasks = []
        for i in range(n):
            self.it.switch_task(i)
            test_data = {self.phs['X']: self.it.test_x, self.phs['Y']: self.it.test_y}
            preds = self.objs['sess'].run(self.vars['predictions'], feed_dict = test_data)
            pred_n_tasks.append(preds)
        return pred_n_tasks

    def beh_show(self):
        print_str = "%d rows: " % len(self.all_predictions)
        print_str += "%s" % str([len(item) for item in self.all_predictions])
        print(print_str)

    # Behavior across n iterations
    def beh(self, i):
        if len(self.all_predictions) <= 1:
            print("Not enough data for behaviour analysis")
            return
        # self.beh_show()
        # print("Access rows %d:%d -> col %d" % (i, len(self.all_predictions)-1, i))
        chosen_classifications = [item[i] for item in self.all_predictions[i:]] # only consider data from index i+
        # n = len(chosen_classifications[0])
        # same = 0
        # for ii in range(n):
        #     reqd = chosen_classifications[0][ii]
        #     is_same = True
        #     for item in chosen_classifications[1:]:
        #         if item[ii] != reqd:
        #             is_same = False
        #             break
        #     if is_same:
        #         same += 1
        # return round(same*100.0/n, 2)
        preds = [list(item) for item in chosen_classifications]
        preds = list(zip(*preds))
        match = lambda x: len(set(x)) == 1 # x[0] == x[-1]
        preds = list(map(match, preds))
        return round(sum(preds)*100.0/len(preds), 2)

    def final_stats(self, n_task, n_permute_tasks):   
        task_accs = self.accuracies(n_permute_tasks)
        pred_n_tasks = self.predictions(n_permute_tasks)
        self.all_predictions += [pred_n_tasks]
        behs = [self.beh(i) for i in range(n_task)] # Check beh on n_task-1 tasks
        print("Final beh: " + str(behs))
        task_accs = ["%.2f%%" % item for item in task_accs]
        print("Task accuracies: " + " ".join(task_accs))
