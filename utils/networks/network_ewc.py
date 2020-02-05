import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils
from tqdm import tqdm
import numpy as np
import argparse
import os

class EWCNetwork(utils.Network):

    def __init__(self, layer_sizes, feature_extractor_needed = False, use_dropout = False,
        activation = 'relu', dropoutv = 0.5, reshape_dims = None, seed = 0, 
        session_config = None, it = None, ewc_const = 100.0, reset_fisher = False,
        use_latest_theta_star = True, use_orig_loss = True, reset_opt = False,
        fisher_avg = False, lr = 0.0001, embedding = False):

        super(EWCNetwork, self).__init__(
            layer_sizes = layer_sizes,
            feature_extractor_needed = feature_extractor_needed,
            use_dropout = use_dropout,
            activation = activation,
            dropoutv = dropoutv,
            reshape_dims = reshape_dims,
            seed = seed,
            session_config = session_config,
            it = it,
            embedding = embedding,
        )
        self.ewc_const = ewc_const
        print("EWC const: %f" % ewc_const)
        self.reset_fisher = reset_fisher
        self.use_latest_theta_star = use_latest_theta_star
        self.use_orig_loss = use_orig_loss
        self.reset_opt = reset_opt

        self.fisher_batch_size = self.it.batch_size
        self.saved_wts = {}
        self.saved_fishers = {}
        self.fisher_avg = fisher_avg
        
        self.lr = lr

    def get_trainable_vars(self, scope = ""):

        var_list = super().get_trainable_vars(scope, silent = True)
        new_var_list = []
        for item in var_list:
            if not item.name.startswith("fisher"):
                new_var_list += [item]
        var_list = new_var_list
        # print("Trainable vars: %s" % str(var_list))
        return var_list

    def setup_phs(self):

        dtype = tf.uint8
        if self.embedding: dtype = tf.float32
        self.phs['X'] = tf.placeholder(dtype, [None, *self.reshape_dims])
        self.phs['Y'] = tf.placeholder(tf.uint8, [None])
        self.phs['fisher_X'] = tf.placeholder(tf.uint8, [self.fisher_batch_size, *self.reshape_dims])
        self.phs['fisher_Y'] = tf.placeholder(tf.uint8, [self.fisher_batch_size])

    def forward(self):

        X = self.phs['X']
        if not self.embedding: X = tf.cast(X, tf.float32) * (1.0 / 255)
        layer = self.apply_feature_extractor(X)
        fisher_ws = []
        fisher_diags = []
        fisher_diagcs = []
        fisher_old_ws = []

        n_layers = len(self.layer_sizes)-1
        for i in range(n_layers):

            layer_name = "d%d" % (i+1)

            layer = utils.dense2(layer, self.layer_sizes[i], self.layer_sizes[i+1], name = layer_name)
            print('Applied dense (%d, %d) of name %s' % (self.layer_sizes[i],
                self.layer_sizes[i+1], layer_name))

            w = utils.get_var("%s/w" % layer_name)
            fisher_w_name = "fisher_diag_%s_w" % layer_name
            fisher_wc_name = "fisher_diag_%s_wc" % layer_name
            fisher_old_w_name = "fisher_old_%s_w" % layer_name
            self.vars[fisher_w_name] = tf.Variable(tf.zeros_like(w), name = fisher_w_name)
            self.vars[fisher_wc_name] = tf.Variable(tf.zeros_like(w), name = fisher_wc_name)
            self.vars[fisher_old_w_name] = tf.Variable(tf.zeros_like(w), name = fisher_old_w_name)
            fisher_ws += [w]
            fisher_diags += [self.vars[fisher_w_name]]
            fisher_diagcs += [self.vars[fisher_wc_name]]
            fisher_old_ws += [self.vars[fisher_old_w_name]]

            b = utils.get_var("%s/b" % layer_name)
            fisher_b_name = "fisher_diag_%s_b" % layer_name
            fisher_bc_name = "fisher_diag_%s_bc" % layer_name
            fisher_old_b_name = "fisher_old_%s_b" % layer_name
            self.vars[fisher_b_name] = tf.Variable(tf.zeros_like(b), name = fisher_b_name)
            self.vars[fisher_bc_name] = tf.Variable(tf.zeros_like(b), name = fisher_bc_name)
            self.vars[fisher_old_b_name] = tf.Variable(tf.zeros_like(b), name = fisher_old_b_name)
            fisher_ws += [b]
            fisher_diags += [self.vars[fisher_b_name]]
            fisher_diagcs += [self.vars[fisher_bc_name]]
            fisher_old_ws += [self.vars[fisher_old_b_name]]


            print('Created zero fishers')

            if i+1 != len(self.layer_sizes)-1:
                if self.use_dropout:
                    layer = self.activation(layer)
                    layer = tf.keras.layers.Dropout(
                        rate = self.dropoutv, 
                        seed = self.seed)(layer, training = self.glob_training_ph)
                    print('Applied activation -> dropout')
                else:
                    layer = self.activation(layer)
                    print('Applied activation')

        self.vars['fX'] = layer
        self.objs['fisher_ws'] = fisher_ws
        self.objs['fisher_diagcs'] = fisher_diagcs
        self.objs['fisher_diags'] = fisher_diags
        self.objs['fisher_old_ws'] = fisher_old_ws

        # Create fisher graph
        print('Creating fisher batch_log_likelihood')

        fisher_X = tf.cast(self.phs['fisher_X'], tf.float32) * (1.0 / 255)
        fisher_Y = tf.one_hot(self.phs['fisher_Y'], depth = self.layer_sizes[-1], dtype = tf.float32)

        if self.feature_extractor_needed:
            fisher_X = self.apply_feature_extractor(fisher_X)
            fisher_Xs = [
                tf.reshape(fx, shape = (1, self.layer_sizes[0]))
                for fx in tf.unstack(fisher_X, num = self.fisher_batch_size, axis = 0)
            ]
        else:
            fisher_Xs = [
                tf.reshape(fx, shape = (1, *self.it.reshape_dims))
                for fx in tf.unstack(fisher_X, num = self.fisher_batch_size, axis = 0)
            ]

        fisher_Ys = tf.unstack(fisher_Y, num = self.fisher_batch_size, axis = 0)

        log_likelihoods = []
        fisher_var_lists = []

        for i in range(self.fisher_batch_size):
            
            raw_output = fisher_Xs[i]

            fisher_var_list = []
            for j in range(n_layers):

                layer_name = "d%d" % (j+1)

                w = tf.identity(utils.get_var("%s/w" % layer_name))
                b = tf.identity(utils.get_var("%s/b" % layer_name))
                fisher_var_list += [w, b]
                raw_output = tf.add(tf.matmul(raw_output, w), b)

                if j+1 != len(self.layer_sizes)-1:
                
                    raw_output = self.activation(raw_output)
                    # No dropout; TODO

            log_likelihood = tf.multiply(fisher_Ys[i], tf.nn.log_softmax(raw_output))
            log_likelihoods += [log_likelihood]
            fisher_var_lists += [fisher_var_list]

        batch_log_likelihood = tf.reduce_sum(log_likelihoods)
        self.vars['batch_log_likelihood'] = batch_log_likelihood
        self.objs['fisher_var_lists'] = fisher_var_lists
    
    def backward(self):

        assert('X' in self.phs)
        assert('Y' in self.phs)
        assert('fX' in self.vars)
        fX = self.vars['fX']
        Y = self.phs['Y']

        Y_one_hot = tf.one_hot(Y, depth = self.layer_sizes[-1], dtype = tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fX, labels = Y_one_hot)
        loss = tf.reduce_mean(loss)
        self.vars['loss'] = loss
        self.vars['losses'] = {}
        self.vars['losses'][0] = loss

        var_list = self.get_trainable_vars()
        self.vars['orig_var_list'] = var_list

        # Fisher stuff
        print('Creating fisher ops')
        fisher_current = [utils.get_var("fisher_diag_%sc" % var.name.split(":")[0].replace("/", "_")) for var in var_list]
        grads = [tf.gradients(self.vars['batch_log_likelihood'], var_list) for var_list in self.objs['fisher_var_lists']]
        fisher_delta = []
        for i in range(len(self.objs['fisher_ws'])):
            fisher_delta += [tf.add_n([tf.square(g[i]) for g in grads])]

        fisher_sum_up_ops = [tf.assign_add(fc, fd) for fc, fd in zip(fisher_current, fisher_delta)]
        self.objs['fisher_sum_up_ops'] = fisher_sum_up_ops

        opt = tf.train.AdamOptimizer(self.lr)
        self.objs['opt'] = opt

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        # print("Trainable vars: %s" % str(var_list))
        print("Trainable vars:")
        self.print_vars(var_list)
        op = self.objs['opt'].minimize(loss, var_list = var_list)
        self.vars['train_op'] = op
        self.vars['train_ops'] = {}
        self.vars['train_ops'][0] = op

        predictions = tf.argmax(tf.nn.softmax(fX), axis = 1)
        predictions = tf.cast(predictions, tf.uint8) # cast to uint8, like Y
        self.vars['predictions'] = predictions

        acc = tf.equal(Y, predictions)
        acc = tf.cast(acc, tf.float32) # For averaging, first cast bool to float32
        acc = tf.reduce_mean(acc)
        self.vars['acc'] = acc

    def save_task_weights(self, n_task):

        print("Saving weights")
        save_ops = []
        for oldw, w in zip(self.objs['fisher_old_ws'], self.objs['fisher_ws']):
            save_ops += [tf.assign(oldw, w)]
        self.objs['sess'].run(save_ops)
        self.saved_wts[n_task] = [] # say task 0
        save_ops = []
        for oldw in self.objs['fisher_old_ws']:
            self.saved_wts[n_task] += [tf.Variable(tf.zeros_like(oldw))]
            save_ops += [tf.assign(self.saved_wts[n_task][-1], oldw)]
        self.objs['sess'].run(save_ops)
        print("Saved wts for task %d" % (n_task))
    
    def reset_fisher_diag(self):

        print("Reset fishers")
        reset_ops = []
        for fd in self.objs['fisher_diags']:
            reset_ops += [tf.assign(fd, tf.zeros_like(fd))]
        self.objs['sess'].run(reset_ops)
    
    def update_fisher_diag(self, n_task):

        if self.reset_fisher:
            self.reset_fisher_diag()

        print("Reset fishers computed")
        reset_ops = []
        for fdc in self.objs['fisher_diagcs']:
            reset_ops += [tf.assign(fdc, tf.zeros_like(fdc))]
        self.objs['sess'].run(reset_ops)

        n_minibatches = self.it.n // self.fisher_batch_size
        self.it.i = 0
        orig = self.objs['sess'].run(utils.sum_up(self.objs['fisher_diagcs']))
        # imgs_sum = [] 
        for batch in range(n_minibatches):
            # print("Batch %d" % batch)
            nX, nY = next(self.it)
            # imgs_sum += [np.sum(nY)]
            train_data = {self.phs['fisher_X']: nX, self.phs['fisher_Y']: nY}
            self.objs['sess'].run(self.objs['fisher_sum_up_ops'], feed_dict = train_data)
            # print(self.objs['sess'].run(self.objs['fisher_diagcs'][0])[0][0])
        newv = self.objs['sess'].run(utils.sum_up(self.objs['fisher_diagcs']))
        # print(orig, newv, n_minibatches, self.fisher_batch_size)
        # print(imgs_sum)
        print('Ran fisher_sum_up_ops (examples: %d)' % (n_minibatches * self.fisher_batch_size))

        division_ops = []
        for fdc in self.objs['fisher_diagcs']:
            division_ops += [tf.assign(fdc, tf.divide(fdc, n_minibatches * self.fisher_batch_size))]
        self.objs['sess'].run(division_ops)

        shown_vars = self.objs['fisher_diags']
        orig = self.objs['sess'].run(utils.sum_up(self.objs['fisher_diags']))
        origs = ["%.2f" % orig]
        assign_ops = []
        for fdc, fd in zip(self.objs['fisher_diagcs'], self.objs['fisher_diags']):
            assign_ops += [tf.assign_add(fd, fdc)]
        self.objs['sess'].run(assign_ops)
        newv = self.objs['sess'].run(utils.sum_up(self.objs['fisher_diags']))
        newvs = ["%.2f" % newv]
        print("changed %s => %s" % (" , ".join(origs), " , ".join(newvs)))
        # print("SHOWN:")
        # self.print_vars(shown_vars)

        self.saved_fishers[n_task-1] = [] # say task 0
        save_ops = []
        for fd in self.objs['fisher_diags']:
            self.saved_fishers[n_task-1] += [tf.Variable(tf.zeros_like(fd))]
            save_ops += [tf.assign(self.saved_fishers[n_task-1][-1], fd)]
        self.objs['sess'].run(save_ops)
        print("Saved fishers for task %d" % (n_task-1))
        
    def update_loss(self, n_task):

        if n_task == 0:
            return

        loss = self.vars['losses'][0] if self.use_orig_loss else self.vars['losses'][n_task-1]

        penalties = []
        old_vars = self.objs['fisher_old_ws'] if self.use_latest_theta_star else self.saved_wts[n_task-1]
        fisher_vars = self.objs['fisher_diags'] if self.use_latest_theta_star else self.saved_fishers[n_task-1]
        for var, old_var, fisher in zip(self.objs['fisher_ws'], old_vars, fisher_vars):

            penalties += [
                tf.multiply(
                    fisher,
                    tf.square(tf.subtract(var, old_var))
                )
            ]
        
        ewc_penalty = tf.add_n([tf.reduce_sum(penalty) for penalty in penalties])
        new_loss = tf.add(loss, tf.multiply(tf.constant(self.ewc_const, tf.float32), ewc_penalty))

        self.vars['loss'] = new_loss
        self.vars['losses'][n_task] = new_loss

        orig_var_list = self.vars['orig_var_list']
        # print("Trainable vars: %s" % str(orig_var_list))
        print("Trainable vars:")
        self.print_vars(orig_var_list)
        if self.reset_opt:
            print('Reset opt')
            self.objs['sess'].run(tf.variables_initializer(self.objs['opt'].variables()))
        op = self.objs['opt'].minimize(new_loss, var_list = orig_var_list)
        self.vars['train_op'] = op
        self.vars['train_ops'][n_task] = op

        print('Updated train_op and loss')