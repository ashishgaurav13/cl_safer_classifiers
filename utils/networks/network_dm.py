import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils
from tqdm import tqdm
import numpy as np
import argparse
import os

class DMNetwork(utils.Network):

    def __init__(self, layer_sizes, feature_extractor_needed = False, use_dropout = False,
        activation = 'relu', dropoutv = 0.5, reshape_dims = None, seed = 0,
        session_config = None, it = None, multiplier = 100.0,
        use_latest_theta_star = True, use_orig_loss = True, reset_opt = False,
        lr = 0.0001, c1 = None, c2 = None, version = 'case3', norm = 'l1', embedding = False):

        super(DMNetwork, self).__init__(
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
        self.multiplier = multiplier
        print("Multiplier: %f" % multiplier)
        self.use_latest_theta_star = use_latest_theta_star
        self.use_orig_loss = use_orig_loss
        self.reset_opt = reset_opt

        self.fisher_batch_size = self.it.batch_size
        self.saved_wts = {}
        self.saved_fishers = {}

        self.vars['distances'] = {}
        self.lr = lr

        self.init_change = c1
        self.incr_change = c2

        self.version = version
        self.norm = norm
        assert(self.version in ['case1', 'case2', 'case3', 'case4'])
        assert(self.norm in ['l1', 'l2'] or self.elastic_norm(self.norm) != -1)
        if self.elastic_norm(self.norm) != -1:
            alpha = self.elastic_norm(self.norm)
            self.norm_op = lambda x: (1-alpha)*tf.math.abs(x)+alpha*tf.math.square(x)
        else:
            self.norm_op = {
                'l1': lambda x: tf.math.abs(x),
                'l2': lambda x: tf.math.square(x),
            }[self.norm]

    def elastic_norm(self, x):
        try:
            assert(x[0] == 'e')
            ret = float(x[1:])
            assert(0 <= ret <= 1)
        except:
            return -1
        else:
            return ret

    def indicator(self, x):
        return tf.cast(tf.math.greater(x, 0), tf.float32)
        # return (0.5 * tf.math.sign(x) + 0.5)

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

        fisher_var_lists = []

        # Classwise direct predicted likelihoods and direct predicted likelihoods
        # Case I, II, III, IV
        nout = self.layer_sizes[-1]
        onehots_n = tf.unstack(tf.one_hot(list(range(nout)), nout), num = nout, axis = 0)
        if self.version == 'case1':
            jlikelihoods = {ii: [] for ii in range(nout)}
        if self.version == 'case2':
            jlikelihoodsqs = {ii: [] for ii in range(nout)}
        if self.version == 'case3':
            likelihoods = []
        if self.version == 'case4':
            likelihoodsqs = []

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

            fisher_var_lists += [fisher_var_list]

            # Case I, II, III, IV
            if self.version == 'case1':
                for key in jlikelihoods.keys():
                    jlikelihoods[key] += [tf.multiply(
                        onehots_n[key],
                        tf.nn.softmax(raw_output)
                    )]
            if self.version == 'case2':
                for key in jlikelihoodsqs.keys():
                    jlikelihoodsqs[key] += [tf.square(tf.multiply(
                        onehots_n[key],
                        tf.nn.softmax(raw_output)
                    ))]
            if self.version == 'case3':
                likelihood = tf.multiply(fisher_Ys[i], tf.nn.softmax(raw_output))
                likelihoods += [likelihood]
            if self.version == 'case4':
                likelihood = tf.multiply(fisher_Ys[i], tf.nn.softmax(raw_output))
                likelihoodsq = tf.square(likelihood) 
                likelihoodsqs += [likelihoodsq]

        self.objs['fisher_var_lists'] = fisher_var_lists

        # Finally, reduce_sum and add to vars
        if self.version == 'case1':
            jbatch_likelihood = {key: tf.reduce_sum(jlikelihoods[key]) for key in jlikelihoods.keys()}
            self.vars['jbatch_likelihood'] = jbatch_likelihood
        if self.version == 'case2':
            jbatch_likelihoodsq = {key: tf.multiply(tf.constant(0.5), tf.reduce_sum(jlikelihoodsqs[key])) for key in jlikelihoodsqs.keys()}    
            self.vars['jbatch_likelihoodsq'] = jbatch_likelihoodsq
        if self.version == 'case3':
            batch_likelihood = tf.reduce_sum(likelihoods)
            self.vars['batch_likelihood'] = batch_likelihood
        if self.version == 'case4':
            batch_likelihoodsq = tf.multiply(tf.constant(0.5), tf.reduce_sum(likelihoodsqs))
            self.vars['batch_likelihoodsq'] = batch_likelihoodsq


    def backward(self):

        assert('X' in self.phs)
        assert('Y' in self.phs)
        assert('fX' in self.vars)
        fX = self.vars['fX']
        Y = self.phs['Y']

        Y_one_hot = tf.one_hot(Y, depth = self.layer_sizes[-1], dtype = tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fX, labels = Y_one_hot)
        loss = tf.reduce_mean(loss)
        self.vars['ce_losses'] = {}
        self.vars['loss'] = loss
        self.vars['losses'] = {}
        self.vars['losses'][0] = loss
        self.vars['ce_losses'][0] = loss
        self.vars['distances'][0] = (loss, tf.constant(0.0))

        var_list = self.get_trainable_vars()
        self.vars['orig_var_list'] = var_list

        # Fisher stuff
        print('Creating fisher ops')
        fisher_current = [utils.get_var("fisher_diag_%sc" % var.name.split(":")[0].replace("/", "_")) for var in var_list]
        if self.version in ['case1', 'case2']:
            relevant_var = {
                'case1': 'jbatch_likelihood',
                'case2': 'jbatch_likelihoodsq',
            }[self.version]
            grads = {
                key: [tf.gradients(self.vars[relevant_var][key], var_list) for var_list in self.objs['fisher_var_lists']] for key in self.vars[relevant_var].keys()
            }
            fisher_delta = {key: [] for key in grads.keys()}
            for i in range(len(self.objs['fisher_ws'])):
                for key in grads.keys():
                    fisher_delta[key] += [tf.add_n([self.norm_op(g[i]) for g in grads[key]])]
            fisher_delta = [tf.reduce_mean(item, axis=0) \
                for item in zip(*[v for v in fisher_delta.values()])]
        if self.version in ['case3', 'case4']:
            relevant_var = {
                'case3': 'batch_likelihood',
                'case4': 'batch_likelihoodsq',
            }[self.version]
            grads = [tf.gradients(self.vars[relevant_var], var_list) for var_list in self.objs['fisher_var_lists']]
            fisher_delta = []
            for i in range(len(self.objs['fisher_ws'])):
                fisher_delta += [tf.add_n([self.norm_op(g[i]) for g in grads])]
        
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

        # Reset is mandatory
        print('Mandatory fisher diagonal reset')
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

        # Radius of influence (Constrained minimization)
        if self.init_change != None and self.incr_change != None:
            task_var = tf.Variable(self.init_change, name = "epsilon_task%d" % (n_task-1), trainable = False)
            self.objs['sess'].run(task_var.initializer)
            self.vars['epsilon_task%d' % (n_task-1)] = task_var
            for prev_n_task in range(n_task-1): self.objs['sess'].run(tf.assign_add(self.vars['epsilon_task%d' % (prev_n_task)], self.incr_change))

        loss = self.vars['losses'][0] if self.use_orig_loss else self.vars['losses'][n_task-1]

        penalties = []
        old_vars = self.objs['fisher_old_ws'] if self.use_latest_theta_star else self.saved_wts[n_task-1]
        fisher_vars = self.objs['fisher_diags'] if self.use_latest_theta_star else self.saved_fishers[n_task-1]
        for var, old_var, fisher in zip(self.objs['fisher_ws'], old_vars, fisher_vars):

            penalties += [
                tf.multiply(
                    fisher,
                    self.norm_op(tf.subtract(var, old_var))
                )
            ]

        ewc_penalty = tf.add_n([tf.reduce_sum(penalty) for penalty in penalties])
        # Create new cross entropy loss
        if self.init_change != None and self.incr_change != None:
            self.vars['ce_losses'][n_task] = self.vars['ce_losses'][n_task-1] * self.indicator(task_var-ewc_penalty)
        new_loss = tf.add(loss, tf.multiply(tf.constant(self.multiplier, tf.float32), ewc_penalty))

        # Remove previous CE loss
        if self.init_change != None and self.incr_change != None:
            print('lol')
            new_loss -= self.vars['ce_losses'][n_task-1]
            new_loss += self.vars['ce_losses'][n_task]

        self.vars['loss'] = new_loss
        self.vars['losses'][n_task] = new_loss
        self.vars['distances'][n_task] = self.setup_distances(n_task)

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

    def setup_distances(self, n_task):

        ce_loss = self.vars['losses'][0]

        penalties = []
        old_vars = self.saved_wts[n_task-1]
        fisher_vars = self.saved_fishers[n_task-1]
        for var, old_var, fisher in zip(self.objs['fisher_ws'], old_vars, fisher_vars):

            # Since this is a KL measurement, this should be tf.square rather than
            # self.norm_op
            penalties += [
                tf.multiply(
                    fisher,
                    tf.square(tf.subtract(var, old_var))
                )
            ]
        ewc_penalty = tf.add_n([tf.reduce_sum(penalty) for penalty in penalties])
        ewc_penalty = tf.multiply(tf.constant(self.multiplier, tf.float32), ewc_penalty)

        print("Created dll and df for task %d" % n_task)
        return (ce_loss, ewc_penalty)

    def eval_distances(self, n_task, max_tasks, begin):

        for n_task_curr in range(max_tasks):

            if n_task_curr in self.vars['distances']:

                self.it.switch_task(n_task_curr)
                test_data = {self.phs['X']: self.it.test_x, self.phs['Y']: self.it.test_y}
                dll = self.objs['sess'].run(self.vars['distances'][n_task_curr][0], feed_dict = test_data)
                df = self.objs['sess'].run(self.vars['distances'][n_task_curr][1], feed_dict = test_data)
                print("Train %d %s => metrics for %d => DLL: %f, DF: %f" % (n_task, "begin" if begin else "end",
                    n_task_curr, dll, df))


            else:

                dll, df = 0.0, 0.0
                print("Train %d %s => metrics for %d => DLL: %f, DF: %f => does not exist" % (n_task, "begin" if begin else "end",
                    n_task_curr, dll, df))
