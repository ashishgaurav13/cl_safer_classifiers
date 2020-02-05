import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import utils
from tqdm import tqdm
import numpy as np
import argparse
import os

class EWCGradientsNetwork(utils.Network):

    def __init__(self, layer_sizes, feature_extractor_needed = False, use_dropout = False,
        activation = 'relu', dropoutv = 0.5, reshape_dims = None, seed = 0,
        session_config = None, it = None, ewc_const = 100.0, reset_fisher = False,
        use_latest_theta_star = True, use_orig_loss = True, reset_opt = False, fix = 0.0,
        fisher_avg = False, newest_variance = False, less_compute = False, correctmask = False,
        lr = 0.0001, embedding = False):

        super(EWCGradientsNetwork, self).__init__(
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
        self.fix = fix

        self.fisher_avg = fisher_avg
        self.newest_variance = newest_variance
        self.less_compute = less_compute
        if self.less_compute:
            print("Need newest_variance for less compute")
            assert(self.newest_variance)

        self.correctmask = correctmask
        if self.correctmask:
            print("Need both newest_variance and less_compute for correctmask")
            assert(self.newest_variance and self.less_compute)

        self.vars['distances'] = {}
        self.all_masks = {}
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
        self.vars['distances'][0] = (loss, tf.constant(0.0))

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

        if self.correctmask and self.newest_variance:
            self.all_masks[n_task+1] = []
            init_ops = []
            for fdc in self.objs['fisher_diagcs']:
                self.all_masks[n_task+1] += [tf.Variable(tf.zeros_like(fdc), trainable = False)]
                init_ops += [self.all_masks[n_task+1][-1].initializer]
                # print("all_masks[%d] created from scratch" % (n_task+1))
            self.objs['sess'].run(init_ops)
            print("Created masks and for task %d" % (n_task+1))

            if n_task in self.all_masks:
                print("Copying previous values")
                for fi, fdc in enumerate(self.objs['fisher_diagcs']):
                    # s_mask = self.objs['sess'].run(tf.reduce_sum(self.all_masks[n_task][fi]))
                    # count_mask = self.objs['sess'].run(tf.reduce_sum(tf.ones_like(self.all_masks[n_task][fi])))
                    # print("all_masks[%d][%d]: %d/%d" % (n_task, fi, s_mask, count_mask))
                    self.all_masks[n_task+1][fi] = tf.cast(tf.math.logical_or(
                        tf.cast(self.all_masks[n_task+1][fi], tf.bool), tf.cast(self.all_masks[n_task][fi], tf.bool)
                    ), tf.float32)
                    s_mask = self.objs['sess'].run(tf.reduce_sum(self.all_masks[n_task+1][fi]))
                    count_mask = self.objs['sess'].run(tf.reduce_sum(tf.ones_like(self.all_masks[n_task+1][fi])))
                    print("all_masks[%d][%d]: %d/%d" % (n_task+1, fi, s_mask, count_mask))




    def reset_fisher_diag(self):

        print("Reset fishers")
        reset_ops = []
        for fd in self.objs['fisher_diags']:
            reset_ops += [tf.assign(fd, tf.zeros_like(fd))]
        self.objs['sess'].run(reset_ops)

    def update_fisher_diag(self, n_task):

        if self.reset_fisher:
            self.reset_fisher_diag()

        if self.newest_variance:
            self.reset_fisher_diag()
            print("Reset fisher because we need to compute fisher with latest logll")

        print("Reset fishers computed")
        reset_ops = []
        for fdc in self.objs['fisher_diagcs']:
            reset_ops += [tf.assign(fdc, tf.zeros_like(fdc))]
        self.objs['sess'].run(reset_ops)

        if self.newest_variance:
            print("Recompute fisher for tasks 0 ... n_task-1")
            total_examples = 0
            if self.less_compute:
                print("Using less compute")
                quotas = []
                for n_task_curr in range(n_task):
                    self.it.switch_task(n_task_curr)
                    quotas += [self.it.n]
                quotas = np.array(quotas) // n_task
                current_used = np.zeros_like(quotas)
                print("quotas: %s" % quotas)

            if self.correctmask:
                init_diagcs = []
                init_ops = []
                for fdc in self.objs['fisher_diagcs']:
                    init_diagcs += [tf.Variable(tf.zeros_like(fdc))]
                    init_ops += [tf.assign(init_diagcs[-1], fdc)] # should be zero anyways
                self.objs['sess'].run(init_ops)
                print("Created zero cum diagcs")

            # if n_task == 3: exit(0)

            for n_task_curr in range(n_task):
                print("Task %d" % n_task_curr)

                self.it.switch_task(n_task_curr)
                n_minibatches = self.it.n // self.fisher_batch_size
                self.it.i = 0
                for batch in range(n_minibatches):
                    if self.less_compute:
                        if current_used[n_task_curr] >= quotas[n_task_curr]:
                            break
                        else:
                            current_used[n_task_curr] += self.fisher_batch_size
                    nX, nY = next(self.it)
                    train_data = {self.phs['fisher_X']: nX, self.phs['fisher_Y']: nY}
                    self.objs['sess'].run(self.objs['fisher_sum_up_ops'], feed_dict = train_data)

                if self.correctmask:

                    fixfix = self.fix #* n_task
                    print("fix = %.2f" % fixfix)
                    # newv = self.objs['sess'].run(utils.sum_up(self.objs['fisher_diagcs']))
                    # print("nv: %d => %f" % (n_task_curr, newv))

                    all_ops = []
                    for fdc_save, fdc in zip(init_diagcs, self.objs['fisher_diagcs']):
                        all_ops += [tf.assign_add(fdc_save, fdc)]
                    self.objs['sess'].run(all_ops)
                    print("Assigned added diagcs to cum diagcs for task %d" % n_task_curr)

                    before_ones = tf.reduce_sum([tf.reduce_sum(item) for item in self.all_masks[n_task]])
                    before_ones = self.objs['sess'].run(before_ones)
                    for fi, fdc in enumerate(self.objs['fisher_diagcs']):
                        new_mask, _, _ = self.get_mask(tf.identity(fdc), fix = fixfix) # self.fix
                        new_mask = tf.cast(new_mask, tf.bool)
                        self.all_masks[n_task][fi] = tf.cast(tf.math.logical_or(
                            tf.cast(self.all_masks[n_task][fi], tf.bool), new_mask), tf.float32)
                    after_ones = tf.reduce_sum([tf.reduce_sum(item) for item in self.all_masks[n_task]])
                    after_ones = self.objs['sess'].run(after_ones)
                    total_params = tf.reduce_sum([tf.reduce_sum(tf.ones_like(item)) for item in self.all_masks[n_task]])
                    total_params = self.objs['sess'].run(total_params)
                    print("Mask: %d/%d -> %d/%d" % (before_ones, total_params, after_ones, total_params))
                    reset_ops = []
                    for fdc in self.objs['fisher_diagcs']:
                        reset_ops += [tf.assign(fdc, tf.zeros_like(fdc))]
                    self.objs['sess'].run(reset_ops)
                    print("Zeroed diagcs")


                newv = self.objs['sess'].run(utils.sum_up(self.objs['fisher_diagcs']))
                if self.less_compute:
                    total_examples += current_used[n_task_curr]
                else:
                    total_examples += n_minibatches * self.fisher_batch_size


            if self.correctmask:

                reassign_ops = []
                for fdc_save, fdc in zip(init_diagcs, self.objs['fisher_diagcs']):
                    reassign_ops += [tf.assign(fdc, fdc_save)]
                self.objs['sess'].run(reassign_ops)
                print("Assign cum diagcs to diagcs")


        else:
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

        if self.newest_variance:
            division_ops = []
            print("Total examples for fisher: %d" % total_examples)
            for fdc in self.objs['fisher_diagcs']:
                division_ops += [tf.assign(fdc, tf.divide(fdc, total_examples))]
            self.objs['sess'].run(division_ops)
        else:
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


    def get_mask(self, fd, fix):

        assert(0 <= fix <= 1)

        # How many params in fd? eg 1000
        total_num = int(self.objs['sess'].run(tf.reduce_sum(tf.ones_like(fd))))
        # How many to fix? eg 10% of 1000 = 100
        to_be_modified = int(fix * total_num)
        # threshold? eg minimum of these top 100 params?
        fd_topk_threshold = tf.reduce_min(tf.nn.top_k(tf.reshape(tf.identity(fd), [-1]), to_be_modified).values)
        # filtered fd mask? eg 1s for bottom 900, 0s for top 100
        fd_filtered = tf.cast(tf.less(tf.identity(fd), fd_topk_threshold), tf.float32)
        # how many can be really modified? eg 900
        num = int(self.objs['sess'].run(tf.reduce_sum(fd_filtered)))
        return fd_filtered, num, total_num


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
        if self.fisher_avg:
            ewc_penalty = tf.multiply(1.0/n_task, ewc_penalty)
        new_loss = tf.add(loss, tf.multiply(tf.constant(self.ewc_const, tf.float32), ewc_penalty))

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

        # op = self.objs['opt'].minimize(new_loss, var_list = orig_var_list)
        grads = self.objs['opt'].compute_gradients(new_loss, var_list = orig_var_list)
        new_grads = []

        if self.correctmask:
            temp_masks = []
            init_ops = []
            for fi, mask in enumerate(self.all_masks[n_task]):
                temp_masks += [tf.Variable(tf.zeros_like(mask), trainable = False)]
                init_ops += [tf.assign(temp_masks[fi], mask)]
            self.objs['sess'].run(init_ops)
            print("Created temp_masks")
            for gv, mask, fd in zip(grads, temp_masks, self.objs['fisher_diags']):
                grad, var = gv
                s = self.objs['sess'].run(tf.reduce_sum(fd))
                fd_filtered = tf.identity(mask)
                num = self.objs['sess'].run(tf.reduce_sum(fd_filtered))
                total_num = self.objs['sess'].run(tf.reduce_sum(tf.ones_like(fd_filtered)))
                print("%s => can modify %d/%d params, sum = %f" % (var.name, num, total_num, s))
                new_grad = tf.multiply(fd_filtered, grad)
                new_grads += [(new_grad, var)]
        else:
            for gv, fd in zip(grads, self.objs['fisher_diags']):
                grad, var = gv
                fd_filtered, num, total_num = self.get_mask(fd, fix = self.fix)
                s = self.objs['sess'].run(tf.reduce_sum(fd))
                print("%s => can modify %d/%d params, sum = %f" % (var.name, num, total_num, s))
                new_grad = tf.multiply(fd_filtered, grad)
                new_grads += [(new_grad, var)]

        op = self.objs['opt'].apply_gradients(new_grads)

        self.vars['train_op'] = op
        self.vars['train_ops'][n_task] = op

        print('Updated train_op and loss')

    def setup_distances(self, n_task):

        ce_loss = self.vars['losses'][0]

        penalties = []
        old_vars = self.saved_wts[n_task-1]
        fisher_vars = self.saved_fishers[n_task-1]
        for var, old_var, fisher in zip(self.objs['fisher_ws'], old_vars, fisher_vars):

            penalties += [
                tf.multiply(
                    fisher,
                    tf.square(tf.subtract(var, old_var))
                )
            ]
        ewc_penalty = tf.add_n([tf.reduce_sum(penalty) for penalty in penalties])
        ewc_penalty = tf.multiply(tf.constant(self.ewc_const, tf.float32), ewc_penalty)

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
