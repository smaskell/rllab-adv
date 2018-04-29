from collections import deque

import keras
from itertools import islice
from random import sample

from rllab.misc import ext
import numpy as np
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer


class Replay_Memory():

    def __init__(self, memory_size=50000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return sample(self.memory, min(batch_size, len(self.memory)))

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)

    def take(self, n):
        length = len(self.memory)
        n = min(n, length)
        return list(islice(self.memory, length - n, length))


HIDDEN_LAYER_SIZE = 30
fear_radius = 20
fear_fade_in = 100000
fear_factor = 5.0

class DangerModel:
    def __init__(self, state_size):
        self.model = self.create_model(state_size)
        self.model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
        self.state_size = state_size

    @staticmethod
    def create_model(state_size):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(HIDDEN_LAYER_SIZE, batch_input_shape=(None, state_size), activation="relu",
                                     kernel_initializer="glorot_normal", bias_initializer="zeros"))
        model.add(keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu",
                                     kernel_initializer="glorot_normal", bias_initializer="zeros"))
        model.add(keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu",
                                     kernel_initializer="glorot_normal", bias_initializer="zeros"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        return model

    def train(self, states, values):
        self.model.train_on_batch(x=states, y=values)

    def predict(self, state):
        return self.model.predict(state.reshape(1, self.state_size))[0]

    def save_model_weights(self, model_file):
        # Helper function to save your model / weights.
        self.model.save_weights(model_file)

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        print('loading weights')
        self.model.load_weights(weight_file)


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            use_danger=False,
            **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.use_danger = use_danger
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.good_states = []
        self.danger_states = []
        self.total_actions = 0
        if self.use_danger:
            self.danger_model = DangerModel(4)
        else:
            self.danger_model = None
        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, is_protagonist=True):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        if is_protagonist == True:
            action_var = self.env.pro_action_space.new_tensor_variable(
                'action',
                extra_dims=1 + is_recurrent,
            )
        else:
            action_var = self.env.adv_action_space.new_tensor_variable(
                'action',
                extra_dims=1 + is_recurrent,
            )

        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def add_danger(self, samples_data):
        if not self.use_danger:
            return
        observations = []

        # print("collecting observations from", len(samples_data['paths']), "paths")
        for path in samples_data['paths']:
            # print("adding", len(path['observations']), "observations")
            observations.append(path['observations'])

        old_advantage = samples_data['advantages']
        advantage = []

        actions = 0
        for obvs_list in observations:
            if len(obvs_list) < 200:
                good = obvs_list[:-fear_radius]
                bad = obvs_list[-fear_radius:]
                self.danger_states.extend(bad)
                self.good_states.extend(good)
            else:
                self.good_states.extend(obvs_list)

            for observation in obvs_list[:-1]:
                adv = old_advantage[actions]
                actions += 1
                self.total_actions += 1
                scaled_fear_factor = min(fear_factor, fear_factor * self.total_actions / fear_fade_in)
                fear = scaled_fear_factor * self.danger_model.predict(observation)[0]
                advantage.append(adv - fear)
            # last action has no fear
            advantage.append(old_advantage[actions])
            actions += 1
            self.total_actions += 1

        # index = 0
        # for old, new in zip(old_advantage, advantage):
        #     if old != new:
        #         print(old, "!=", new, "at index", index)
        #     index+=1
        # print("matches?", np.all(np.array(old_advantage) == np.array(advantage)))

        samples_data['advantages'] = np.array(advantage)

        train_good = sample(self.good_states, min(len(self.good_states), 64))
        train_danger = sample(self.danger_states, min(len(self.danger_states), 64))
        training_set = np.zeros((len(train_good) + len(train_danger), 5))
        training_set[:len(train_good), :-1] = train_good
        training_set[len(train_good):, :-1] = train_danger
        training_set[len(train_good):, -1] = np.ones(len(train_danger))
        np.random.shuffle(training_set)

        self.danger_model.train(training_set[:, :-1], training_set[:, -1])


    @overrides
    def optimize_policy(self, itr, samples_data):
        self.add_danger(samples_data)
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
