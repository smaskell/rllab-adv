import time

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
import matplotlib.pyplot as plt
import numpy as np
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv
import pickle
import argparse
import os
import gym
import random
#from IPython import embed

## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True, help='Name of adversarial environment')
parser.add_argument('--path_length', type=int, default=1000, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[100,100,100], help='layer definition')
parser.add_argument('--if_render', type=int, default=0, help='Should we render?')
parser.add_argument('--after_render', type=int, default=100, help='After how many to animate')
parser.add_argument('--n_exps', type=int, default=1, help='Number of training instances to run')
parser.add_argument('--n_itr', type=int, default=25, help='Number of iterations of the alternating optimization')
parser.add_argument('--n_pro_itr', type=int, default=1, help='Number of iterations for the portagonist')
parser.add_argument('--n_adv_itr', type=int, default=1, help='Number of interations for the adversary')
parser.add_argument('--batch_size', type=int, default=4000, help='Number of training samples for each iteration')
parser.add_argument('--save_every', type=int, default=100, help='Save checkpoint every save_every iterations')
parser.add_argument('--n_process', type=int, default=1, help='Number of parallel threads for sampling environment')
parser.add_argument('--adv_fraction', type=float, default=0.25, help='fraction of maximum adversarial force to be applied')
parser.add_argument('--step_size', type=float, default=0.01, help='kl step size for TRPO')
parser.add_argument('--gae_lambda', type=float, default=0.97, help='gae_lambda for learner')
parser.add_argument('--folder', type=str, default=os.environ['HOME'], help='folder to save result in')
parser.add_argument('--danger', action="store_true")


def count_catastrophies(observations):
    count = 0
    for episode in observations:
        if len(episode) < 200:
            count += 1
    return count

## Parsing Arguments ##
args = parser.parse_args()
env_name = args.env
path_length = args.path_length
layer_size = tuple(args.layer_size)
ifRender = bool(args.if_render)
afterRender = args.after_render
n_exps = args.n_exps
n_itr = args.n_itr
n_pro_itr = args.n_pro_itr
n_adv_itr = args.n_adv_itr
batch_size = args.batch_size
save_every = args.save_every
n_process = args.n_process
adv_fraction = args.adv_fraction
step_size = args.step_size
gae_lambda = args.gae_lambda
save_dir = args.folder

timestamp = int(time.time())

global catastrophies
global total_actions
global num_catastrophies
catastrophies = [(0, 0)]
total_actions = 0
num_catastrophies = 0
def add_catastrophies(observations):
    global catastrophies
    global total_actions
    global num_catastrophies
    for episode in observations:
        total_actions += len(episode)
        if len(episode) < 200:
            num_catastrophies += 1
            catastrophies.append((total_actions, num_catastrophies))


## Looping over experiments to carry out ##
for ne in range(n_exps):
    ## Environment definition ##
    ## The second argument in GymEnv defines the relative magnitude of adversary. For testing we set this to 1.0.
    env = normalize(GymEnv(env_name, adv_fraction))
    env_orig = normalize(GymEnv(env_name, 1.0))

    ## Protagonist policy definition ##
    pro_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=True
    )
    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Zero Adversary for the protagonist training ##
    zero_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val = 0.0
    )

    adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Initializing the parallel sampler ##
    parallel_sampler.initialize(n_process)

    ## Optimizer for the Protagonist ##
    pro_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=zero_adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_pro_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=True,
        use_danger=args.danger
    )

    ## Beginning alternating optimization ##
    for ni in range(n_itr):
        logger.log('\n\n\n####expNO{} global itr# {} n_pro_itr# {}####\n\n\n'.format(ne,ni,args.n_pro_itr))
        ## Train protagonist
        pro_algo.train()
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
        logger.log('{} catastrophies in {} episodes'.format(count_catastrophies(pro_algo.observations), len(pro_algo.observations)))
        add_catastrophies(pro_algo.observations)

    ## Shutting down the optimizer ##
    pro_algo.shutdown_worker()

with open('catastrophies-baseline-{}.csv'.format(timestamp), 'w') as file:
    for ep, cat in catastrophies:
        file.write("{}, {}\n".format(ep, cat))



logger.log('\n\n\n#### DONE ####\n\n\n')
