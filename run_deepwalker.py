 # Our stuff
from deepwalker.envs.deepwalker_env import DeepWalkerEnv
#ENV = DeepWalkerEnv
ENV_STR = 'DeepWalkerEnv'

import dwanim_convert
import common

# To define our RL environment
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register # To give our env an id so we can vectorize it

# To use a training algorithm
from stable_baselines3 import PPO
from stable_baselines3 import A2C
MODEL = A2C
MODEL_STR = 'A2C'
# To train
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback

import argparse
import os
import math
import numpy as np

# For training visualization
from tabulate import tabulate
import winsound
#if LAVALAMP_PRINT:
#   from ....lavalamp import lavalamp # 'attempted relative import with no known parent package'... i'll just copy paste for now

import matplotlib
REALTIME_PLOT = False
if REALTIME_PLOT:
    matplotlib.use('GTK3Agg') # This is what makes the window work while other stuff is happening
from matplotlib import pyplot as plt

# Interaction utils
from msvcrt import kbhit

import os
import sys


SAVE_DIR = r'.'
MP4_NAME = 'video_deepwalker.mp4'


def train(save_path, n_iters, eval=True):
    
    '''# Model
    if MODEL == PPO:
        model = PPO(policy='MlpPolicy', # args copied from https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit1/unit1.ipynb#scrollTo=2MXc15qFE0M9
                    env=env,
                    n_steps=1024,
                    batch_size=64,
                    n_epochs=4,
                    gamma=.999,
                    gae_lambda=.98,
                    ent_coef=.01,
                    verbose=False)

    elif MODEL == A2C:
        model = A2C(policy = 'MlpPolicy',
                    env = env,
                    gae_lambda = 0.9,
                    gamma = 0.99,
                    learning_rate = 0.00096,
                    max_grad_norm = 0.5,
                    n_steps = 8,
                    vf_coef = 0.4,
                    ent_coef = 0.0,
                    tensorboard_log = './tensorboard',
                    policy_kwargs=dict(
                    log_std_init=-2, ortho_init=False),
                    normalize_advantage=False,
                    use_rms_prop= True,
                    use_sde= True,
                    verbose=False)'''
    
    #TODO options
    env = DeepWalkerEnv(show=False)
    env.reset()
    
    #TODO train
    print("training not supported yet")

    return env

    

def demo(load_path, anim_to_use, mp4=False):
    try:
        model = MODEL.load(load_path)
    except FileNotFoundError:
        return
    
    #TODO options
    env = DeepWalkerEnv(anim_to_use=anim_to_use, 
                        show=True,
                        mp4_path = os.path.join(SAVE_DIR, MP4_NAME) if mp4 else None)
    env.reset()
    env.print_deepwalker_info()

    
    print('Running the model. Press any key to exit')

    exit = False
    try:
        while not kbhit():
            #TODO Infer action
            action = None
            _, _, done, exit = env.step(action)

            if done:
                print("reset")
                env.reset()
            elif exit:
                print("exited")
                break
            else:
                # Display pose
                #env.print_deepwalker_pose()
                env.display_ref_pose_text()
                env.display_curr_pose_text()
    except Exception as e: # If user exits the window, we get this error
        if "Not connected to physics server." not in str(e):
            raise e
        else:
            print("Window closed. Session ended.")

    return env


def direct_anim(anim_to_use, anim_ctrls=False, mp4=False):
    """Just plays the animation on the humanoid
    """
    env = DeepWalkerEnv(direct_anim=True, 
                        anim_to_use=anim_to_use, 
                        anim_ctrls=anim_ctrls, 
                        show = True, 
                        store_rw_history = False,
                        mp4_path = os.path.join(SAVE_DIR, MP4_NAME) if mp4 else None)
    env.reset()
    env.print_deepwalker_info()
    
    print('Running the animation(s). Press any key to exit')

    exit = False
    try:
        while not kbhit():
            _, _, done, exit = env.step(None)

            if done:
                print("done with anim")
                env.reset()
            elif exit:
                print("exited")
                break
            else:
                # Display pose
                #env.print_deepwalker_pose()
                env.display_ref_pose_text()
                #env.display_curr_pose_text()
    except Exception as e: # If user exits the window, we get this error
        if "Not connected to physics server." not in str(e):
            raise e
        else:
            print("Window closed. Session ended.")


def env_only():
    """Just runs the environment with no model
    """
    env = DeepWalkerEnv(no_anim=True, no_action=True, show=True, store_rw_history=False)
    env.reset()
    env.print_deepwalker_info()
    
    print('Running the environment. Press any key to exit')

    exit = False
    try:
        while not kbhit() and not exit:
            _, _, _, exit = env.step(None)
    except Exception as e: # If user exits the window, we get this error
        if "Not connected to physics server." not in str(e):
            raise e
        else:
            print("Window closed. Session ended.")




def plot(reward_breakdown_history, block=True, **kwargs):

    # Skip every other one if there's a lot
    n_episodes = len(reward_breakdown_history)
    if n_episodes <= 100:
        use_interval = 1
    else:
        use_interval = int( n_episodes / 100 ) # round

    # Determine how many rows and columns
    n_plots = len(reward_breakdown_history)
    if n_plots > 3:
        n_cols = math.ceil( math.sqrt( n_plots ) )
    else:
        n_cols = n_plots
    n_rows= math.ceil( n_plots / n_cols )
    #print(f"n_plots = {n_plots}, n_rows = {n_rows},   n_cols = {n_cols}")

    # Open up plot window
    # Modeled after https://stackoverflow.com/a/15724978
    figures, axes = plt.subplots(n_rows, n_cols)

    # TODO...
    '''
    #axis.set_aspect("equal")
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            i = (row_i) * n_cols + col_i
            i *= use_interval
            if i >= n_plots:
                break
            #print(f"row_i = {row_i},    col_i = {col_i},    i = {i}")

            
            if n_plots == 1:
                axis = axes
            elif n_rows > 1 and n_cols > 1:
                axis = axes[row_i, col_i]
            # subplots() returns a 1d array if there aren't multiple rows, and we can't subscript with two params if it's a 1d array
            else:
                axis = axes[i]
            
            if len( self.histories_t[i] ) > 0:
                axis.set_xlim( 0, self.histories_t[i][-1] )
            else:
                print(f"Weird: No time history for episode {i + 1}")
                axis.set_xlim( 0, 1 )
            axis.set_ylim( -.1, .2 )

            # Top right corner text
            axis.text(0, 1, f"Ep. {i + 1}", horizontalalignment='left', verticalalignment='top', transform=axis.transAxes, fontsize=18)

            self.plot_points = axis.plot(   self.histories_t[i], self.histories_reward_obj_z[i], "b-",    # Object Z Position Reward ~ Blue
                                            self.histories_t[i], self.histories_reward_clutches[i], "g-", # Distance to Clutches Reward ~ Green
                                            self.histories_t[i], self.histories_reward_grasp[i], "m-",    # Grasp Reward ~ Magenta
                                            self.histories_t[i], self.histories_reward_on_target[i], "r-" # On-Target Reward ~ Red
                                            )
    '''

    plt.show(block=block, **kwargs)

    '''if not block:
        while not kbhit():
            time.sleep(.25)'''

    return

    

if __name__ == '__main__':

    # Parse arguments, then run one of the above functions

    parser = argparse.ArgumentParser(
        description='Trains a new ML model in a deepwalker pybullet environment',
        epilog=''
    )
    
    parser.add_argument('--version', '-v', type=str, default='0', help='(str) The version to save/load the model. Default: 0')
    parser.add_argument('--plot', '-p', action='store_true', help='Whether to plot the rewards after training/demoing')
    parser.add_argument('--mp4', '-m', action='store_true', help='Whether to save an mp4')

    parser.add_argument_group('Training')
    parser.add_argument('--train', '-t', action='store_true', help='Train and save a new model')
    parser.add_argument('--iterations', '-i', type=int, default=50_000, help='Number of timesteps to train. Default: 50k')
    parser.add_argument('--noeval', '-ne', action='store_true', help='Whether to skip evaluation at the end of training (~20 sec)')

    parser.add_argument_group('Demo')
    parser.add_argument('--demo', '-d', action='store_true', help='Show a saved model in action, in a pybullet window')
    parser.add_argument('--anim', '-a', type=str, default=None, help='The anim xml filepath to use as reference during every reset of the demo. '
                                                                    + 'Also works for --directanim. If not used, random animations will be chosen from the DeepWalkerAnims folder')
    parser.add_argument('--basepose', '-bp', action='store_true', help='Use the basepose animation')
    parser.add_argument('--test', '-tc', action='store_true', help='Use the test_ctrls animation')

    parser.add_argument_group('Directly Animated')
    parser.add_argument('--directanim', '-da', action='store_true', help='Don\'t use a model, directly play the reference animation in the pybullet environment')
    parser.add_argument('--animctrl', '-ac', action='store_true', help='Still use pybullet physics by making the animation be the target poses for the robot\'s motors')


    parser.add_argument_group('Env Only')
    parser.add_argument('--envonly', '-eo', action='store_true', help='Don\'t use a model or a reference animation, just show pybullet environment')
    
    args = parser.parse_args()

    model_name = f'{ ENV_STR.lower() }-{ MODEL_STR.lower() }-{ args.version }'
    model_path = os.path.join(SAVE_DIR, model_name)

    anim = None
    if args.anim:
        anim = args.anim
    elif args.basepose:
        anim = common.BASEPOSE_ANIM_PATH
    elif args.test:
        anim = common.TEST_ANIM_PATH

    print(f"Using model '{model_name}' ({model_path})")

    if args.train:
        # Train then plot then demo
        # Train
        env = train(model_path, args.iterations, eval=not args.noeval)

        # Plot
        if args.plot:
            plot(env.reward_breakdown_history, block=False)
        
        # Demo
        demo_env = demo(model_path, mp4 = args.mp4)
        if not demo_env:
            print(f'\nModel not found: {model_path}\nRun this file with -train to train and save a model. ')
        else:
            demo_env.close()

        env.close()

    elif args.demo:
        # Demo then plot
        # Demo
        env = demo(model_path, anim, mp4 = args.mp4)
        if not env:
            print(f'\nModel not found: {model_path}\nRun this file with -train to train and save a model. ')
        else:
            # Plot
            if args.plot:
                plot(env.reward_breakdown_history, block=True)
                
            env.close()
    
    elif args.directanim or args.animctrl:
        direct_anim(anim, anim_ctrls = args.animctrl, mp4 = args.mp4)
        # if anim_to_use is None, dwanim_reader will choose a random animation from the folder, which is currently just the ref animation
    
    elif args.envonly:
        env_only()

    else:
        print('Must either train or demo')
        parser.print_help()
    
    