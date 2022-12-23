#https://www.etedal.net/2020/04/pybullet-panda_2.html

REALTIME_PLOT = False # Currently realtime plotting doesn't work, if you set this to true you'll get errors
LAVALAMP_PRINT = False # If false, uses progress bar

DEFAULT_SAVENAME = 'tray-panda-test'
SAVE_PATH = r'C:\Users\robert.krawczyk\Documents\NNI\pybullet-tests\tray-panda'


# To define our RL environment
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register # To give our env an id so we can vectorize it

# To run our physics simulation
import pybullet as p
import pybullet_data

# To use a training algorithm
from stable_baselines3 import PPO
from stable_baselines3 import A2C
MODEL = A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
if not LAVALAMP_PRINT:
    from stable_baselines3.common.callbacks import ProgressBarCallback

import os
import sys
import math
import numpy as np
import random

# For training visualization
from tabulate import tabulate
import winsound
#if LAVALAMP_PRINT:
#   from ....lavalamp import lavalamp # "attempted relative import with no known parent package"... i'll just copy paste for now

import matplotlib
if REALTIME_PLOT:
    matplotlib.use('GTK3Agg') # This is what makes the window work while other stuff is happening
from matplotlib import pyplot as plt

# Interaction utils
from msvcrt import kbhit
import time
from screeninfo import get_monitors
from datetime import datetime


# Making our own environment
class TrayPandaEnv(gym.Env):
    '''A gym learning environment that runs a pybullet physics simulation in the process of learning'''

    def __init__(self, name=DEFAULT_SAVENAME, pbwindow=True, mp4=False, verbose=True, plot=False, debuglines=True):
        # Init pybullet
        pybullet_kwargs = {}

        # Window and mp4 options
        screen = get_monitors()[0]
        if pbwindow:
            pybullet_kwargs["options"] = f"--width={screen.width-5} --height={screen.height-40}"

            if mp4:
                mp4_filename = f"{name}.{datetime.now().ctime()}.mp4".replace(" ", "_")
                pybullet_kwargs["options"] += f" --mp4=\"{mp4_filename}\" --mp4fps=240"
                pass

        p.connect(p.GUI if pbwindow else p.DIRECT, **pybullet_kwargs)


        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.45,0.35,-0.2])
        
        self.action_space = spaces.Box(np.array([-3, -3, -3, -1]), np.array([3, 3, 3, 1]), dtype=np.float32) # 4-length action, each bounded between -1 and 1
        self.observation_space = spaces.Box(np.array([-1]*12), np.array([1]*12), dtype=np.float32) # 12-length observations, each bounded between -1 and 1
    

        if plot:
            if REALTIME_PLOT:
                # Open up plot window
                # Modeled after https://stackoverflow.com/a/15724978
                self.figure, self.axis = plt.subplots()
                self.axis.set_aspect("equal")
                self.axis.set_xlim(0, 10)
                self.axis.set_ylim(-5, 10)
                #self.axis.hold(True)
                
                # ?
                plt.show(block=False) # False = don't wait for the window to close before doing other stuff
                plt.draw()

                # Save background for blit
                self.plot_background = self.figure.canvas.copy_from_bbox(self.axis.bbox)

                # Plot initial
                self.plot_points = self.axis.plot(  [0], [0], "b-", # Object Z Position Reward:       Blue Line
                                                    [0], [0], "g-", # Distance to Clutches Reward:    Green Line
                                                    [0], [0], "m-", # Grasp Reward:                   Magenta Line
                                                    [0], [0], "r-"  # On-Target Reward ~ Red
                                                    ) 



        # Set vars
        # Always this orientation
        self.fixed_effector_orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.])

        # Cached vals for rewards functions
        self.prev_dist_obj_clutches = 0
        self.prev_dist_obj_target = 0
        self.prev_grasp = 0

        # History for graphing
        self.histories_t = []
        self.histories_reward_obj_z = []
        self.histories_reward_clutches = []
        self.histories_reward_grasp = []
        self.histories_reward_on_target = []

        # Reset counts
        self.reset_count = -1
        self.step_count = 0
        self.print_count = 0

        # Printing
        self.name = name
        self.mp4 = mp4
        self.verbose = verbose
        self.plotting = plot
        self.debuglines = debuglines

        if self.verbose and LAVALAMP_PRINT:
            print("[\n[\n[\n[ Starting up...")


    def reset(self):
        '''Loads the stuff into the scene, resets poses, sets gravity'''
        if self.mp4:
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False) # Disable rendering until we load everything
        p.setGravity(0, 0, -10)

        # Load Stuff
        
        urdfs_path = pybullet_data.getDataPath()
        # Import plane
        plane_uid = p.loadURDF(os.path.join( urdfs_path, 'plane.urdf' ), basePosition=[0,0,-0.65])

        # Import panda (robot arm)
        panda_rest_pose = (0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08) # Rest position of each joint motor on the panda
        self.panda_uid = p.loadURDF(os.path.join(urdfs_path, 'franka_panda/panda.urdf'), useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.panda_uid, i, panda_rest_pose[i])
        
        table_uid = p.loadURDF(os.path.join(urdfs_path, 'table/table.urdf'), basePosition=[0.5,0,-0.65])

        tray_uid = p.loadURDF(os.path.join(urdfs_path, 'tray/tray.urdf'), basePosition=[0.65,0,0])

        obj_startpos = (random.uniform(0.5, 0.8), random.uniform(-0.2, 0.2), 0.05)
        self.obj_uid = p.loadURDF(os.path.join(urdfs_path, 'random_urdfs/000/000.urdf'), basePosition=obj_startpos)

        # Read Scene

        # Now that we've reset the arm, get the position of the 'hand' which will become the effector's position
        post_reset_effector_pos = p.getLinkState(self.panda_uid, 11)[0]
        # And the fingers' state
        post_reset_fingers_state = ( p.getJointState(self.panda_uid, 9)[0], p.getJointState(self.panda_uid, 10)[0] )
        # And the object's state
        obj_pos_rot = p.getBasePositionAndOrientation(self.obj_uid)
        obj_pos_rot = obj_pos_rot[0] + obj_pos_rot[1]


        # This is the format we are going for when we assemble observations: effector position x | y | z | fingers state a | b
        observation = post_reset_effector_pos + post_reset_fingers_state + obj_pos_rot
        if self.verbose:
            #print(f"\nReset| Effector Position = {post_reset_effector_pos}, Fingers State = {post_reset_fingers_state}, Object State = {obj_pos_rot}")
            pass

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True) # Reenable rendering now that everything is loaded

        if self.mp4:
            # Start recording mp4
            mp4_filename = f"{self.name}.{datetime.now().ctime()}.mp4".replace(" ", "_")
            mp4_path = os.path.join(SAVE_PATH, mp4_filename)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, mp4_path)

        self.reset_count += 1
        self.step_count = 0
        self.print_count = 0

        # Add another episode to the histories
        self.histories_t.append( [] )
        self.histories_reward_obj_z.append( [] )
        self.histories_reward_clutches.append( [] )
        self.histories_reward_grasp.append( [] )
        self.histories_reward_on_target.append( [] )

        return observation
    

    def step(self, action):

        # Render at real time (rather than instant steps)
        p.configureDebugVisualizer( p.COV_ENABLE_SINGLE_STEP_RENDERING )

        # Read state
        curr_effector_pos = p.getLinkState(self.panda_uid, 11)[0]
        curr_obj_pos = p.getBasePositionAndOrientation(self.obj_uid)[0]

        # Decide new effector position
        effector_interp_rate = 0.009 # Speed at which the effector interps to its target
        effector_dx = action[0] * effector_interp_rate
        effector_dy = action[1] * effector_interp_rate
        effector_dz = action[2] * effector_interp_rate
        fingers_state = action[3]

        # Decide new pose
        new_effector_pos = [curr_effector_pos[0] + effector_dx,
                            curr_effector_pos[1] + effector_dy,
                            curr_effector_pos[2] + effector_dz]
        new_joint_states = p.calculateInverseKinematics(self.panda_uid, 11, new_effector_pos, self.fixed_effector_orientation)
        new_joint_states = new_joint_states[:8] # ignore 8th joint
        #print(f"Step| Joint States From IK = {new_joint_states}, Fingers State = {fingers_state}")

        p.setJointMotorControlArray( self.panda_uid,
                                     [0,1,2,3,4,5,6,7,9,10], # ignore 8th joint 
                                     p.POSITION_CONTROL, 
                                     list(new_joint_states) + 2*[fingers_state] )
        
        p.stepSimulation()

        # Read the state after the step
        next_effector_pos = p.getLinkState(self.panda_uid, 11)[0]
        next_fingers_state = ( p.getJointState(self.panda_uid, 9)[0], p.getJointState(self.panda_uid, 10)[0] )
        next_obj_pos_rot = p.getBasePositionAndOrientation(self.obj_uid) # In a real robotics setting you don't know this, but in a game we do
        next_obj_pos_rot = next_obj_pos_rot[0] + next_obj_pos_rot[1]
        observation = next_effector_pos + next_fingers_state + next_obj_pos_rot

        #print(f"\nStep| Effector Position = {next_effector_pos},    Fingers State = {next_fingers_state},    Object State = {next_obj_pos_rot}")
        #print(f"Step| Observation = ({type(observation)}) {observation}\n")

        # Determine rewards
        if self.step_count > 10 * 240: # 10 sec
            reward = 0
            done = True
        elif next_obj_pos_rot[2] > 0.45:
            reward = 100
            done = True
        else:
            next_obj_pos_arr = np.array( next_obj_pos_rot[:3] )

            #---Reward--------------------------------------------------------------------------------------------
            #TODO: Add energy use penalty / direction change penalty
            #TODO: Make a keyframe animation and try to imitate it
            #TODO: Get realtime plotting working

            """
            # Difference in distance between effector and object
            prev_dist_hand_obj = np.linalg.norm( np.array( curr_effector_pos ) - next_obj_pos_arr )
            dist_hand_obj = np.linalg.norm( np.array( next_effector_pos ) - next_obj_pos_arr )
            d_dist = prev_dist_hand_obj - dist_hand_obj
            """

            # Object height (delta)
            d_obj_z = next_obj_pos_rot[2] - curr_obj_pos[2]

            # Distance to clutches (delta) and Distance to target (delta)
            finger1_pos = p.getLinkState(self.panda_uid, 9)[0]
            finger2_pos = p.getLinkState(self.panda_uid, 10)[0]
            # avg finger positions + offset to fingertips
            clutches_pos = [ (finger1_pos[i] + finger2_pos[i]) / 2 for i in (0,1,2) ] + np.array((0,0,-.03)) # Slight offset so the it's at the fingertips not the middle
            # .15 units below clutches is where we want the obj to be
            target_pos = clutches_pos + np.array((0,0,-.15))

            # distance to clutches
            dist_obj_clutches = np.linalg.norm( np.array( clutches_pos ) - next_obj_pos_arr )
            d_dist_clutches = self.prev_dist_obj_clutches - dist_obj_clutches
            self.prev_dist_obj_clutches = dist_obj_clutches # update var

            # distance to target
            dist_obj_target = np.linalg.norm( np.array( curr_obj_pos ) - target_pos )
            d_dist_target = self.prev_dist_obj_target - dist_obj_target
            self.prev_dist_obj_target = dist_obj_target # update var
            

            # if wer're not close to target, try to get close to target
            if dist_obj_target > .08:
                # Targeting State
                on_target_coef = 30
                clutches_coef = 0
                grasp_coef = -0.00001
                obj_z_coef = 0
            else:
                on_target_coef = 0
                # if we're close to target, switch to wanting to be close to clutches
                if dist_obj_clutches > .0001:
                    # Closing In State
                    clutches_coef = 30
                    grasp_coef = -0.00001
                    obj_z_coef = 0
                # if we're close to clutches, switch to wanting to grasp and get the object high
                else:
                    # Grabbing and Holding State
                    clutches_coef = 50
                    grasp_coef = 50
                    obj_z_coef = 20


            # Grasp amount (delta)
            d_grasp = self.prev_grasp - action[3]
            self.prev_grasp = action[3] # update var

            
            # Just changed it to reward based on distance rather than delta distance

            rw_obj_z = obj_z_coef * d_obj_z
            #rw_obj_z = obj_z_coef * curr_obj_pos[2]

            rw_clutches = clutches_coef * d_dist_clutches
            #rw_clutches = clutches_coef * dist_obj_clutches

            rw_grasp = grasp_coef * d_grasp

            rw_on_target = on_target_coef * d_dist_target
            #rw_on_target = on_target_coef * dist_obj_target

            # Reward Equation
            reward = rw_obj_z + rw_clutches + rw_grasp + rw_on_target
            #----------------------------------------------------------------------------------------------------

            # Update history to be plotted
            self.histories_t                [-1] += [self.step_count / 240]
            self.histories_reward_obj_z     [-1] += [rw_obj_z]
            self.histories_reward_clutches  [-1] += [rw_clutches]
            self.histories_reward_grasp     [-1] += [rw_grasp]
            self.histories_reward_on_target [-1] += [rw_on_target]


            if self.debuglines:
                # Update debug lines in pybullet viewport
                if on_target_coef > 0:
                    # Target line
                    p.addUserDebugLine(curr_obj_pos, target_pos, lineColorRGB=[1, 0, 0], lineWidth=2.0, lifeTime=1/240)
                
                if clutches_coef > 0:
                    # Clutches line
                    p.addUserDebugLine(curr_obj_pos, clutches_pos, lineColorRGB=[0, 1, 0], lineWidth=2.0, lifeTime=1/240)


            # Update lavalamp table and real-time graph every 1/4 second
            if self.step_count % 60 == 0:
                self.print_count += 1

                # Print
                if self.verbose and LAVALAMP_PRINT:
                    lampy = lavalamp(self.print_count)

                    # Display variables in a single-row table
                    table_rows = [[ reward, d_obj_z, clutches_coef, d_dist_clutches, grasp_coef, d_grasp ]]
                    table_rows = [[ val.__round__(4) for val in table_rows[0] ]]
                    table = tabulate( table_rows, headers=('Reward', 'Obj Height Diff', 'Clutches Coef', 'Clutches Distance Diff', 'Grasp Coef', 'Grasp Diff') )

                
                    print(f"\x1B[4A{lavalamp(self.print_count)} Reset {self.reset_count}, Step {self.step_count}\n{table}")
                    #print_w_lavalamp( f"Reset {self.reset_count}, Step {self.step_count}\n{table}\r\r\r", self.print_count )"""

                if self.plotting and REALTIME_PLOT:

                    if REALTIME_PLOT:
                        # Update plots
                        #             line idx          x data points       y data points
                        self.plot_points[0].set_data( self.histories_t[-1], self.histories_reward_obj_z[-1] )
                        self.plot_points[1].set_data( self.histories_t[-1], self.histories_reward_clutches[-1] )
                        self.plot_points[2].set_data( self.histories_t[-1], self.histories_reward_grasp[-1] )

                        # Update screen
                        plt.show(block=False)
                        #self.figure.canvas.draw()

            done = False

        
        self.step_count += 1

        return observation, reward, done, {}#"reward data": (rw_obj_z, rw_clutches, rw_grasp)} # returned reward data currently not used


    # The tutorial was for robotics so they made a render function to emulate a camera next to the robot,
    #   telling the robot where the 'obj' is. We don't need anything like that because we're in a game,
    #   we can actually know the ground truth positions of things.
    '''
    def render(self, mode='human'):
        pass
    '''

    def plot(self, block=True, **kwargs):

        # Skip every other one if there's a lot
        n_episodes = len(self.histories_t)
        if n_episodes <= 100:
            use_interval = 1
        else:
            use_interval = int( n_episodes / 100 ) # round

        # Determine how many rows and columns
        n_plots = len(self.histories_t)
        if n_plots > 3:
            n_cols = math.ceil( math.sqrt( n_plots ) )
        else:
            n_cols = n_plots
        n_rows= math.ceil( n_plots / n_cols )
        #print(f"n_plots = {n_plots}, n_rows = {n_rows},   n_cols = {n_cols}")

        # Open up plot window
        # Modeled after https://stackoverflow.com/a/15724978
        figures, axes = plt.subplots(n_rows, n_cols)

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

        plt.show(block=block, **kwargs)

        if not block:
            while not kbhit():
                time.sleep(.25)

        return


    def close(self):
        if self.plotting:
            plt.close()

        if self.mp4:
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
        p.disconnect()



def main(savename=DEFAULT_SAVENAME, train_iters=20_000, evaluate=True, mp4=False, plot=True):
    # Environment
    env = TrayPandaEnv(name=savename, pbwindow=False, mp4=mp4, verbose=True, plot=plot)
    #env = make_vec_env(env, n_envs=16) # Vectorize environment to run 16 independently at same time
    observation = env.reset()

    # Model
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
        model = A2C(policy = "MlpPolicy",
                    env = env,
                    gae_lambda = 0.9,
                    gamma = 0.99,
                    learning_rate = 0.00096,
                    max_grad_norm = 0.5,
                    n_steps = 8,
                    vf_coef = 0.4,
                    ent_coef = 0.0,
                    tensorboard_log = "./tensorboard",
                    policy_kwargs=dict(
                    log_std_init=-2, ortho_init=False),
                    normalize_advantage=False,
                    use_rms_prop= True,
                    use_sde= True,
                    verbose=False)

    

    print(f"\n\nTraining {savename} in TrayPandaEnv...")

    # Train! The model knows how to train with the environment format we gave it
    model.learn(train_iters, progress_bar=not LAVALAMP_PRINT)

    # Save the trained model (somewhere idk)
    model_path = os.path.join(SAVE_PATH, savename)
    model.save(model_path)
    env.close()
    print(f"\nSaved trained model '{savename}'")

    winsound.MessageBeep()

    if evaluate:
        print("Evaluating...")
        # Evaluate the trained model
        # Make a new environment just to test
        eval_env = TrayPandaEnv(pbwindow=False, verbose=False)
        eval_env = Monitor(eval_env)
        eval_env.reset()
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=2, deterministic=True)
        eval_env.close()
        print(f"Mean reward = {mean_reward:.2f} +/- {std_reward}")


    print("Showing rewards graphs from training...")
    if plot:
        env.plot(block=False)
    
    # Run the model in a window
    show_model(savename=savename, plot=False) # Will loop until user presses a key or X's it out


    try:
        env.close()
    except p.error as e:
        pass
    
    print("Closed")



def show_model(savename=DEFAULT_SAVENAME, mp4=False, plot=True):
    model = load(savename)
    if not model:
        return

    # Run forever in 20 second episodes
    print("Running model...\nPress Enter to exit simulation and plot rewards  ", end="")

    # New environment, open up pybullet window
    env = TrayPandaEnv(name=savename, pbwindow=True, mp4=mp4, verbose=False, plot=True)
    observation = env.reset()

    while True:
        if kbhit(): # if user hits a key (kbhit = keyboard hit)
            break

        action, _ = model.predict(observation)

        # Render at real time (rather than instant steps)
        p.configureDebugVisualizer( p.COV_ENABLE_SINGLE_STEP_RENDERING )

        # Step
        try:
            observation, reward, done, info = env.step(action)
        except p.error as e:
            break
        
        if done:
            #print("Done with this episode!")
            print(".", end="")
            env.reset()
    
    print("Showing rewards graph. X the window out to close")
    if plot:
        env.plot()
        print("Closed")



def justplot(n_episodes=6, savename=DEFAULT_SAVENAME):
    model = load(savename)
    if not model:
        return
    
    env = TrayPandaEnv(name=savename, pbwindow=False, verbose=False, plot=True)

    print(f"Running {n_episodes} new episodes to plot  ", end="")

    for ep in range( n_episodes ):
        print(".", end="")

        done = False
        observation = env.reset()
        while not done:
            # Step
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            print("_", end="")
    
    print(f"\nPlotting {n_episodes} rewards graphs.")
    env.plot()



def load(savename=DEFAULT_SAVENAME):
    model_path = os.path.join(SAVE_PATH, savename)
    try:
        model = MODEL.load(model_path)
    except FileNotFoundError as e:
        print(f"\nError: No model with name '{savename}' found in {SAVE_PATH}'\nRun this file without 'show' to train and save a model. ")
        return
    return model
    



def lavalamp(n, leftside="<(", rightside=")>", bg="-", ball="o"):
    lavalamp_frames = ("o-------", "-o------", "--o-----", "---o----", "----o---", "-----o--", "------o-", "-------o")
    if n % 14 < 8:
        frame_i = n % 14
    else:
        frame_i = -(n % 7) - 1
    
    inside = lavalamp_frames[frame_i].replace("-", bg).replace("o", ball)
    return f"{leftside}{inside}{rightside}"





if __name__ == '__main__':

    # Parse Args

    if "-version" in sys.argv:
        name_token_i = sys.argv.index("-version")
        savename = DEFAULT_SAVENAME + "-" + sys.argv[ name_token_i + 1 ]
    else:
        savename = None


    if "-train" in sys.argv:
        # Train (default)
        kwargs = {}
        if savename:
            kwargs["savename"] = savename

        train_token_i = sys.argv.index("-train")
        kwargs["train_iters"] = int( sys.argv[ train_token_i + 1 ] )

        kwargs["evaluate"] = "-noeval" not in sys.argv
        kwargs["plot"] = "-noplot" not in sys.argv
        kwargs["mp4"] = "-mp4" in sys.argv
            
        print(f"Calling main() with kwargs {kwargs}")
        main(**kwargs)

    elif "-show" in sys.argv:
        kwargs = {}
        if savename:
            kwargs["savename"] = savename
        
        kwargs["mp4"] = "-mp4" in sys.argv

        print(f"Calling show_model() with kwargs {kwargs}")
        show_model(**kwargs)

    elif "-justplot" in sys.argv:
        kwargs = {}
        if savename:
            kwargs["savename"] = savename

        justplot_i = sys.argv.index("-justplot")
        try:
            kwargs["n_episodes"] = int( sys.argv[ justplot_i + 1 ] )
        except IndexError as e:
            pass
        
        print(f"Calling justplot() with kwargs {kwargs}")
        justplot(**kwargs)
    
    else:
        print(f"""
tray_panda.py Usage


-train [n steps]    Train a new model, then show it in a pybullet window
    -noeval         Don't calculate the mean standard error after training (takes 20 sec or so)
    -noplot         Don't plot the rewards after training

-show               Just show a saved model in a pybullet window, and plot the rewards afterwards

-justplot [n eps]   Just plot the rewards of [n eps] newly run episodes (default: 6)

General Options:
-version [version]  You define the version to include in the saved model name (default: no version)
-mp4                Save the episode(s) as mp4(s) in the environment root folder (doesn't do anything with -justplot)

Example: tray_panda.py -train 50000 -version 0
Later: tray_panda.py -show""")
    