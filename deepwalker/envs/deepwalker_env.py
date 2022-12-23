
# We are making a gym (aka OpenAI) environment
import gym
from gym import error, spaces, utils
import numpy as np

# We are making a pybullet physics simulation
import pybullet as p
import pybullet_data

# We are making the model imitate animation clips
r'''from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
TEST_FBX_PATH = r'C:\Users\rmore\Projects\ASE\ase\poselib\data\01_01_cmu.fbx'
from poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive'''
import pyfbx.pyfbx as fbx
import pyfbx.pyfbx.animation

# Utils
from screeninfo import get_monitors
from os.path import join
from tabulate import tabulate




class DeepWalkerEnv(gym.Env):
    '''A gym learning environment that creates a humanoid in pybullet and runs pybullet to determine rewards based on how well the humanoid is walking.'''

    def __init__(self, name='unnamed-deepwalkerenv', show=True, debuglines=True, verbose=True, store_rw_history=True, mp4_path=None):
        print('init deepwalker env')
        self.debuglines = debuglines
        self.name = name
        self.verbose = verbose
        self.storing_rw_history = store_rw_history
        self.mp4_path = mp4_path

        # Init pybullet
        pybullet_kwargs = {}

        # Kwargs for showing window and saving mp4
        if show:
            screen = get_monitors()[0]
            pybullet_kwargs['options'] = f'--width={screen.width-5} --height={screen.height-60}'

            if self.mp4_path:
                pybullet_kwargs['options'] += f' --mp4="{mp4_path}" --mp4fps=240'
        
        p.connect(p.GUI if show else p.DIRECT, **pybullet_kwargs)


        # Need to make these specifically for the humanoid
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.45,0.35,-0.2])
        
        #self.action_space = spaces.Box(np.array([-3, -3, -3, -1]), np.array([3, 3, 3, 1]), dtype=np.float32) # 4-length action, each bounded between -1 and 1
        #self.observation_space = spaces.Box(np.array([-1]*12), np.array([1]*12), dtype=np.float32) # 12-length observations, each bounded between -1 and 1

        self.episode_count = 0

        if self.storing_rw_history:
            self.reward_breakdown_history = []
            # Each element in this will be a python list, which will contain [n steps] numpy lists of size: [n reward terms]


    def reset(self):
        '''Loads the stuff into the scene, resets poses, sets gravity and stuff'''

        # Stop previous mp4 until reset is complete
        if self.mp4_path:
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

        # Reset pybullet world
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False) # Disable rendering until we load everything
        p.setGravity(0, 0, -10)

        # Import the 3D stuff
        urdfs_path = pybullet_data.getDataPath()
        plane_uid = p.loadURDF(join( urdfs_path, 'plane.urdf' ), basePosition=[0,0,-0.65])
        walker_uid = p.loadURDF(join( urdfs_path, r'humanoid\humanoid.urdf' ), basePosition=[0,0,1])

        # For debug, print joint info
        if self.verbose:
            jit_headers = ('i', 'Joint Name', 'Type', 'Min', 'Max', 'Link Name', 'Axis', 'Pos (local)', 'Orn (local)', 'Parent i' )
            jit_rows = [] # joint info table (jit)
            for i in range(p.getNumJoints(walker_uid)):
                jit_row = list(p.getJointInfo(walker_uid, i))

                jit_row = jit_row[:3] + jit_row[8:10] + jit_row[12:] # Cut out unnecessary info
                jit_row[2] = ('revolute', 'prismatic', 'spherical', 'planar', 'fixed')[jit_row[2]] # Type index -> type name
                #jit_row[6] = ('x', 'y', 'z')[ jit_row[6].index(1.0) ] # Axis vector -> axis name
                # Parent index -> parent name
                if jit_row[-1] == -1:
                    jit_row[-1] = 'none'
                else:
                    jit_row[-1] = str( jit_rows[ jit_row[-1] ][1] ) # get name from row at index [this row's last elem (aka parent i)]
                
                jit_rows.append( jit_row )
            print(tabulate(jit_rows, headers=jit_headers))
            

        # Import FBX
        #DEBUG (TODO: Get real clips)
        '''motion = SkeletonMotion.from_fbx(fbx_file_path=TEST_FBX_PATH, root_joint='Hips', fps=60)
        plot_skeleton_motion_interactive(motion)'''

        # Set base pose
        #TODO
        
        # Done resetting the pybullet world
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True) # Reenable rendering now that everything is loaded

        # Make initial observation
        #TODO
        observation = None


        # Store new data
        self.episode_count += 1
        self.step_count = 0

        if self.storing_rw_history:
            self.reward_breakdown_history.append( [] )

        if self.mp4_path:
            # Start saving a new mp4
            partition = self.mp4_path.rpartition('.mp4')
            reset_mp4_path = partition[0] + f'_ep{self.episode_count}' + partition[1]
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, reset_mp4_path)
        
        
        return observation

    

    def step(self):

        exit = False

        try:
            # Render at real time (rather than instant steps)
            p.configureDebugVisualizer( p.COV_ENABLE_SINGLE_STEP_RENDERING )
            # STEP
            p.stepSimulation()
        
        except pybullet.error as e: # If user exits the window, we get this error
            if "Not connected to physics server" in str(e):
                exit = True
            else:
                raise e

        observation = None
        reward = None
        done = None
        return observation, reward, done, exit

        

    

    def close(self):
        if self.mp4_path:
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

        p.disconnect()