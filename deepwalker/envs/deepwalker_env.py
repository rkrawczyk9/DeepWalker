#IN PROGRESS: FIGURING OUT WHICH CONTROL MEANS WHAT, how to retarget animation to the robot
"""
  i  Joint Name      Type         Min    Max  Link Name       Axis             Pos (local)                      Orn (local)           Parent i
---  --------------  ---------  -----  -----  --------------  ---------------  -------------------------------  --------------------  -----------------
  0  root            fixed       0     -1     root            (0.0, 0.0, 0.0)  (0.0, 0.0, 0.0)                  (0.0, 0.0, 0.0, 1.0)  none
  1  chest           spherical   0     -1     chest           (0.0, 0.0, 0.0)  (0.0, 0.664604, 0.0)             (0.0, 0.0, 0.0, 1.0)  b'root'
  2  neck            spherical   0     -1     neck            (0.0, 0.0, 0.0)  (0.0, 0.41557600000000006, 0.0)  (0.0, 0.0, 0.0, 1.0)  b'chest'
  3  right_shoulder  spherical   0     -1     right_shoulder  (0.0, 0.0, 0.0)  (-0.0962, 0.494, 0.73244)        (0.0, 0.0, 0.0, 1.0)  b'chest'
  4  right_elbow     revolute    0      3.14  right_elbow     (0.0, 0.0, 1.0)  (0.0, -0.5391519999999999, 0.0)  (0.0, 0.0, 0.0, 1.0)  b'right_shoulder'
  5  right_wrist     fixed       0     -1     right_wrist     (0.0, 0.0, 0.0)  (0.0, -0.555788, 0.0)            (0.0, 0.0, 0.0, 1.0)  b'right_elbow'
  6  left_shoulder   spherical   0     -1     left_shoulder   (0.0, 0.0, 0.0)  (-0.0962, 0.494, -0.73244)       (0.0, 0.0, 0.0, 1.0)  b'chest'
  7  left_elbow      revolute    0      3.14  left_elbow      (0.0, 0.0, 1.0)  (0.0, -0.5391519999999999, 0.0)  (0.0, 0.0, 0.0, 1.0)  b'left_shoulder'
  8  left_wrist      fixed       0     -1     left_wrist      (0.0, 0.0, 0.0)  (0.0, -0.555788, 0.0)            (0.0, 0.0, 0.0, 1.0)  b'left_elbow'
  9  right_hip       spherical   0     -1     right_hip       (0.0, 0.0, 0.0)  (0.0, -0.28, 0.339548)           (0.0, 0.0, 0.0, 1.0)  b'root'
 10  right_knee      revolute   -3.14   0     right_knee      (0.0, 0.0, 1.0)  (0.0, -0.8461839999999999, 0.0)  (0.0, 0.0, 0.0, 1.0)  b'right_hip'
 11  right_ankle     spherical   0     -1     right_ankle     (0.0, 0.0, 0.0)  (0.0, -0.83948, 0.0)             (0.0, 0.0, 0.0, 1.0)  b'right_knee'
 12  left_hip        spherical   0     -1     left_hip        (0.0, 0.0, 0.0)  (0.0, -0.28, -0.339548)          (0.0, 0.0, 0.0, 1.0)  b'root'
 13  left_knee       revolute   -3.14   0     left_knee       (0.0, 0.0, 1.0)  (0.0, -0.8461839999999999, 0.0)  (0.0, 0.0, 0.0, 1.0)  b'left_hip'
 14  left_ankle      spherical   0     -1     left_ankle      (0.0, 0.0, 0.0)  (0.0, -0.83948, 0.0)             (0.0, 0.0, 0.0, 1.0)  b'left_knee'
numActiveThreads = 0
stopping threads
Thread with taskId 0 with handle 0000000000000BBC exiting
Thread TERMINATED
finished
numActiveThreads = 0
btShutDownExampleBrowser stopping threads
Thread with taskId 0 with handle 0000000000000154 exiting
"""


import common
from common import DWJointIndex
from common import DWPose
import dwanim_reader

# We are making the model imitate animation clips
r'''from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
TEST_FBX_PATH = r'C:\Users\rmore\Projects\ASE\ase\poselib\data\01_01_cmu.fbx'
from poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive''' # to use ASE's poselib, not working rn

# We are making a gym (aka OpenAI) environment
import gym
from gym import error, spaces, utils
import numpy as np

# We are making a pybullet physics simulation
import pybullet as p
import pybullet_data

# Utils
from screeninfo import get_monitors
from os.path import join
from tabulate import tabulate


class DeepWalkerEnv(gym.Env):
    '''A gym learning environment that creates a humanoid in pybullet and runs pybullet to determine rewards based on how well the humanoid is walking.'''

    def __init__(self,
                 name='unnamed-deepwalkerenv', 
                 no_anim=False, 
                 direct_anim=False, 
                 anim_to_use=None, 
                 anim_ctrls=False, 
                 use_action=False, 
                 show=True, 
                 debuglines=True, 
                 debug_anim_values=True,
                 verbose=True, 
                 store_rw_history=True, 
                 mp4_path=None):
        """_summary_

        Animation Args:
            no_anim (bool, optional): Whether to NOT play animation. If True the humanoid will just fall on the ground. Defaults to False.
            direct_anim (bool, optional): Whether to directly animate the humanoid, rather than using physics. Defaults to False.
                anim_to_use (bool, optional): A filepath to an anim.xml file to use for every episode/reset. If None, a random one will be selected each episode/reset. Defaults to None.
                anim_ctrls (bool, optional): Whether to directly use the ctrl values from the dwanim. If False, convert the dwanim's joint world rotations to get ctrl values. Defaults to False

        Input Args:
            use_action(bool, optional): Whether to NOT use the action passed into step(). If direct_anim, this is auto set to True. Defaults to False.

        Visualization Args:
            name (str, optional): The model's name, just for printing. Defaults to 'unnamed-deepwalkerenv'.
            show (bool, optional): Whether to show what's going on in a pybullet window. Defaults to True.
            debuglines (bool, optional): Whether to display debug lines used for rewards. Defaults to True.
            verbose (bool, optional): Whether to print a lot. Defaults to True.
            store_rw_history (bool, optional): Whether to store past reward values, for plotting. Defaults to True.
            mp4_path (_type_, optional): _description_. Defaults to None.
        """
        print('init deepwalker env')
        self.name = name

        # Storing animation args
        self.no_anim = no_anim
        self.direct_anim = direct_anim

        self.anim_to_use = anim_to_use

        self.use_random_anims = self.anim_to_use is None
        if self.use_random_anims:
            self.anim_to_use = anim_to_use

        self.anim_ctrls = anim_ctrls

        # Storing input args
        self.use_action = use_action and not self.direct_anim # if direct_anim, set this to true - can't use action ctrls and anim ctrls

        # Storing visualization args
        self.debuglines = debuglines
        self.storing_rw_history = store_rw_history
        self.mp4_path = mp4_path
        self.verbose = verbose

        # Init pybullet
        pybullet_kwargs = {}

        # Kwargs for showing window and saving mp4
        if show:
            #screen = get_monitors()[0]
            #pybullet_kwargs['options'] = f'--width={screen.width-6} --height={screen.height-60}'
            pybullet_kwargs['options'] = f'--width={1920} --height={1080}'

            if self.mp4_path:
                pybullet_kwargs['options'] += f' --mp4="{mp4_path}" --mp4fps=30'
        
        p.connect(p.GUI if show else p.DIRECT, **pybullet_kwargs)


        # Need to make these specifically for the humanoid
        p.resetDebugVisualizerCamera(cameraDistance=16.59, cameraYaw=14.80, cameraPitch=-11.20, cameraTargetPosition=[0.85,-5.84,0.53])
        
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
        
        # Store new data
        self.episode_count += 1
        self.step_count = 0

        # Reset pybullet world
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False) # Disable rendering until we load everything
        if not self.direct_anim:
            p.setGravity(0, 0, -10)
        else:
            p.setGravity(0, 0, -10)

        # Import the 3D stuff
        urdfs_path = pybullet_data.getDataPath()
        plane_uid = p.loadURDF(join( urdfs_path, 'plane.urdf' ), basePosition=[0,0,0])
        self.walker_uid = p.loadURDF(join( urdfs_path, "humanoid\\humanoid.urdf" ), basePosition=[0,0,1])
        
        #self.curr_ctrls = DWPose().ctrl_array()

        if not self.no_anim:
            anim_filepath = None if self.use_random_anims else self.anim_to_use

            # Startup deepwalker anim reader
            self.dwanim_reader = dwanim_reader.DWAnimReader(dwanim_filepath=anim_filepath)
            if self.dwanim_reader.valid:
                print(f"Successfully opened {self.dwanim_reader.dwanim_filepath} ({self.dwanim_reader.num_timesteps} keyframes)")
            else:
                print(f"Failed to open {anim_filepath}")
                return

            # Read frame 0. Frame 0 will be re-read on first step()
            self.curr_ref_pose, _ = self.dwanim_reader.read_at_time(0)

            # Set base pose directly to the frame 0 anim pose
            # This is done whether we're in no_anim mode or not
            self._reset_to_ref_pose()
        
        # Done resetting the pybullet world
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True) # Reenable rendering now that everything is loaded

        # Make initial observation
        #TODO
        observation = None

        if self.storing_rw_history:
            self.reward_breakdown_history.append( [] )

        if self.mp4_path:
            # Start saving a new mp4
            partition = self.mp4_path.rpartition('.mp4')
            reset_mp4_path = partition[0] + f'_ep{self.episode_count}' + partition[1]
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, reset_mp4_path)
        
        
        return observation

    

    def step(self, action):
        """_summary_
        #TODO what is action? ANSWER: action is ctrl array... what format?

        Args:
            action (list of floats?): _description_

        Raises:
            pybullet.error: Any pybullet errors besides closing the pybullet window

        Returns:
            _type_: observation, reward, done, exit
        """
        exit = False

        #TODO action is a flattened ctrl array. each ctrl is 4 floats. parse them into a ctrl array
            

        # Do anim stuff
        if not self.no_anim:

            # Read this timestep's pose
            self.curr_ref_pose, end_of_anim = self.dwanim_reader.read_at_time(self.step_count)

            if end_of_anim:
                # Say we're done with current episode, but we may not want to exit, to play another anim
                print("Animation done")
                return None, None, True, False
            
            if not self.curr_ref_pose.valid:
                print("step(): Read an invalid DWPose:\n- " + "\n- ".join(self.errors))
                return

            if self.direct_anim:
                # Two direct anim modes: match rotations or use anim ctrls
                if self.anim_ctrls:
                    self._set_ctrls_from_ref_pose_ctrls()
                    pass
                    
                else:
                    self._reset_to_ref_pose()

        # Make ctrls, not from anim
        if self.use_action:
            if not action:
                print("No action given")
                return None, None, False, True
            else:
                """p.setJointMotorControlArray( self.walker_uid,
                                            jointIndices = range(len(common.DWJointIndex)),
                                            controlMode = p.POSITION_CONTROL,
                                            targetPositions = action )"""
                #TODO parse controls from action
                #self.curr_ctrls = action
                pass


        # STEP
        try:
            # Render at real time (rather than instant steps)
            p.configureDebugVisualizer( p.COV_ENABLE_SINGLE_STEP_RENDERING )
            # STEP
            p.stepSimulation()

            # Observation
            self._update_curr_pose()
        
        except p.error as e: # If user exits the window, we get this error
            if "Not connected to physics server" in str(e):
                exit = True
            else:
                raise e


        #TODO observation

        observation = None

        #TODO Reward
        reward = 0.0

        self.step_count += 1

        reward = None
        done = None

        return observation, reward, done, exit

        

    

    def close(self):
        if self.mp4_path:
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

        p.disconnect()
    

    def _get_joint_state(self, joint_index):
        """Safely gets a DWJoint's position, rotation, and velocities"""
        #joint_info = p.getJointInfo(self.walker_uid, i)
        if joint_index not in common.REVOLUTE_DW_JOINTS:
            rot, vel, _, _ = p.getJointStateMultiDof(self.walker_uid, joint_index) # TODO check if this is world or local space
            # Check rotation quat
            if not rot or any([np.isnan(r) for r in rot]): # supposed to be a quat but sometimes NaNs or just ()
                rot = [0.0, 0.0, 0.0, 1.0]
            # Check rotational velocity quat
            #if not rotvel or any([np.isnan(rv) for rv in rotvel]): # supposed to be a quat but sometimes NaNs or just ()
            #    rotvel = [0.0, 0.0, 0.0, 1.0]

            joint_rot_euler = p.getEulerFromQuaternion( rot )
            #joint_rotvel_euler = p.getEulerFromQuaternion( rotvel )
        else:
            rot, vel, _, _ = p.getJointState(self.walker_uid, joint_index) # TODO check if this is world or local space
            joint_rot_euler = [rot, 0.0, 0.0]
            #joint_rotvel_euler = [rotvel, 0.0, 0.0]

        link_state = p.getLinkState(self.walker_uid, joint_index) # current shortcut, get link position rather than joint position. inaccurate but we're not using it yet

        return common.BulletJointState( lrot = joint_rot_euler,
                                        #lrotvel = joint_rotvel_euler, 
                                        wpos = link_state[2], 
                                        wposvel = vel or [0,0,0])
    

    def _update_curr_pose(self):
        """Updates self.curr_pose from joint states
        """
        pose = DWPose()

        # State of the Base
        base_pos, base_orn = p.getBasePositionAndOrientation(self.walker_uid)
        base_orn = p.getEulerFromQuaternion(base_orn)

        # State of each Joint
        for i in range(len(common.DWJointIndex)):
            pose[i][1] = self._get_joint_state(i)

            #pose[i][0] = self.curr_ctrls[i]
        
        self.curr_pose = pose
        self.curr_pose.check()
    

    def _reset_to_ref_pose(self):
        """Directly set the deepwalker's pose to be the ref/animation pose, ignoring physics.
        """
        if not self.curr_ref_pose:
            return

        for joint_index, joint in enumerate(self.curr_ref_pose):
            if not joint:
                continue

            joint_ctrl, joint_state = joint
            joint_type = p.getJointInfo(self.walker_uid, joint_index)[2]

            try:
                if joint_index in common.REVOLUTE_DW_JOINTS:
                    p.resetJointState(self.walker_uid, joint_index, targetValue = joint_ctrl)
                
                else:
                    p.resetJointStateMultiDof(self.walker_uid, joint_index, targetValue = joint_ctrl)

            except p.error as e:
                print(f"Failed to reset {common.DW_JOINT_NAMES[joint_index]} to ref pose ctrl: {joint_ctrl} (joint type: {('revolute', 'prismatic', 'spherical', 'planar', 'fixed')[joint_type]})")
                pass

            #p.setActivationState(False) # is there a function for this
        
        # Set base (Hips) loc and rot
        base_pos = self.curr_ref_pose[0][1].wpos
        base_rot = self.curr_ref_pose[0][0]
        p.resetBasePositionAndOrientation(self.walker_uid, base_pos, base_rot) # LEFT OFF
        #print(f"deepwalker_env._reset_to_ref_pose: Set base to rotation {base_rot}, position {base_pos}")
    

    def _set_ctrls_from_ref_pose_ctrls(self, max_force=1000):
        for joint_index, joint in enumerate(self.curr_ref_pose):
            if not joint:
                continue

            joint_ctrl, _ = joint
            if joint_index in common.REVOLUTE_DW_JOINTS:
                p.setJointMotorControl2(self.walker_uid, 
                                        joint_index, 
                                        controlMode = p.POSITION_CONTROL, 
                                        targetPosition = joint_ctrl,
                                        force = max_force)
            else:
                p.setJointMotorControlMultiDof( self.walker_uid, 
                                                joint_index, 
                                                controlMode = p.POSITION_CONTROL, 
                                                targetPosition = joint_ctrl,
                                                force = [max_force, max_force, max_force])


    # Printing
    def print_deepwalker_pose(self):
        print(self.curr_pose.table_str())


    def print_deepwalker_info(self):
        if self.verbose:
            # Make joint info table (jit)
            jit_headers = ('i', 'Joint Name', 'Type', 'Min', 'Max', 'Link Name', 'Axis', 'Pos (local)', 'Orn (local)', 'Parent i' )
            jit_rows = []

            for i in range(p.getNumJoints(self.walker_uid)):
                jit_row = list(p.getJointInfo(self.walker_uid, i))

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
    
    # Displaying Text
    def display_ref_pose_text(self):
        if self.curr_ref_pose and self.curr_ref_pose.valid:
            self._display_pose_text( self.curr_ref_pose, color = [0.2, 1.0, 0.5], offset = [0.0, 0.0, 0.01] )
    
    def display_curr_pose_text(self):
        if self.curr_pose and self.curr_pose.valid:
            self._display_pose_text( self.curr_pose, color = [1.0, .5, 0.2], offset = [0.0, 0.0, -0.01] )

    def _display_pose_text(self, pose, color = [0.2, 1.0, 0.5], skip_zero_ctrl=True, skip_still=False, offset=[0.0, 0.0, 0.0]):
        for joint_index, joint_name, joint_ctrl, joint_state in pose.table_str(return_rows = True):
            # Check if control value is zero
            is_zero_ctrl = False
            if type(joint_ctrl) in (float, int):
                is_zero_ctrl = joint_ctrl == 0.0
            elif type(joint_ctrl) in (list, tuple):
                is_zero_ctrl = all([ c == 0.0 for c in joint_ctrl ])

            if is_zero_ctrl and skip_zero_ctrl:
                continue
             
            if all([ x == 0.0 for vel in (joint_state.lrotvel, joint_state.wrotvel, joint_state.wposvel) for x in vel ]): # if zero velocities
                if skip_still:
                    continue

            pos = [ joint_state.wpos[i] + offset[i] for i in (0,1,2) ]
            p.addUserDebugText(f"{joint_index} {joint_name}: ctrl={joint_ctrl}", pos, lifeTime = common.TIMESTEP, textColorRGB = color)