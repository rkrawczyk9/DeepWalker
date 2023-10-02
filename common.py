from enum import IntEnum

from tabulate import tabulate

from functools import partial

# Makin' angles
from pybullet import getQuaternionFromEuler
from math import pi

TIMESTEP = 1.0 / 240.0

ANIMS_DIR = r"R:\Code\DeepWalkerAnims\anims"
DWANIMS_DIR = r"R:\Code\DeepWalkerAnims\dwanims"
TEST_ANIM_PATH = r"R:\Code\DeepWalkerAnims\tech_data\basepose.dwanim.xml"
BASEPOSE_ANIM_PATH = r"R:\Code\DeepWalkerAnims\tech_data\test_ctrls.dwanim.xml"

DW_JOINT_NAMES = ('Root', 'Chest', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle')
REVOLUTE_DW_JOINTS = 4, 7, 10, 13 # Elbows and knees
def get_default_hips_bullet_state(): return BulletJointState(wpos = [0.0, 0.0, 2.9], wrot = [pi/2, 0, -pi/2])

SPACE_RATIO_ANIM_TO_DW = 1.0 / 10

class DWJointIndex(IntEnum):
    Root = 0
    Chest = 1
    Neck = 2
    RShoulder = 3
    RElbow = 4
    RWrist = 5
    LShoulder = 6
    LElbow = 7
    LWrist = 8
    RHip = 9
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
# A pose is just a list/tuple with these indices containing corresponding humanoid URDF states


class BulletJointState:
    """Kinematic data about a joint (or any object, really), in pybullet space.
    Current rotation, rotational velocity, position, and positional velocity.

    For context, a deepwalker pose is constituted as a tuple, with indices matching each DWJointIndex, where each element is a small tuple containing the motor control value and one of these.
    """
    def __init__(self, lrot=[0.0, 0.0, 0.0], lrotvel=[0.0, 0.0, 0.0], wrot=[0.0, 0.0, 0.0], wrotvel=[0.0, 0.0, 0.0], wpos=[0.0, 0.0, 0.0], wposvel=[0.0, 0.0, 0.0]):
        """
        Args:
            lrot (3 float list, optional): Parent-space rotation.               Defaults to [0.0, 0.0, 0.0].
            lrotvel (3 float list, optional): Parent-space rotational velocity. Defaults to [0.0, 0.0, 0.0].
            wrot (3 float list, optional): World-space rotation.                Defaults to [0.0, 0.0, 0.0].
            wrotvel (3 float list, optional): World-space rotational velocity.  Defaults to [0.0, 0.0, 0.0].
            wpos (3 float list, optional): World-space position.                Defaults to [0.0, 0.0, 0.0].
            wposvel (3 float list, optional): World-space positional velocity.  Defaults to [0.0, 0.0, 0.0].
        """
        self.lrot = lrot
        self.lrotvel = lrotvel
        self.wrot = wrot
        self.wrotvel = wrotvel
        self.wpos = wpos
        self.wposvel = wposvel
    
    def flattened(self):
        """Get this joint state's values as a single tuple of 12 floats. Order: lrot, lrotvel, wrot, wrotvel, wpos, wposvel

        Returns:
            tuple: Tuple containing this instance's values
        """
        return tuple(self.lrot + self.lrotvel + self.wrot + self.wrotvel + self.wpos + self.wposvel)

    # To Strings
    def __str__(self):
        return "<BulletJointState: lrot={:<4} lrotvel={:<4} wrot={:<4} wrotvel={:<4} wpos={:<4} wposvel={:<4}>".format(self.lrot, self.lrotvel, self.wrot, self.wrotvel, self.wpos, self.wposvel)

    def __format__(self, __format_spec: str): return self.__str__()

    def __repr__(self): return self.__str__()


class MayaJointState:
    """Kinematic data about a joint (or any object, really), in maya space.
    Current rotation, rotational velocity, position, and positional velocity.

    For context, here are the data structures where this class is used:
    - An anim pose is constituted as a dictionary of joint names to these.
    - A deepwalker pose is constituted as a tuple, with indices matching each DWJointIndex, where each element is a small tuple containing the motor control value and one of these.
    """
    def __init__(self, lrot=[0.0, 0.0, 0.0], lrotvel=[0.0, 0.0, 0.0], wrot=[0.0, 0.0, 0.0], wrotvel=[0.0, 0.0, 0.0], wpos=[0.0, 0.0, 0.0], wposvel=[0.0, 0.0, 0.0]):
        """
        Args:
            lrot (3 float list, optional): Parent-space rotation.               Defaults to [0.0, 0.0, 0.0].
            lrotvel (3 float list, optional): Parent-space rotational velocity. Defaults to [0.0, 0.0, 0.0].
            wrot (3 float list, optional): World-space rotation.                Defaults to [0.0, 0.0, 0.0].
            wrotvel (3 float list, optional): World-space rotational velocity.  Defaults to [0.0, 0.0, 0.0].
            wpos (3 float list, optional): World-space position.                Defaults to [0.0, 0.0, 0.0].
            wposvel (3 float list, optional): World-space positional velocity.  Defaults to [0.0, 0.0, 0.0].
        """
        self.lrot = lrot
        self.lrotvel = lrotvel
        self.wrot = wrot
        self.wrotvel = wrotvel
        self.wpos = wpos
        self.wposvel = wposvel
    
    def flattened(self):
        """Get this anim joint state's values as a single tuple of 12 floats. Order: lrot, lrotvel, wrot, wrotvel, wpos, wposvel

        Returns:
            tuple: Tuple containing this instance's values
        """
        return tuple(self.lrot + self.lrotvel + self.wrot + self.wrotvel + self.wpos + self.wposvel)

    # To Strings
    def __str__(self):
        return "<MayaJointState: lrot={:<4} lrotvel={:<4} wrot={:<4} wrotvel={:<4} wpos={:<4} wposvel={:<4}>".format(self.lrot, self.lrotvel, self.wrot, self.wrotvel, self.wpos, self.wposvel)

    def __format__(self): return self.__str__()

    def __repr__(self): return self.__str__()


def maya_to_bullet_joint_state(maya_joint_state):
    """Does the Maya to PyBullet space conversion

    Args:
        anim_joint_state (MayaJointState): _description_

    Returns:
        BulletJointState: _description_
    """
    #TODO actual conversion
    # TEMP BYPASS: one to one
    lrot = maya_joint_state.lrot
    lrotvel =  maya_joint_state.lrotvel
    wrot = maya_joint_state.wrot
    wrotvel =  maya_joint_state.wrotvel
    axis_order = (0, 2, 1)
    axis_multipliers = (1, 1, -1)
    wpos = [
        SPACE_RATIO_ANIM_TO_DW * maya_joint_state.wpos[0],
        SPACE_RATIO_ANIM_TO_DW * maya_joint_state.wpos[1],
        SPACE_RATIO_ANIM_TO_DW * maya_joint_state.wpos[2]
    ]
    wposvel = [ maya_joint_state.wposvel[ i ] * SPACE_RATIO_ANIM_TO_DW for i in axis_order ]

    return BulletJointState(lrot=lrot, lrotvel=lrotvel, wrot=wrot, wrotvel=wrotvel, wpos=wpos, wposvel=wposvel)



class DWPose(list):
    """Deepwalker's pose at a single point in time.
    The URDF control value and BulletJointState for each joint in Deepwalker."""

    def __init__(self, from_list = None):

        # Presets
        if not from_list:
            # Default - All zeroes
            from_list = [ [[0.0, 0.0, 0.0, 1.0] if i not in REVOLUTE_DW_JOINTS else 0.0, BulletJointState()] for i in range(len(DWJointIndex)) ] # list comprehension because * doesnt make copies
            from_list[0][1] = get_default_hips_bullet_state()
            from_list[0][0] = list(getQuaternionFromEuler( from_list[0][1].wrot ))

        # Initialize from list by stealing elements
        self.clear()
        for elem in from_list:
            self.append(elem)
        
        # Check validity
        self.valid = False
        self.errors = []
        self.check()
    

    def Empty():
        # Empty preset - Nones instead of sublists
        return DWPose([None for _ in range(len(DWJointIndex))]) # list comprehension because * doesnt make copies
    

    def check(self, quiet = True) -> bool:
        self.errors = []

        right_len = len(self) == len(DWJointIndex)
        if not right_len:
            self.errors.append(f"Invalid number of elements: found {len(self)}, should be {len(DWJointIndex)}==len(DWJointIndex)")

        # Check sublists
        sublists_ok = True
        for joint_index, sublist in enumerate(self):
            # Sublist
            if sublist is None:
                sublists_ok = False
                self.errors.append(f"Sublist is None for {DW_JOINT_NAMES[joint_index]}")
                continue
            elif len(sublist) != 2:
                sublists_ok = False
                self.errors.append(f"Wrong length sublist (should be 2) for {DW_JOINT_NAMES[joint_index]}: {sublist} (len {len(sublist)})")
                continue

            ctrl, state = sublist

            # [0] Float ctrl
            if isinstance(ctrl, float):
                #TODO check if right joint
                pass
            # [0] List ctrl
            elif isinstance(ctrl, list):
                #TODO check if right joint
                # Check all floats
                if not all([isinstance(e, float) for e in ctrl]):
                    sublists_ok = False
                    self.errors.append(f"Non-float control element(s) in {DW_JOINT_NAMES[joint_index]}: {ctrl}")
                elif all([q == 0.0 for q in ctrl]):
                    sublists_ok = False
                    self.errors.append(f"Invalid quaternion control for {DW_JOINT_NAMES[joint_index]}: {ctrl}")
            # [0] Wrong type ctrl
            else:
                sublists_ok = False
                self.errors.append(f"Non-float/list control for {DW_JOINT_NAMES[joint_index]}: {ctrl} ({type(ctrl)})")
                
            # [1] State
            if not isinstance(state, BulletJointState):
                sublists_ok = False
                self.errors.append(f"Wrong type state (should be BulletJointState) for {DW_JOINT_NAMES[joint_index]}: {ctrl} ({type(ctrl)})")
                break

        self.valid = right_len and sublists_ok
        if not quiet and not self.valid:
            print("Warning: DWPose Invalid:\n- " + "\n- ".join(self.errors))
        return self.valid
    

    def ctrl_array(self) -> list:
        """The ctrl value for each joint n a list

        Returns:
            list: _description_
        """
        return [ctrl for ctrl, state in self]

    
    def flattened_states(self) -> list:
        """The state of each joint, all flattened into a single list of floats.

        Returns:
            list
        """
        return [f   for joint in self   for f in joint[1].flattened()]
    

    def flattened(self) -> list:
        """Ctrls and states flattened into a list of floats

        Returns:
            list of floats: Flattened deepwalker pose, should be length 13 * num joints in pose
        """
        flattened = []
        for joint in self:
            flattened += [ joint[0] ] # ctrl
            flattened += joint[1].flattened() # bulletjointstate
        return flattened
    

    # To String's
    def table_str(self, return_rows=False) -> str:
        """Makes a table of each joint's index, name, current control value, and current state"""
        if not self.valid:
            return "<Invalid deepwalker pose>"

        rows = []
        for i, joint in enumerate(self):
            ctrl, state = joint
            rows.append( [i, DW_JOINT_NAMES[i], ctrl, state] )
        
        headers = ('i', 'DWJoint', 'Ctrl', 'State')
        if return_rows:
            return rows
        else:
            return tabulate(rows, headers=headers)
    
    def __str__(self) -> str:
        #list_str = super().__str__()
        #return list_str + f"<{'valid' if self.valid else 'invalid'}>"
        return self.table_str()
    
    def __format__(self, __format_spec: str): return self.__str__()

    def __repr__(self): return self.__str__()


    # Statics
    def Blend(pose_a, pose_b, b_weight):
        """Blends the values of two DWPoses

        Args:
            pose_a (DWPose): _description_
            pose_b (DWPose): _description_
            b_weight (float): Blend weight. 0 to 1.

        Returns:
            DWPose: A new DWPose with the blended values
        """
        interp_float = partial( _weighted_mean, b_weight = b_weight )

        interp_pose = []

        for i in range(len(pose_a)):
            if not pose_a[i] or not pose_b[i]:
                continue

            # Unpack
            ctrl_a, state_a = pose_a[i]
            ctrl_b, state_b = pose_b[i]
            
            # Interp each subcomponent of the BulletJointState
            lrot_y =    interp_float( state_a.lrot[0],      state_b.lrot[0] )
            lrot_p =    interp_float( state_a.lrot[1],      state_b.lrot[1] )
            lrot_r =    interp_float( state_a.lrot[2],      state_b.lrot[2] )
            lrotvel_y = interp_float( state_a.lrotvel[0],   state_b.lrotvel[0] )
            lrotvel_p = interp_float( state_a.lrotvel[1],   state_b.lrotvel[1] )
            lrotvel_r = interp_float( state_a.lrotvel[2],   state_b.lrotvel[2] )
            wrot_y =    interp_float( state_a.wrot[0],      state_b.wrot[0] )
            wrot_p =    interp_float( state_a.wrot[1],      state_b.wrot[1] )
            wrot_r =    interp_float( state_a.wrot[2],      state_b.wrot[2] )
            wrotvel_y = interp_float( state_a.wrotvel[0],   state_b.wrotvel[0] )
            wrotvel_p = interp_float( state_a.wrotvel[1],   state_b.wrotvel[1] )
            wrotvel_r = interp_float( state_a.wrotvel[2],   state_b.wrotvel[2] )
            wpos_x =    interp_float( state_a.wpos[0],      state_b.wpos[0] )
            wpos_y =    interp_float( state_a.wpos[1],      state_b.wpos[1] )
            wpos_z =    interp_float( state_a.wpos[2],      state_b.wpos[2] )
            wposvel_x = interp_float( state_a.wposvel[0],   state_b.wposvel[0] )
            wposvel_y = interp_float( state_a.wposvel[1],   state_b.wposvel[1] )
            wposvel_z = interp_float( state_a.wposvel[2],   state_b.wposvel[2] )

            # Construct new ref state
            state = BulletJointState(   lrot = [lrot_y, lrot_p, lrot_r], 
                                        lrotvel = [lrotvel_y, lrotvel_p, lrotvel_r], 
                                        wrot = [wrot_y, wrot_p, wrot_r], 
                                        wrotvel = [wrotvel_y, wrotvel_p, wrotvel_r], 
                                        wpos = [wpos_x, wpos_y, wpos_z], 
                                        wposvel = [wposvel_x, wposvel_y, wposvel_z] )

            # Interp the control value
            if isinstance(ctrl_a, float):
                ctrl = interp_float(ctrl_a, ctrl_b)
            else:
                ctrl = [ interp_float(ctrl_a[i], ctrl_b[i]) for i in range(len(ctrl_a)) ]

            # Alternative for possible rotation quality improvement: Recalculate the control value from the new ref state
            #ctrl = _ref_state_to_ctrl(state)

            interp_pose.append( [ctrl, state] )

        return DWPose(interp_pose)


    def Unflatten(flattened):
        """Unflattens a list of (13 * num joints) floats into deepwalker pose format

        Args:
            flattened (list of floats): should be length 13 * num joints in pose

        Returns:
            tuple: As deepwalker pose
        """
        unflattened = []
        if len(flattened) % (1 + 12) != 0:
            print(f"Error: unflatten_dwpose: flattened is wrong length. Should be in chunks of 13")
            return False

        for joint_i in range(0, len(flattened), 1 + 12):
            ctrl =      unflattened[joint_i + 0]

            lrot =      unflattened[joint_i + 1  : joint_i + 4] # should be 3-elem list, y p r
            lrotvel =   unflattened[joint_i + 4  : joint_i + 7] # should be 3-elem list, y p r
            wrot =      unflattened[joint_i + 1  : joint_i + 4] # should be 3-elem list, y p r
            wrotvel =   unflattened[joint_i + 4  : joint_i + 7] # should be 3-elem list, y p r
            wpos =      unflattened[joint_i + 7  : joint_i + 10] # should be 3-elem list, x y z
            wposvel =   unflattened[joint_i + 10 : joint_i + 13] # should be 3-elem list, x y z

            state = BulletJointState(lrot=lrot, lrotvel=lrotvel, wrot=wrot, wrotvel=wrotvel, wpos=wpos, wposvel=wposvel)

            unflattened.append( (ctrl, state) )
        
        return DWPose(unflattened)


def _weighted_mean(float_a, float_b, b_weight = 0.5):
    return (1 - b_weight)*float_a + (b_weight)*float_b 