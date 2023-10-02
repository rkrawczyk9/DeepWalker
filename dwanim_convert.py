import anim_reader
import common
from common import DWJointIndex
from common import DWPose

import xml.etree.ElementTree as xml # our deepwalker anims are saved in xml format

from pybullet import getQuaternionFromEuler
import math

from os.path import join
import os # walk
import sys # cmd line args

# Mapping deepwalker's joints to the animations' joint names
# Currently the animations are all using the joint names from humanoid URDF, so this is a trivial map
DWJOINT_TO_ANIM_JOINT_NAME = {
    DWJointIndex.Root:      "root",
    DWJointIndex.Chest:     "chest",
    DWJointIndex.Neck:      "neck",
    DWJointIndex.RShoulder: "right_shoulder",
    DWJointIndex.RElbow:    "right_elbow",
    DWJointIndex.RWrist:    "right_wrist",
    DWJointIndex.LShoulder: "left_shoulder",
    DWJointIndex.LElbow:    "left_elbow",
    DWJointIndex.LWrist:    "left_wrist",
    DWJointIndex.RHip:      "right_hip",
    DWJointIndex.RKnee:     "right_knee",
    DWJointIndex.RAnkle:    "right_ankle",
    DWJointIndex.LHip:      "left_hip",
    DWJointIndex.LKnee:     "left_knee",
    DWJointIndex.LAnkle:    "left_ankle"
}



def process_all_anims():
    """Converts all anim.xml's in the anims dir (recursive) into dwanim.xml's so Deepwalker can understand them"""

    # Check that anims dir works
    if not os.path.exists(common.ANIMS_DIR) or not os.path.exists(common.DWANIMS_DIR):
        print(f"The folder referenced by common.ANIMS_DIR or common.DWANIMS_DIR doesn't exist. (Check {common.ANIMS_DIR} and {common.DWANIMS_DIR})")
        return False

    # Get all anim.xml files in anims dir (recursive)
    anim_filepaths = []
    for dirpath, dirnames, filenames in os.walk(common.ANIMS_DIR):
        anim_filepaths += [ join( dirpath, f ) for f in filenames if f.endswith( ".anim.xml" ) ]
    
    # Convert each one
    results = []
    for anim_filepath in anim_filepaths:
        print(f"Processing anim '{anim_filepath}'")
        success = process_anim(anim_filepath)
        results += [success]
    
    return results


def process_anim(anim_filepath):
    """Converts one anim.xml to dwanim.xml so Deepwalker can understand it
    """
    anim_filename = os.path.basename(anim_filepath)

    dwanim_save_filename = anim_filename.partition(".")[0] + ".dwanim.xml"
    dwanim_save_filepath = join( common.DWANIMS_DIR, dwanim_save_filename )

    reader = anim_reader.AnimReader(anim_filepath)

    #TODO open xml with create/replace permission
    xml_root = xml.Element("dwanim")
    xml_tree = xml.ElementTree(xml_root)

    # For each frame in the animation
    end = False
    for frame_no in range(reader.num_frames):

        #print(f"Processing frame {frame_no}-{frame_no + 1}")

        # Read this frame's pose
        this_anim_pose, end = reader.read_frame(frame_no)
        if end:
            break

        # Convert this frame's pose
        this_dwpose = _anim_pose_to_deepwalker_pose(this_anim_pose)

        # Read next frame's pose, to interpolate
        next_anim_pose, next_end = reader.read_frame(frame_no + 1)
        
        if next_end:
            # We're on the last frame, just do the one. Don't need to interpolate into the next frame
            timestep_no = frame_no * int(common.TIMESTEP/30)
            _add_timestep_to_xml(xml_root, timestep_no, this_dwpose)
            break

        # Convert the next frame's pose
        next_dwpose = _anim_pose_to_deepwalker_pose(next_anim_pose)

        # Calculate Timesteps per Frame
        # TIMESTEP is in seconds per timestep. 1 / that to get timesteps per second. Divide by 30 to get timesteps per frame
        # Hint: It's 8
        timesteps_per_frame = int( 1 / common.TIMESTEP / 30 )

        # Make a pose for each tiny 1/240sec timestep to flesh out the gap until the next 1/30sec frame
        #
        # Diagram:
        #
        # Frames over time:     |       |       |       |       |       |       |       |       |
        # Timesteps over time:  |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #                                                                         ^^^
        # This for loop generates all these in-between timesteps:               ||||||||
        #   (including the first one, which is index 0 of the range() below)    ^
        for timestep_this_frame in range( timesteps_per_frame ):

            timestep_no = frame_no * timesteps_per_frame + timestep_this_frame

            # Interpolate pose
            percent_through_frame = timestep_this_frame / (timesteps_per_frame)
            blended_dwpose = DWPose.Blend(this_dwpose, next_dwpose, percent_through_frame)

            # Add this timestep with the blended pose to the xml tree
            _add_timestep_to_xml(xml_root, timestep_no, blended_dwpose)
    

    # Make indents (if you don't do this it's all one long line)
    xml.indent(xml_tree)

    # Save to file
    xml_tree.write(dwanim_save_filepath)

    # Check
    return check_dwanim(dwanim_save_filepath)





def _add_timestep_to_xml(dwanim_root, timestep_no, dwpose):
    timestep_tree = xml.SubElement(dwanim_root , "timestep")

    no_elem = xml.SubElement(timestep_tree, "no")
    no_elem.text = str(timestep_no)

    pose_elem = xml.SubElement(timestep_tree, "pose")

    for joint_index, joint in enumerate(dwpose):
        if not joint:
            continue
        joint_ctrl, joint_state = joint

        _add_dwjoint_xml_element(pose_elem, joint_index, joint_ctrl, joint_state)


def _add_dwjoint_xml_element(parent_tree, dwjoint_index, dwjoint_control, dwjoint_state):
    """Adds xml elements from data about a deepwalker joint

    Args:
        parent_tree (_type_): _description_
        dwjoint_index (int): _description_
        dwjoint_state (BulletJointState): _description_
    """
    joint_elem = xml.SubElement(parent_tree, "joint")

    index_elem = xml.SubElement(joint_elem, "index")
    index_elem.text = str(dwjoint_index)

    ctrl_elem = xml.SubElement(joint_elem, "ctrl")
    ctrl_elem.text = str(dwjoint_control).replace(" ", "")

    _add_3f_xml_element(joint_elem, "lrot", "y", dwjoint_state.lrot[0], "p", dwjoint_state.lrot[1], "r", dwjoint_state.lrot[2])

    _add_3f_xml_element(joint_elem, "lrotvel", "y", dwjoint_state.lrotvel[0], "p", dwjoint_state.lrotvel[1], "r", dwjoint_state.lrotvel[2])

    _add_3f_xml_element(joint_elem, "wrot", "y", dwjoint_state.wrot[0], "p", dwjoint_state.wrot[1], "r", dwjoint_state.wrot[2])

    _add_3f_xml_element(joint_elem, "wrotvel", "y", dwjoint_state.wrotvel[0], "p", dwjoint_state.wrotvel[1], "r", dwjoint_state.wrotvel[2])

    _add_3f_xml_element(joint_elem, "wpos", "x", dwjoint_state.wpos[0], "y", dwjoint_state.wpos[1], "z", dwjoint_state.wpos[2])

    _add_3f_xml_element(joint_elem, "wposvel", "x", dwjoint_state.wposvel[0], "y", dwjoint_state.wposvel[1], "z", dwjoint_state.wposvel[2])


def _add_3f_xml_element(parent_tree, name, float1_name, float1, float2_name, float2, float3_name, float3):
    # Add element
    elem = xml.SubElement(parent_tree, name)

    # Add subelement and set its value
    float1_elem = xml.SubElement(elem, float1_name)
    float1_elem.text = str(float1)

    # Add subelement and set its value
    float2_elem = xml.SubElement(elem, float2_name)
    float2_elem.text = str(float2)

    # Add subelement and set its value
    float3_elem = xml.SubElement(elem, float3_name)
    float3_elem.text = str(float3)

    return elem, float1_elem, float2_elem, float3_elem


def _anim_pose_to_deepwalker_pose(anim_pose, verbose = False) -> DWPose:
    """Converts a pose from an AnimReader into a deepwalker pose

    Args:
        anim_pose (dict<str: JointState>): The anim pose to convert. Anim joint names to their states. 

    Returns:
        tuple of float with BulletJointState: The new deepwalker pose. The indices of the tuple match the correct DWJointIndex.
                                                Each element is the control value with the reference state for the deepwalker joint.
        
    """

    dw_pose = DWPose.Empty()

    # For each deepwalker joint, find the corresponding anim joint, and convert the anim's pose to a deepwalker pose
    anim_pose_joint_names = anim_pose.keys()


    for dwjoint_index, anim_joint_name in DWJOINT_TO_ANIM_JOINT_NAME.items():

        if verbose:
            print(f"Converting anim joint {anim_joint_name} to DWJoint {dwjoint_index}: {common.DW_JOINT_NAMES[dwjoint_index]}")

        # Find joint by name in anim pose, if it exists
        anim_joint_state = anim_pose.get( anim_joint_name, None ) # type: MayaJointState

        # Skip this dwjoint if no corresponding anim joint found
        if not anim_joint_state:
            continue
        
        # Convert maya space to pybullet space, and any other conversions that are needed down the line
        ref_state = common.maya_to_bullet_joint_state( anim_joint_state )

        # Convert desired joint rotation etc. into a control value for the humanoid URDF joint
        ctrl = _ref_state_to_ctrl(dwjoint_index, ref_state)

        # Add this joint's control value and ref state to a small tuple at the appropriate DWJointIndex
        # This will keep in line with the specifications for a deepwalker pose
        dw_pose[ dwjoint_index ] = (ctrl, ref_state)

    # Update validity
    dw_pose.check( quiet = False )

    if verbose:
        print(f"Final Deepwalker Pose:\n{dw_pose}")

    return dw_pose


def _ref_state_to_ctrl(dwjoint_index, joint_ref_state):
    """Converts a desired reference state to a humanoid URDF control value.

    Args:
        joint_ref_state (BulletJointState): The desired world position, rotation etc. in pybullet space

    Returns:
        float: The control value which can be passed into p.resetJointState
    """
    # LEFT OFF I converted to radians and then quaternions, but a lot of joints, the root especially, are still wigging out
    
    if dwjoint_index in common.REVOLUTE_DW_JOINTS:
        return math.radians( joint_ref_state.lrot[2] )
    else:
        rot = joint_ref_state.wrot if dwjoint_index == 0 else joint_ref_state.lrot
        return list( getQuaternionFromEuler( [ math.radians( x ) for x in rot ] ) )



def check_dwanim(dwanim_filepath):
    #TODO check that frame numbers and timesteps are correct and stuff
    return "check not implemented"


if __name__ == "__main__":
    # Process anims from command line

    successes = []

    # If user didn't enter any specific anims, do them all
    if len(sys.argv) == 1:
        print("Processing all anim.xml's into dwanim.xml's")
        successes = process_all_anims()

    # Otherwise do specific anims
    else:
        for anim_filepath in sys.argv[1:]:
            # Check that it exists
            if not os.path.isfile(anim_filepath):
                print(f"{anim_filepath} is not an anim.xml filepath")
                break

            print(f"Processing {anim_filepath}")
            successes.append( process_anim(anim_filepath) )

    print(f"Done. Successes: {successes}")