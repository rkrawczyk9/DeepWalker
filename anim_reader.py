import xml.etree.ElementTree as xml # our deepwalker anims are saved in xml format

import common

import os
from random import choice as randomchoice
import math



# There could be optimizations made here, like parsing with pull or stream rather than DOM. Not sure if memory will be an issue
class AnimReader:
    """Parses an anim.xml one frame at a time
    """

    def __init__(self, anim_xml_filepath = None, verbose = True):
        self.verbose = verbose

        if not anim_xml_filepath:

            # Randomly select an animation from the anims folder

            if not common.ANIMS_DIR:
                print("Error: AnimReader.__init__: anims dir needs to be specified")
                return

            for _, _, files in os.walk(common.ANIMS_DIR):
                anim_files = [ f for f in files if f.endswith( ".anim.xml" ) ]
                anim_xml_filepath = os.path.join( common.ANIMS_DIR, randomchoice(anim_files) )
                print(f"Randomly selected anim: {anim_xml_filepath}")
                break
        
        # Parse whole XML file (could be optimized)
        tree = xml.parse(anim_xml_filepath)

        # Get all frames, they should be on the root level
        self.frame_trees = tree.getroot().findall("./frame")
        self.num_frames = len(self.frame_trees)


    def read_frame(self, frame_no):
        """Read a frame from the anim xml.
        #TODO if the frame as already been read, cache it

        Args:
            frame_no (int): The frame number when the pose should be gotten.

        Returns:
            anim_pose, done
            anim_pose (dict< str: MayaJointState >): The new anim pose. (Not in deepwalker format yet)
            done (bool): Whether there are no more frames
        """
        # TODO remove this and actually do linear interpolation between frames
        if isinstance(frame_no, float):
            frame_no = math.floor(frame_no)

        #print(f"\nReading Frame {frame_no}")

        try:
            frame_tree = self.frame_trees[frame_no]
        except IndexError:
            return None, True

        pose_tree = frame_tree.find("pose")
        anim_pose = {}
        for joint_tree in pose_tree.findall("./joint"):
            name = joint_tree.find("name").text

            # Read Local Rotation
            lrot = _parse_3floats(joint_tree, "lrot", "y", "p", "r")

            # Read Local Rotational Velocity
            lrotvel = _parse_3floats(joint_tree, "lrotvel", "y", "p", "r")

            # Read World Rotation
            wrot = _parse_3floats(joint_tree, "wrot", "y", "p", "r")

            # Read World Rotational Velocity
            wrotvel = _parse_3floats(joint_tree, "wrotvel", "y", "p", "r")

            # Read World Position
            wpos = _parse_3floats(joint_tree, "wpos", "x", "y", "z")
            
            # Read World Positional Velocity
            wposvel = _parse_3floats(joint_tree, "wposvel", "x", "y", "z")

            # Save it as a new MayaJointState in the dictionary/pose
            anim_pose[name] = common.MayaJointState(lrot=lrot, lrotvel=lrotvel, wrot=wrot, wrotvel=wrotvel, wpos=wpos, wposvel=wposvel)

        return anim_pose, False
    
    #TODO precomputing, should be easy


def _parse_3floats(parent_tree, element_str, comp_a_str, comp_b_str, comp_c_str):
    element_tree = parent_tree.find(element_str)
    if element_tree:
        comp_a = float(element_tree.find(comp_a_str).text)
        comp_b = float(element_tree.find(comp_b_str).text)
        comp_c = float(element_tree.find(comp_c_str).text)
        return [comp_a, comp_b, comp_c]
    return



if __name__ == "__main__":
    # Read template for a test
    animreader = AnimReader("../DeepWalkerAnims/template.dwanim.xml")

    done = False
    for i in range(100):
        pose, done = animreader.read_frame(i)
        if done:
            print("done")
            break
        print(f"pose = {pose}")