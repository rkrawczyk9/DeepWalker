import dwanim_convert
import gen_test_anim
import gen_basepose_anim
import xml.etree.ElementTree as xml # our deepwalker anims are saved in xml format

import common
from common import DWPose

import os
from random import choice as randomchoice
import math



# There could be optimizations made here, like parsing with pull or stream rather than DOM. Not sure if memory will be an issue
class DWAnimReader:
    """Parses an anim.xml one frame at a time
    """

    def __init__(self, dwanim_filepath = None, select_random = False, debug_pose_history = False, verbose = True):
        """_summary_

        Args:
            dwanim_filepath (_type_, optional): _description_. Defaults to None.
            select_random (bool, optional): If True, a random dwanim from common.DWANIMS_DIR will be chosen. If False, the basepose animation will be chosen. Defaults to False.
            debug_pose_history (bool, optional): If True, saves past poses in a list, for debugging blending. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """
        self.verbose = verbose

        self.make_debug_pose_history = debug_pose_history
        self.debug_pose_history = []

        self.dwanim_filepath = dwanim_filepath
        if not self.dwanim_filepath:

            if not select_random:
                self.dwanim_filepath = common.BASEPOSE_ANIM_PATH
            else:
                # Randomly select an animation from the dwanims folder

                if not os.path.exists(common.DWANIMS_DIR):
                    print(f"Error: common.DWANIMS_DIR is invalid ({common.DWANIMS_DIR})")
                    return

                for _, _, files in os.walk(common.DWANIMS_DIR):
                    dwanim_files = [ f for f in files if f.endswith( ".dwanim.xml" ) and "template" not in f ]
                    self.dwanim_filepath = os.path.join( common.DWANIMS_DIR, randomchoice(dwanim_files) )
                    print(f"Randomly selected dwanim: {self.dwanim_filepath}")
            
                    if not self.dwanim_filepath:
                        raise Exception(f"No dwanims in common.DWANIMS_DIR folder: {common.DWANIMS_DIR}\nFiles: {files}")
                        return
        
        # Re-generate test anim if that's what we're using
        if self.dwanim_filepath == common.TEST_ANIM_PATH:
            gen_test_anim.main()
        elif self.dwanim_filepath == common.BASEPOSE_ANIM_PATH:
            gen_basepose_anim.main()
        
        # Parse whole XML file (could be optimized)
        dwanim_tree = xml.parse(self.dwanim_filepath)

        # Get all frames, they should be on the root level
        self.timestep_trees = dwanim_tree.getroot().findall("./timestep")
        self.num_timesteps = len(self.timestep_trees)

        self.valid = True
        if self.num_timesteps == 0:
            self.valid = False
            raise Exception(f"DWAnimReader is invalid: No timesteps in {self.dwanim_filepath}")
    

    def read_at_time(self, timestep_no, verbose=False, quiet=False):
        """Reads the current animation at the specified time, blending keyframes if applicable.
        If self.make_debug_pose_history, this is where the pose saving happens, and printing

        Args:
            timestep_no (_type_): _description_
            verbose: whether to print how we found the pose for this time
            quiet: whether to suppress errors

        Returns:
            (DWPose, bool): The pose at this time or None, and whether it's past the end of the animation
        """
        if not self.valid:
            if not quiet:
                print("DWAnimReader: Cannot read animation because reader is invalid. Returning default pose.")
            return DWPose(), True

        # Get the matching/latest timestep
        latest_timestep_tree, latest_timestep_no = self._get_latest_timestep(timestep_no, verbose=verbose)
        if not latest_timestep_tree:
            if verbose:
                print(f"read_at_time: End of animation (no latest timestep)")
            self._save_debug_pose(timestep_no, False, None)
            return None, True
        # Parse the pose
        latest_dwpose = _parse_timestep_tree(latest_timestep_tree, verbose=verbose, quiet=quiet)
        
        # If there's a keyframe at this timestep, no need to blend
        if latest_timestep_no == timestep_no:
            if verbose:
                print(f"read_at_time: There's a keyframe at this time ({timestep_no}), ez get")
            self._save_debug_pose(timestep_no, False, latest_dwpose)
            return latest_dwpose, False
        
        # Get next timestep
        next_timestep_tree, next_timestep_no = self._get_next_timestep(timestep_no, verbose=verbose)
        if not next_timestep_tree:
            if verbose:
                print(f"read_at_time: End of animation (no next timestep)")
            self._save_debug_pose(timestep_no, False, None)
            return None, True
        next_dwpose = _parse_timestep_tree(next_timestep_tree, verbose=verbose, quiet=quiet)

        # Blend between A (latest) and B (next)
        if (next_timestep_no - latest_timestep_no) == 0:
            b_weight = 0
        else:
            b_weight = (timestep_no - latest_timestep_no) / (next_timestep_no - latest_timestep_no)
        if verbose:
            print(f"read_at_time: Blending timesteps...   {latest_timestep_no}  ->  {timestep_no} ({int(b_weight*100)}%)  ->  {next_timestep_no}")
        blended_dwpose = common.DWPose.Blend(latest_dwpose, next_dwpose, b_weight)

        self._save_debug_pose(timestep_no, True, blended_dwpose, latest_dwpose, next_dwpose)
        return blended_dwpose, False
    

    def _save_debug_pose(self, timestep_no, blended, pose, src_blend_pose=None, dest_blend_pose=None):
        if self.make_debug_pose_history:
            self.debug_pose_history.append((timestep_no, True, pose))
            if not blended:
                print(f"Debug Pose History| t: {timestep_no}   Direct Keyframe Get\n{pose}\n")
            else:
                print(f"""Debug Pose History| t: {timestep_no}   BLEND\n{src_blend_pose}
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
{dest_blend_pose}
==================================================================================================
{pose}
""")

    
    def _get_latest_timestep(self, timestep_no, verbose = False):
        """_summary_

        Args:
            timestep_no (int): _description_

        Returns:
            (Element, int) or None: appropriate timestep element. None if end of dwanim
        """
        def myprint(msg):
            if verbose:
                print(f"_get_latest_timestep({timestep_no}): {msg}")

        # Try by index, like for baked animation
        myprint(f"trying by index")
        try:
            timestep_tree = self.timestep_trees[timestep_no]
        except IndexError:
            myprint(f"index out of bounds")
            pass
        else:
            # If the timestep at index isn't actually the matching one
            if timestep_no == int(timestep_tree.find("no").text):
                myprint(f"using by index")
                return timestep_tree, timestep_no
        myprint(f"by index failed")

        # Find any timesteps with matching numbers. If there are any, choose the first one
        matching_timesteps = [t for t in self.timestep_trees if timestep_no == int( t.find("no").text )]
        if len(matching_timesteps) > 0:
            myprint(f"using match: {matching_timesteps[0]}")
            timestep_tree = matching_timesteps[0]
            return timestep_tree, int(timestep_tree.find("no").text)
        myprint(f"by match failed")
        
        # Find any timesteps with greater numbers. If there are none, we've reached the end of the file, return None
        later_timesteps = [t for t in self.timestep_trees if timestep_no < int( t.find("no").text )]
        if len(later_timesteps) == 0:
            myprint(f"none greater. this must be the end")
            return None, -1 # Reached end of animation
        myprint(f"none greater check failed")
        
        # Find any previous timesteps. If there are any choose the last one
        prev_timesteps = [t for t in self.timestep_trees if timestep_no > int( t.find("no").text )]
        prev_nos = [int( t.find("no").text ) for t in prev_timesteps]
        myprint(f"prev nos: {prev_nos}")
        if len(prev_timesteps) > 0:
            myprint(f"using latest: {prev_timesteps[-1]}")
            timestep_tree = prev_timesteps[-1]
            return timestep_tree, int(timestep_tree.find("no").text)
        myprint(f"none latest")

        # If no previous, use first timestep in the total file. (There must be some. If there were none, we would have thrown an error during __init__)
        myprint(f"using first in file")
        timestep_tree = self.timestep_trees[0]
        return timestep_tree, int(timestep_tree.find("no").text)
    
    #TODO precomputing, should be easy


    def _get_next_timestep(self, timestep_no, verbose = False):
        """_summary_

        Args:
            timestep_no (int): _description_

        Returns:
            (Element, int) or None: appropriate timestep element. None if end of dwanim
        """
        def myprint(msg):
            if verbose:
                print(f"_get_next_timestep({timestep_no}): {msg}")

        # Try by index, like for baked animation
        myprint(f"trying by index")
        try:
            timestep_tree = self.timestep_trees[timestep_no]
        except IndexError:
            myprint(f"index out of bounds")
            pass
        else:
            # If the timestep at index isn't actually the matching one
            if timestep_no == int(timestep_tree.find("no").text):
                myprint(f"using by index")
                return timestep_tree, timestep_no
        myprint(f"by index failed")

        # Find any timesteps with matching numbers. If there are any, choose the first one
        matching_timesteps = [t for t in self.timestep_trees if timestep_no == int( t.find("no").text )]
        if len(matching_timesteps) > 0:
            myprint(f"using match: {matching_timesteps[0]}")
            timestep_tree = matching_timesteps[0]
            return timestep_tree, int(timestep_tree.find("no").text)
        myprint(f"by match failed")
        
        # Find any timesteps with greater numbers. If there are none, we've reached the end of the file, return None
        later_timesteps = [t for t in self.timestep_trees if timestep_no < int( t.find("no").text )]
        if len(later_timesteps) == 0:
            myprint(f"none greater. this must be the end")
            return None, -1

        # If there are some greater, use first greater
        else:
            myprint(f"using first greater: {later_timesteps[0]}")
            timestep_tree = later_timesteps[0]
            return timestep_tree, int(timestep_tree.find("no").text)


def _parse_3floats(parent_tree, element_str, comp_a_str, comp_b_str, comp_c_str):
    element_tree = parent_tree.find(element_str)
    if element_tree:
        comp_a = float(element_tree.find(comp_a_str).text)
        comp_b = float(element_tree.find(comp_b_str).text)
        comp_c = float(element_tree.find(comp_c_str).text)
        return [comp_a, comp_b, comp_c]
    return


def _parse_timestep_tree(timestep_tree, verbose=False, quiet=True):
        """Read the pose at a timestep from the dwanim xml.

        Args:
            timestep_tree (Element): The timestep XML Element tree

        Returns:
            dwanim_pose (tuple): The deepwalker pose, parsed from the timestep tree
        
        #TODO if the timestep has already been read, cache it
        """        
        def print_or_not(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        print_or_not(f"\nParsing Timestep {timestep_tree}")

        pose_tree = timestep_tree.find("pose")
        dwpose = DWPose.Empty()
        #print(f"before read:\n{dwpose}")
        for joint_tree in pose_tree.findall("./joint"):

            # Read joint index
            dwjointindex = int( joint_tree.find("index").text )

            # Check joint index
            if dwjointindex >= len(common.DWJointIndex):
                if not quiet:
                    print(f"Warning: _parse_timestep: Invalid joint index {dwjointindex} in timestep {timestep_no} of {self.dwanim_filepath}")
                continue

            # Read control value
            ctrl_txt = joint_tree.find("ctrl").text
            if ctrl_txt[0] not in ("(", "["):
                ctrl = float( ctrl_txt )
            else:
                items_txt = ctrl_txt[1:-1].split(",")
                ctrl = [float(i_txt) for i_txt in items_txt]

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

            # Add this ctrl value and ref state to the pose
            dwpose[dwjointindex] = [ ctrl, common.BulletJointState(lrot=lrot, lrotvel=lrotvel, wrot=wrot, wrotvel=wrotvel, wpos=wpos, wposvel=wposvel) ]
        
        dwpose.check(quiet=quiet)
        return dwpose



if __name__ == "__main__":
    # Read template for a test
    dwanimreader = DWAnimReader("../DeepWalkerAnims/template.dwanim.xml")

    done = False
    for i in range(100):
        dwpose, done = dwanimreader.read_timestep(i)
        if done:
            print("done")
            break
        print(dwpose.table_str())