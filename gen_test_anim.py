import xml.etree.ElementTree as xml # our deepwalker anims are saved in xml format
import dwanim_convert
import common
from common import DWPose
from pybullet import getQuaternionFromEuler


def main():
    generate_controls_test_dwanim()

def generate_controls_test_dwanim(max_angle=2, time_per_joint=360):
    
    root = xml.Element("dwanim")
    tree = xml.ElementTree(root)

    for joint_index in range(len(common.DWJointIndex)):

        # Make a pose where the control is positive and one where it's negative
        for signed_or_not, x in enumerate((max_angle, -max_angle)):

            # Determine keyframe time
            start_time = (joint_index * time_per_joint) + (signed_or_not * time_per_joint / 2)

            # For single-axis joints, only test the one axis
            if joint_index in common.REVOLUTE_DW_JOINTS:
                pose = DWPose()
                pose[joint_index][0] = x
                pose[joint_index][1].lrot = [x,0,0]
                dwanim_convert._add_timestep_to_xml(root, int(start_time), pose)

            # For multi-axis joints, make a pose testing each axis
            else:
                pose1 = DWPose()
                rot1 = [x,0,0]
                pose1[joint_index][0] = getQuaternionFromEuler(rot1)
                pose1[joint_index][1].lrot = rot1
                dwanim_convert._add_timestep_to_xml(root, int(start_time + 0), pose1)

                pose2 = DWPose()
                rot2 = [0,x,0]
                pose2[joint_index][0] = getQuaternionFromEuler(rot2)
                pose2[joint_index][1].lrot = rot2
                dwanim_convert._add_timestep_to_xml(root, int(start_time + (1 * time_per_joint / 6)), pose2)

                pose3 = DWPose()
                rot3 = [x,0,0]
                pose3[joint_index][0] = getQuaternionFromEuler(rot3)
                pose3[joint_index][1].lrot = rot3
                dwanim_convert._add_timestep_to_xml(root, int(start_time + (2 * time_per_joint / 6)), pose3)

    # Make indents (if you don't do this it's all one long line)
    xml.indent(tree)

    # Save to file
    tree.write(common.TEST_ANIM_PATH)

    print(f"Generated the animation that tests positive and negative control values.\nSaved to {common.TEST_ANIM_PATH}")
    


if __name__ == "__main__":
    main()