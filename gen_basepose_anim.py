import xml.etree.ElementTree as xml # our deepwalker anims are saved in xml format
import dwanim_convert
import common
from pybullet import getQuaternionFromEuler

def generate_basepose_anim():
    # Init xml
    root = xml.Element("dwanim")
    tree = xml.ElementTree(root)

    # Use default DWPose
    basepose = common.DWPose()

    # Add keyframes at 0sec and 10sec
    dwanim_convert._add_timestep_to_xml(root, 0, basepose)
    dwanim_convert._add_timestep_to_xml(root, 10 * 240, basepose)

    # Make indents (if you don't do this it's all one long line)
    xml.indent(tree)

    # Save to file
    tree.write(common.BASEPOSE_ANIM_PATH)

    print(f"Generated the basepose animation.\nSaved to {common.BASEPOSE_ANIM_PATH}")


def main():
    generate_basepose_anim()
if __name__ == "__main__":
    main()