import xml.etree.ElementTree as ET
import numpy as np


def add_fromto_sensors(xml_path, target_geoms, goal_geoms, cutoff=1.0):
    """
    Clears any existing <fromto> sensors and adds new ones for all (target, goal) pairs to the <sensor> section of a MuJoCo XML file.
    NOTE: This also changes the formatting of the scene.xml, pay attention to not commit these changes.

    Args:
        xml_path (str): Path to the MuJoCo XML file.
        target_geoms (list of str): List of target geometry names.
        goal_geoms (list of str): List of goal geometry names.
        cutoff (float): Cutoff distance for the sensor.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")
    else:
        # Remove all existing children (clear all sensors)
        for child in list(sensor_elem):
            sensor_elem.remove(child)

    for geom1 in target_geoms:
        for geom2 in goal_geoms:
            fromto_elem = ET.Element("fromto")
            fromto_elem.set("geom1", geom1)
            fromto_elem.set("geom2", geom2)
            fromto_elem.set("cutoff", str(cutoff))
            sensor_elem.append(fromto_elem)

    tree.write(xml_path)


def geom_distance(
    model, data, geom1_name: str, geom2_name: str, distmax: float = 1.0
) -> float:
    """
    Calculate the minimum distance between two geoms using mujoco.mj_geomDistance
    https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance"

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object.
        geom1_name (str): Name of the first geometry.
        geom2_name (str): Name of the second geometry.
        distmax (float): The maximum distance to consider.

    Returns:
        float: The minimum distance between the two geoms or distmax if the distance is greater than distmax.
    """
    geom1_id = model.geom(geom1_name).id
    geom2_id = model.geom(geom2_name).id
    fromto = np.zeros(6)
    dist = mujoco.mj_geomDistance(model, data, geom1_id, geom2_id, distmax, fromto)
    return dist
