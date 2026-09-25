"""Generate contact-instrumented model overlays without editing PX4 assets."""

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def prepare(source: Path, destination: Path) -> None:
    model_names = ("x500_base", "OakD-Lite", "x500", "x500_depth")
    destination = destination.resolve()
    for model_name in model_names:
        original = source / model_name
        output = destination / model_name
        output.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(original / "model.sdf")
        for include in tree.getroot().iter("include"):
            uri = include.find("uri")
            name = uri.text.removeprefix("model://")
            if name in model_names:
                uri.text = str(destination / name / "model.sdf")
        for uri in tree.getroot().iter("uri"):
            prefix = f"model://{model_name}/"
            if uri.text and uri.text.startswith(prefix):
                uri.text = (original / uri.text[len(prefix):]).resolve().as_uri()
        for link in tree.getroot().find("model").findall("link"):
            collisions = link.findall("collision")
            if not collisions:
                continue
            sensor = ET.SubElement(link, "sensor", name="rendezvous_contact", type="contact")
            contact = ET.SubElement(sensor, "contact")
            for collision in collisions:
                ET.SubElement(contact, "collision").text = collision.get("name")
            # Contact system uses contact/topic, not sensor/topic in Harmonic.
            ET.SubElement(contact, "topic").text = "/aerointercept/contacts"
        if model_name == "x500":
            plugin = ET.SubElement(tree.getroot().find("model"), "plugin",
                filename="gz-sim-pose-publisher-system", name="gz::sim::systems::PosePublisher")
            for key, value in {
                "publish_model_pose": "true", "publish_nested_model_pose": "false",
                "publish_link_pose": "false", "publish_visual_pose": "false",
                "publish_collision_pose": "false", "publish_sensor_pose": "false",
                "use_pose_vector_msg": "true", "update_frequency": "50",
                "topic": "/aerointercept/model_poses",
            }.items():
                ET.SubElement(plugin, key).text = value
        ET.indent(tree)
        tree.write(output / "model.sdf", encoding="utf-8", xml_declaration=True)
        shutil.copyfile(original / "model.config", output / "model.config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
