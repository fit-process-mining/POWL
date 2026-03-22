import base64
import copy
import os
import re
import shutil
import tempfile
import webbrowser
from enum import Enum

from pm4py.util import exec_utils

from powl.objects.tagged_powl.base import TaggedPOWL
from powl.objects.tagged_powl.builders import expand_frequency_tags

from powl.visualization.powl.variants import basic, net


class POWLVisualizationVariants(Enum):
    BASIC = basic
    NET = net


DEFAULT_VARIANT = POWLVisualizationVariants.BASIC


def inline_images_and_svgs(svg_content):
    img_pattern = re.compile(r'<image[^>]+xlink:href=["\'](.*?)["\'][^>]*>')

    def encode_file_to_base64(file_path):
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    def read_file_content_and_viewbox(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            content = re.sub(r"<\?xml.*?\?>", "", content, flags=re.DOTALL)
            content = re.sub(r"<!DOCTYPE.*?>", "", content, flags=re.DOTALL)
            viewBox_match = re.search(r'viewBox="([^"]*)"', content)
            viewBox = viewBox_match.group(1) if viewBox_match else "0 0 1 1"
            svg_content_match = re.search(r"<svg[^>]*>(.*?)</svg>", content, re.DOTALL)
            svg_content = svg_content_match.group(1) if svg_content_match else content
            return svg_content, viewBox

    def replace_with_inline_content(match):
        file_path = match.group(1)
        if file_path.lower().endswith(".svg"):
            svg_data, viewBox = read_file_content_and_viewbox(file_path)
            viewBox_values = [float(v) for v in viewBox.split()]
            actual_width, actual_height = viewBox_values[2], viewBox_values[3]

            intended_width = float(
                match.group(0).split('width="')[1].split('"')[0].replace("px", "")
            )
            intended_height = float(
                match.group(0).split('height="')[1].split('"')[0].replace("px", "")
            )
            x = float(match.group(0).split('x="')[1].split('"')[0])
            y = float(match.group(0).split('y="')[1].split('"')[0])

            scale_x = intended_width / actual_width
            scale_y = intended_height / actual_height

            return f'<g transform="translate({x},{y}) scale({scale_x},{scale_y})">{svg_data}</g>'
        else:
            base64_data = encode_file_to_base64(file_path)
            return match.group(0).replace(
                file_path, f"data:image/png;base64,{base64_data}"
            )

    return img_pattern.sub(replace_with_inline_content, svg_content)


def apply(powl: TaggedPOWL, variant=DEFAULT_VARIANT, frequency_tags=True) -> str:
    """
    Method for POWL model representation

    Parameters
    -----------
    powl
        POWL model
    parameters
        Possible parameters of the algorithm:
            Parameters.FORMAT -> Format of the image (PDF, PNG, SVG; default PNG)
    variant
        Variant of the algorithm to use:
            - POWLVisualizationVariants.BASIC (default)
            - POWLVisualizationVariants.NET: BPMN-like visualization with decision gates
    frequency_tags
        If True, show skip/repeat tags directly on the rendered elements.
        If False, materialize those tags into explicit visualization-only structure.

    Returns
    -----------
    str
        SVG Content
    """
    powl = copy.deepcopy(powl)
    if not frequency_tags:
        powl = expand_frequency_tags(powl, expand_activities=True)

    viz = exec_utils.get_variant(variant).apply(powl)
    svg_content = viz.pipe().decode("utf-8")

    svg_content_with_inline_images = inline_images_and_svgs(svg_content)

    return svg_content_with_inline_images


def save(svg_content: str, output_file_path: str):
    """
    Save the diagram in a specified format.

    Parameters
    -----------
    svg_content : str
        SVG content
    output_file_path : str
        Path where the output file should be saved
    """

    with tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".svg") as tmpfile:
        tmpfile.write(svg_content)
        tmpfile_path = tmpfile.name

    if output_file_path.endswith("svg"):
        shutil.move(tmpfile_path, output_file_path)
    else:
        raise Exception(f"Unsupported format! Please use 'svg'.")

    if os.path.exists(tmpfile_path):
        os.remove(tmpfile_path)


def view(svg_content: str):
    """
    View the diagram

    Parameters
    -----------
    svg_content
        SVG content
    """

    with tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".svg") as tmpfile:
        tmpfile.write(svg_content)
        tmpfile_path = tmpfile.name

        absolute_path = os.path.abspath(tmpfile_path)
        return webbrowser.open("file://" + absolute_path)
