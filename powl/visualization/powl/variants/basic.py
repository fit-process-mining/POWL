from __future__ import annotations

import importlib.resources
import tempfile

from graphviz import Digraph

from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.base import TaggedPOWL
from powl.objects.tagged_powl.choice_graph import ChoiceGraph
from powl.objects.tagged_powl.partial_order import PartialOrder

min_width = "1.5"
fillcolor = "#fcfcfc"
opacity_change_ratio = 0.02
FONT_SIZE = "18"
PEN_WIDTH = "1"


def apply(powl: TaggedPOWL) -> Digraph:
    filename = tempfile.NamedTemporaryFile(suffix=".gv")

    viz = Digraph("powl", filename=filename.name, engine="dot")
    viz.attr("node", shape="ellipse", fixedsize="false")
    viz.attr(nodesep="1")
    viz.attr(ranksep="1")
    viz.attr(compound="true")
    viz.attr(overlap="scale")
    viz.attr(splines="true")
    viz.attr(rankdir="TB")
    viz.attr(style="filled")
    viz.attr(fillcolor=fillcolor)

    repr_powl(powl, viz, level=0)
    viz.format = "svg"

    return viz


def mark_block(block, powl: TaggedPOWL):
    skip_order = powl.is_skippable()
    loop_order = powl.is_repeatable()

    if skip_order:
        icon = "skip-loop-tag.svg" if loop_order else "skip-tag.svg"
    elif loop_order:
        icon = "loop-tag.svg"
    else:
        block.attr(label="")
        return

    with importlib.resources.path("powl.visualization.powl.variants.icons", icon) as gimg:
        image = str(gimg)
        block.attr(
            label=f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                                <TR><TD WIDTH="55" HEIGHT="27" FIXEDSIZE="TRUE"><IMG SRC="{image}" SCALE="BOTH"/></TD></TR>
                                </TABLE>>"""
        )
        block.attr(labeljust="r")


def repr_powl(powl: TaggedPOWL, viz, level: int):
    font_size = FONT_SIZE
    this_node_id = str(id(powl))
    current_color = darken_color(fillcolor, amount=opacity_change_ratio * level)
    block_id = None

    if isinstance(powl, Activity):
        if powl.is_silent():
            viz.node(
                this_node_id,
                label="",
                style="filled",
                fillcolor="black",
                shape="square",
                width="0.3",
                height="0.3",
                fixedsize="true",
            )
        else:
            label = escape_for_html(powl.label)
            label = f"<{label}"
            if powl.role is not None and powl.organization is not None:
                label += f"""<br/><font color="grey" point-size="10">({powl.organization}, {powl.role})</font><br/>"""
            elif powl.organization is not None:
                label += f"""<br/><font color="grey" point-size="10">({powl.organization})</font><br/>"""
            elif powl.role is not None:
                label += f"""<br/><font color="grey" point-size="10">({powl.role})</font><br/>"""
            label += ">"

            icon = None
            if powl.is_skippable() and powl.is_repeatable():
                icon = "skip-loop-tag.svg"
            elif powl.is_skippable():
                icon = "skip-tag.svg"
            elif powl.is_repeatable():
                icon = "loop-tag.svg"

            if icon:
                with importlib.resources.path("powl.visualization.powl.variants.icons", icon) as gimg:
                    viz.node(
                        this_node_id,
                        label="\n" + label,
                        imagepos="tr",
                        image=str(gimg),
                        shape="box",
                        width=min_width,
                        fontsize=font_size,
                        style="filled",
                        fillcolor=current_color,
                    )
            else:
                viz.node(
                    this_node_id,
                    label,
                    shape="box",
                    fontsize=font_size,
                    width=min_width,
                    style="filled",
                    fillcolor=current_color,
                )

    elif isinstance(powl, PartialOrder):
        block_id = get_block_id(powl)
        child_id_map = {}
        with viz.subgraph(name=block_id) as block:
            block.attr(margin="20,20")
            block.attr(style="filled")
            block.attr(fillcolor=current_color)
            mark_block(block, powl)

            this_node_id = make_anchor(block, block_id)

            reduced = powl.transitive_reduction()
            for child in powl.children:
                child_id_map[child] = repr_powl(child, block, level + 1)
            for u, v in reduced.edges:
                add_edge(block, child_id_map[u], child_id_map[v])

    elif isinstance(powl, ChoiceGraph):
        block_id = get_block_id(powl)
        child_id_map = {}
        with viz.subgraph(name=block_id) as block:
            block.attr(margin="20,20")
            block.attr(style="filled")
            block.attr(fillcolor=current_color)
            mark_block(block, powl)

            this_node_id = make_anchor(block, block_id)
            start_id = f"start_{this_node_id}"
            end_id = f"end_{this_node_id}"

            with importlib.resources.path("powl.visualization.powl.variants.icons", "gate_navy.svg") as gimg:
                gate_img = str(gimg)
                block.node(start_id, label="", shape="diamond", color="navy", width="0.5", height="0.5", fixedsize="true", image=gate_img)
                block.node(end_id, label="", shape="diamond", color="navy", width="0.5", height="0.5", fixedsize="true", image=gate_img)

            for child in powl.children:
                child_id_map[child] = repr_powl(child, block, level + 1)

            for child in powl.start_nodes():
                add_edge(block, (start_id, None), child_id_map[child], color="navy", style="dashed")
            for child in powl.end_nodes():
                add_edge(block, child_id_map[child], (end_id, None), color="navy", style="dashed")
            for u, v in powl.get_edges():
                add_edge(block, child_id_map[u], child_id_map[v], color="navy", style="dashed")

    else:
        raise TypeError(f"Unknown POWL node type: {type(powl)}")

    return this_node_id, block_id


def get_block_id(powl: TaggedPOWL) -> str:
    return "cluster_" + str(id(powl))


def add_edge(block, child_1_ids, child_2_ids, directory="forward", color="black", style=""):
    child_1_base_id, block_id_1 = child_1_ids
    child_2_base_id, block_id_2 = child_2_ids

    kwargs = {
        "dir": directory,
        "color": color,
        "style": style,
        "penwidth": PEN_WIDTH,
    }
    if block_id_1:
        kwargs["ltail"] = block_id_1
        kwargs["minlen"] = "2"
    if block_id_2:
        kwargs["lhead"] = block_id_2
        kwargs["minlen"] = "2"
    block.edge(child_1_base_id, child_2_base_id, **kwargs)


def darken_color(color, amount):
    import matplotlib.colors as mcolors

    amount = min(0.3, amount)

    rgb = mcolors.to_rgb(color)
    darker = [x * (1 - amount) for x in rgb]
    return mcolors.to_hex(darker)


def make_anchor(block, block_id):
    anchor_id = f"anchor_{block_id}"
    block.node(
        anchor_id,
        label="",
        shape="point",
        width="0.01",
        height="0.01",
        fixedsize="true",
        style="invis",
    )
    return anchor_id


def escape_for_html(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
