import tempfile
from collections import Counter
from copy import copy
from enum import Enum
from typing import Any, Dict, Optional

import graphviz

from graphviz import Digraph
from pm4py.objects.dfg import util as dfu
from pm4py.objects.dfg.obj import DFG

from pm4py.util import constants, exec_utils


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    FORMAT = "format"
    MAX_NO_EDGES_IN_DIAGRAM = "maxNoOfEdgesInDiagram"
    START_ACTIVITIES = "start_activities"
    END_ACTIVITIES = "end_activities"
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    FONT_SIZE = "font_size"
    RANKDIR = "rankdir"
    BGCOLOR = "bgcolor"


def apply(
    dfg_obj: DFG, parameters: Optional[Dict[Any, Any]] = None
) -> graphviz.Digraph:
    """
    Visualize a frequency directly-follows graph

    Parameters
    -----------------
    dfg_obj
        Directly-follows graph
    parameters
        Variant-specific parameters

    Returns
    -----------------
    gviz
        Graphviz digraph
    """
    if parameters is None:
        parameters = {}

    max_no_of_edges_in_diagram = exec_utils.get_param_value(
        Parameters.MAX_NO_EDGES_IN_DIAGRAM, parameters, 100000
    )
    font_size = exec_utils.get_param_value(Parameters.FONT_SIZE, parameters, 32)
    font_size = str(font_size)

    dfg = dfg_obj.graph
    start_activities = dfg_obj.start_activities
    end_activities = dfg_obj.end_activities

    activities = dfu.get_vertices(dfg_obj)

    rankdir = exec_utils.get_param_value(
        Parameters.RANKDIR, parameters, constants.DEFAULT_RANKDIR_GVIZ
    )
    bgcolor = exec_utils.get_param_value(
        Parameters.BGCOLOR, parameters, constants.DEFAULT_BGCOLOR
    )

    activities_count = Counter({key: 0 for key in activities})
    for el in dfg:
        activities_count[el[1]] += dfg[el]
    if isinstance(start_activities, dict):
        for act in start_activities:
            activities_count[act] += start_activities[act]

    return graphviz_visualization(
        activities_count,
        dfg,
        measure="frequency",
        max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
        start_activities=start_activities,
        end_activities=end_activities,
        font_size=font_size,
        bgcolor=bgcolor,
        rankdir=rankdir,
    )


def graphviz_visualization(
    activities_count,
    dfg,
    start_activities,
    end_activities,
    measure="frequency",
    max_no_of_edges_in_diagram=100000,
    font_size="12",
    bgcolor=constants.DEFAULT_BGCOLOR,
    rankdir=constants.DEFAULT_RANKDIR_GVIZ,
):
    """
    Do GraphViz visualization of a DFG graph

    Parameters
    -----------
    activities_count
        Count of attributes in the log (may include attributes that are not in the DFG graph)
    dfg
        DFG graph
    measure
        Describes which measure is assigned to edges in direcly follows graph (frequency/performance)
    max_no_of_edges_in_diagram
        Maximum number of edges in the diagram allowed for visualization
    start_activities
        Start activities of the log
    end_activities
        End activities of the log
    font_size
        Size of the text on the activities/edges
    bgcolor
        Background color of the visualization (i.e., 'transparent', 'white', ...)
    rankdir
        Direction of the graph ("LR" for left-to-right; "TB" for top-to-bottom)

    Returns
    -----------
    viz
        Digraph object
    """

    filename = tempfile.NamedTemporaryFile(suffix=".gv")
    filename.close()

    viz = Digraph(
        "",
        filename=filename.name,
        engine="dot",
        graph_attr={"bgcolor": bgcolor, "rankdir": 'TB'},
    )

    # first, remove edges in diagram that exceeds the maximum number of edges in the diagram
    dfg_key_value_list = []
    for edge in dfg:
        dfg_key_value_list.append([edge, dfg[edge]])
    # more fine grained sorting to avoid that edges that are below the threshold are
    # undeterministically removed
    dfg_key_value_list = sorted(
        dfg_key_value_list, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True
    )
    dfg_key_value_list = dfg_key_value_list[
        0 : min(len(dfg_key_value_list), max_no_of_edges_in_diagram)
    ]
    dfg_allowed_keys = [x[0] for x in dfg_key_value_list]
    dfg_keys = list(dfg.keys())
    for edge in dfg_keys:
        if edge not in dfg_allowed_keys:
            del dfg[edge]

    activities_count_int = copy(activities_count)

    activities_in_dfg = set(activities_count)

    # represent nodes
    viz.attr("node", shape="box")

    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities_in_dfg)))

    activities_map = {}

    for act in activities_to_include:
        viz.node(str(hash(act)), act, fontsize=font_size)
        activities_map[act] = str(hash(act))

    # make edges addition always in the same order
    dfg_edges = sorted(list(dfg.keys()))

    # represent edges
    for edge in dfg_edges:
        label = str(dfg[edge])
        viz.edge(
            str(hash(edge[0])), str(hash(edge[1])), fontsize=font_size, penwidth="2.0"
        )

    start_activities_to_include = [
        act for act in start_activities if act in activities_map
    ]
    end_activities_to_include = [act for act in end_activities if act in activities_map]

    if start_activities_to_include:
        viz.node(
            "@@startnode",
            "START",
            style="filled",
            fillcolor="lightgrey",
            fontsize=font_size,
        )
        for act in start_activities_to_include:
            label = (
                str(start_activities[act])
                if isinstance(start_activities, dict) and measure == "frequency"
                else ""
            )
            viz.edge(
                "@@startnode", activities_map[act], fontsize=font_size, penwidth="2.0"
            )

    if end_activities_to_include:
        # <&#9632;>
        viz.node(
            "@@endnode",
            "END",
            style="filled",
            fillcolor="lightgrey",
            fontsize=font_size,
        )
        for act in end_activities_to_include:
            label = (
                str(end_activities[act])
                if isinstance(end_activities, dict) and measure == "frequency"
                else ""
            )
            viz.edge(
                activities_map[act], "@@endnode", fontsize=font_size, penwidth="2.0"
            )

    viz.attr(overlap="false")
    viz.attr(fontsize="11")

    viz.format = "SVG"

    return viz
