"""Layout post-processing for BPMN Visualizer.
Calculates bounding boxes for lanes and pools strictly based on flow nodes,
places text annotations with collision avoidance (considering element name labels),
and calculates orthogonal waypoints for all flows.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any

from .constants import (
    POOL_HEADER_WIDTH,
    POOL_MARGIN_X,
    POOL_MARGIN_Y,
    LANE_PADDING_TOP,
    LANE_PADDING_BOTTOM,
    LANE_PADDING_LEFT,
    LANE_PADDING_RIGHT,
    BLACKBOX_MARGIN_TOP,
    ELEMENT_BOUNDS,
    LABEL_CARDINAL_GAP,
    LABEL_CORNER_MARGIN_X,
    LABEL_CORNER_MARGIN_Y,
)
from .input_parser import (
    BPMNModel,
    Process,
    Lane,
    FlowNode,
    FlowEdge,
    Participant,
    TextAnnotation,
    Bounds,
    Waypoint,
)


def _get_node_obstacle_bounds(node: FlowNode) -> Bounds:
    """Calculate the full obstacle bounding box for a node including its external name label."""
    nb = node.bounds
    x, y, w, h = nb.x, nb.y, nb.width, nb.height

    if node.label_bounds is not None:
        lb = node.label_bounds
        min_x = min(x, lb.x)
        min_y = min(y, lb.y)
        max_x = max(x + w, lb.x + lb.width)
        max_y = max(y + h, lb.y + lb.height)
        return Bounds(
            x=round(min_x, 1),
            y=round(min_y, 1),
            width=round(max_x - min_x, 1),
            height=round(max_y - min_y, 1),
        )

    if node.name:
        # Events and Gateways have external labels below the shape
        if "Event" in node.element_type or "Gateway" in node.element_type:
            label_w = max(w + 30.0, len(node.name) * 7.0)
            label_lines = 2 if len(node.name) > 15 else 1
            label_h = 14.0 * label_lines
            total_h = h + 6.0 + label_h
            min_x = min(x, x + (w - label_w) / 2.0)
            max_x = max(x + w, x + (w + label_w) / 2.0)
            return Bounds(
                x=round(min_x, 1),
                y=round(y, 1),
                width=round(max_x - min_x, 1),
                height=round(total_h, 1),
            )

    return Bounds(x=x, y=y, width=w, height=h)


def _check_box_overlap(b1: Bounds, b2: Bounds, margin: float = 4.0) -> bool:
    """Check whether two bounding boxes overlap, accounting for a safety margin."""
    x1 = b1.x - margin
    y1 = b1.y - margin
    w1 = b1.width + 2 * margin
    h1 = b1.height + 2 * margin

    x2 = b2.x
    y2 = b2.y
    w2 = b2.width
    h2 = b2.height

    overlap_x = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return (overlap_x > 0.0) and (overlap_y > 0.0)


def _segment_intersects_box(p1: Waypoint, p2: Waypoint, box: Bounds, margin: float = 4.0) -> bool:
    """Check whether an orthogonal line segment intersects a bounding box (with safety margin)."""
    bx1 = box.x - margin
    bx2 = box.x + box.width + margin
    by1 = box.y - margin
    by2 = box.y + box.height + margin

    # Horizontal segment
    if abs(p1.y - p2.y) < 1e-3:
        y = p1.y
        if by1 < y < by2:
            seg_min_x = min(p1.x, p2.x)
            seg_max_x = max(p1.x, p2.x)
            overlap_x = min(seg_max_x, bx2) - max(seg_min_x, bx1)
            if overlap_x > 0.1:
                return True
        return False

    # Vertical segment
    if abs(p1.x - p2.x) < 1e-3:
        x = p1.x
        if bx1 < x < bx2:
            seg_min_y = min(p1.y, p2.y)
            seg_max_y = max(p1.y, p2.y)
            overlap_y = min(seg_max_y, by2) - max(seg_min_y, by1)
            if overlap_y > 0.1:
                return True
        return False

    # Diagonal fallback
    seg_min_x, seg_max_x = min(p1.x, p2.x), max(p1.x, p2.x)
    seg_min_y, seg_max_y = min(p1.y, p2.y), max(p1.y, p2.y)
    overlap_x = min(seg_max_x, bx2) - max(seg_min_x, bx1)
    overlap_y = min(seg_max_y, by2) - max(seg_min_y, by1)
    return (overlap_x > 0.1) and (overlap_y > 0.1)


def _path_collides(path: List[Waypoint], obstacles: List[Bounds], margin: float = 4.0) -> bool:
    """Check whether any segment in the waypoint path collides with any obstacle."""
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        for obs in obstacles:
            if _segment_intersects_box(p1, p2, obs, margin=margin):
                return True
    return False


def _clean_waypoints(waypoints: List[Waypoint]) -> List[Waypoint]:
    """Remove redundant duplicate or collinear waypoints."""
    if len(waypoints) <= 2:
        return [Waypoint(x=round(w.x, 1), y=round(w.y, 1)) for w in waypoints]

    cleaned: List[Waypoint] = [Waypoint(x=round(waypoints[0].x, 1), y=round(waypoints[0].y, 1))]
    for w in waypoints[1:]:
        rw = Waypoint(x=round(w.x, 1), y=round(w.y, 1))
        if abs(rw.x - cleaned[-1].x) < 1e-3 and abs(rw.y - cleaned[-1].y) < 1e-3:
            continue
        cleaned.append(rw)

    simplified: List[Waypoint] = []
    for i in range(len(cleaned)):
        if 0 < i < len(cleaned) - 1:
            prev_p = simplified[-1]
            curr_p = cleaned[i]
            next_p = cleaned[i + 1]
            if abs(prev_p.x - curr_p.x) < 1e-3 and abs(curr_p.x - next_p.x) < 1e-3:
                continue
            if abs(prev_p.y - curr_p.y) < 1e-3 and abs(curr_p.y - next_p.y) < 1e-3:
                continue
        simplified.append(cleaned[i])

    return simplified


def _get_node_ports(bounds: Bounds) -> Dict[str, Waypoint]:
    """Get the 4 standard cardinal ports for a node's bounding box."""
    cx = round(bounds.x + bounds.width / 2.0, 1)
    cy = round(bounds.y + bounds.height / 2.0, 1)
    left_x = round(bounds.x, 1)
    right_x = round(bounds.x + bounds.width, 1)
    top_y = round(bounds.y, 1)
    bottom_y = round(bounds.y + bounds.height, 1)

    return {
        "left": Waypoint(x=left_x, y=cy),
        "right": Waypoint(x=right_x, y=cy),
        "top": Waypoint(x=cx, y=top_y),
        "bottom": Waypoint(x=cx, y=bottom_y),
    }


def _determine_node_port_usage(proc: Process) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Determine which ports are naturally targeted by incoming flows for each node."""
    incoming_ports: Dict[str, set] = {nid: set() for nid in proc.flow_nodes}
    outgoing_ports: Dict[str, set] = {nid: set() for nid in proc.flow_nodes}

    for sf in proc.sequence_flows:
        s_node = proc.flow_nodes.get(sf.source_ref)
        t_node = proc.flow_nodes.get(sf.target_ref)
        if not s_node or not t_node:
            continue

        sb = s_node.bounds
        tb = t_node.bounds
        s_cy = sb.y + sb.height / 2.0
        t_cy = tb.y + tb.height / 2.0
        s_cx = sb.x + sb.width / 2.0
        t_cx = tb.x + tb.width / 2.0

        # Determine target ingress port
        if "Gateway" in t_node.element_type and len(t_node.incoming) > 1:
            if s_cy < tb.y - 15.0:
                incoming_ports[t_node.id].add("top")
            elif s_cy > (tb.y + tb.height + 15.0):
                incoming_ports[t_node.id].add("bottom")
            else:
                incoming_ports[t_node.id].add("left")
        else:
            if sb.x + sb.width <= tb.x + 15.0:
                incoming_ports[t_node.id].add("left")
            elif sb.y + sb.height <= tb.y + 15.0:
                incoming_ports[t_node.id].add("top")
            elif sb.y >= tb.y + tb.height - 15.0:
                incoming_ports[t_node.id].add("bottom")
            else:
                incoming_ports[t_node.id].add("left")

    return incoming_ports, outgoing_ports


def _find_collision_free_path(
    s_port_name: str,
    s_pt: Waypoint,
    t_port_name: str,
    t_pt: Waypoint,
    obstacles: List[Bounds],
) -> Optional[List[Waypoint]]:
    """Try various orthogonal routing geometries between two specific ports and return the first
    collision-free path with minimal bends, strictly enforcing approach/departure buffer clearance
    at all ports (e.g. top port approaches strictly from above with buffer, bottom from below, etc.).
    """
    sx, sy = s_pt.x, s_pt.y
    tx, ty = t_pt.x, t_pt.y
    BUFFER = 20.0

    # 1. Direct Straight Line
    if abs(sx - tx) < 1e-3 and s_port_name in ("top", "bottom") and t_port_name in ("top", "bottom"):
        if (s_port_name == "bottom" and t_port_name == "top" and ty >= sy + BUFFER) or \
           (s_port_name == "top" and t_port_name == "bottom" and ty <= sy - BUFFER):
            p = [Waypoint(sx, sy), Waypoint(tx, ty)]
            if not _path_collides(p, obstacles):
                return _clean_waypoints(p)

    if abs(sy - ty) <= 4.0 and s_port_name in ("left", "right") and t_port_name in ("left", "right"):
        if (s_port_name == "right" and t_port_name == "left" and tx >= sx + BUFFER) or \
           (s_port_name == "left" and t_port_name == "right" and tx <= sx - BUFFER):
            p = [Waypoint(sx, sy), Waypoint(tx, sy)]
            if not _path_collides(p, obstacles):
                return _clean_waypoints(p)

    # 2. Single-bend L-Paths (HV or VH)
    if s_port_name in ("left", "right") and t_port_name in ("top", "bottom"):
        valid_s = (s_port_name == "right" and tx >= sx + 5.0) or (s_port_name == "left" and tx <= sx - 5.0)
        valid_t = (t_port_name == "top" and sy <= ty - 2.0) or (t_port_name == "bottom" and sy >= ty + 2.0)
        if valid_s and valid_t:
            p = [Waypoint(sx, sy), Waypoint(tx, sy), Waypoint(tx, ty)]
            if not _path_collides(p, obstacles):
                return _clean_waypoints(p)

    if s_port_name in ("top", "bottom") and t_port_name in ("left", "right"):
        valid_s = (s_port_name == "bottom" and ty >= sy + 2.0) or (s_port_name == "top" and ty <= sy - 2.0)
        valid_t = (t_port_name == "left" and sx <= tx - 5.0) or (t_port_name == "right" and sx >= tx + 5.0)
        if valid_s and valid_t:
            p = [Waypoint(sx, sy), Waypoint(sx, ty), Waypoint(tx, ty)]
            if not _path_collides(p, obstacles):
                return _clean_waypoints(p)

    # 3. Two-bend S/Z-Paths (HVH or VHV)
    if s_port_name in ("left", "right") and t_port_name in ("left", "right"):
        mid_x = round((sx + tx) / 2.0, 1)
        cand_x_list = [mid_x]
        for obs in obstacles:
            if min(sx, tx) - 30.0 <= obs.x + obs.width and obs.x <= max(sx, tx) + 30.0:
                cand_x_list.append(round(obs.x - 25.0, 1))
                cand_x_list.append(round(obs.x + obs.width + 25.0, 1))

        cand_x_list = sorted(set(cand_x_list), key=lambda x: abs(x - mid_x))
        for xc in cand_x_list:
            if s_port_name == "right" and xc < sx + 5.0:
                continue
            if s_port_name == "left" and xc > sx - 5.0:
                continue
            if t_port_name == "left" and xc > tx - 5.0:
                continue
            if t_port_name == "right" and xc < tx + 5.0:
                continue
            p = [Waypoint(sx, sy), Waypoint(xc, sy), Waypoint(xc, ty), Waypoint(tx, ty)]
            if not _path_collides(p, obstacles):
                return _clean_waypoints(p)

    if s_port_name in ("top", "bottom") and t_port_name in ("top", "bottom"):
        mid_y = round((sy + ty) / 2.0, 1)
        cand_y_list = [mid_y]
        for obs in obstacles:
            if min(sy, ty) - 30.0 <= obs.y + obs.height and obs.y <= max(sy, ty) + 30.0:
                cand_y_list.append(round(obs.y - 25.0, 1))
                cand_y_list.append(round(obs.y + obs.height + 25.0, 1))

        cand_y_list = sorted(set(cand_y_list), key=lambda y: abs(y - mid_y))
        for yc in cand_y_list:
            if s_port_name == "bottom" and yc < sy + 5.0:
                continue
            if s_port_name == "top" and yc > sy - 5.0:
                continue
            if t_port_name == "top" and yc > ty - 5.0:
                continue
            if t_port_name == "bottom" and yc < ty + 5.0:
                continue
            p = [Waypoint(sx, sy), Waypoint(sx, yc), Waypoint(tx, yc), Waypoint(tx, ty)]
            if not _path_collides(p, obstacles):
                return _clean_waypoints(p)

    # 4. Three-bend / Four-bend Detour Paths around intervening obstacles (only for backward loops or opposing vertical flows)
    if tx <= sx + 5.0 or (s_port_name == "bottom" and t_port_name == "top" and ty < sy) or (s_port_name == "top" and t_port_name == "bottom" and ty > sy):
        max_obs_y = max([obs.y + obs.height for obs in obstacles]) if obstacles else max(sy, ty)
        route_y_bot = round(max(sy, ty, max_obs_y) + 30.0, 1)
        if s_port_name in ("bottom", "right") and t_port_name in ("bottom", "left"):
            valid_s = (s_port_name == "bottom" and route_y_bot >= sy + 10.0) or (s_port_name == "right")
            valid_t = (t_port_name == "bottom" and route_y_bot >= ty + 10.0) or (t_port_name == "left")
            if valid_s and valid_t:
                p = [
                    Waypoint(sx, sy),
                    Waypoint(sx, route_y_bot),
                    Waypoint(tx, route_y_bot),
                    Waypoint(tx, ty),
                ]
                if not _path_collides(p, obstacles):
                    return _clean_waypoints(p)

        min_obs_y = min([obs.y for obs in obstacles]) if obstacles else min(sy, ty)
        route_y_top = round(min(sy, ty, min_obs_y) - 30.0, 1)
        if route_y_top >= 30.0 and s_port_name in ("top", "right") and t_port_name in ("top", "left"):
            valid_s = (s_port_name == "top" and route_y_top <= sy - 10.0) or (s_port_name == "right")
            valid_t = (t_port_name == "top" and route_y_top <= ty - 10.0) or (t_port_name == "left")
            if valid_s and valid_t:
                p = [
                    Waypoint(sx, sy),
                    Waypoint(sx, route_y_top),
                    Waypoint(tx, route_y_top),
                    Waypoint(tx, ty),
                ]
                if not _path_collides(p, obstacles):
                    return _clean_waypoints(p)

    # 5. Discrete Grid Search (A*) for arbitrary multi-pool / multi-lane obstacle environments
    x_coords = sorted(set([sx, tx, round((sx + tx) / 2.0, 1)] +
                         [round(obs.x - 30.0, 1) for obs in obstacles] +
                         [round(obs.x + obs.width + 30.0, 1) for obs in obstacles] +
                         [round(obs.x + obs.width / 2.0, 1) for obs in obstacles]))
    y_coords = sorted(set([sy, ty, round((sy + ty) / 2.0, 1)] +
                         [round(obs.y - 25.0, 1) for obs in obstacles] +
                         [round(obs.y + obs.height + 25.0, 1) for obs in obstacles] +
                         [round(obs.y + obs.height / 2.0, 1) for obs in obstacles]))

    # Filter coordinates
    x_coords = [x for x in x_coords if x >= 30.0]
    y_coords = [y for y in y_coords if y >= 30.0]

    import heapq
    start_state = (sx, sy, None)  # x, y, last_direction
    entry_id = 0
    h_start = abs(sx - tx) + abs(sy - ty)
    queue = [(h_start, 0.0, 0, entry_id, start_state, [Waypoint(sx, sy)])]
    visited = {}

    while queue:
        f_score, g_score, bends, _, (cx, cy, last_dir), path = heapq.heappop(queue)

        if abs(cx - tx) < 1e-3 and abs(cy - ty) < 1e-3:
            # Check target port entry direction constraint
            if len(path) >= 2:
                prev = path[-2]
                if t_port_name == "top" and not (abs(prev.x - tx) < 1e-3 and prev.y <= ty - 2.0):
                    continue
                if t_port_name == "bottom" and not (abs(prev.x - tx) < 1e-3 and prev.y >= ty + 2.0):
                    continue
                if t_port_name == "left" and not (abs(prev.y - ty) < 1e-3 and prev.x <= tx - 2.0):
                    continue
                if t_port_name == "right" and not (abs(prev.y - ty) < 1e-3 and prev.x >= tx + 2.0):
                    continue
            return _clean_waypoints(path)

        state_key = (cx, cy, last_dir)
        if state_key in visited and visited[state_key] <= g_score:
            continue
        visited[state_key] = g_score

        neighbors = []
        # Horizontal moves
        for nx in x_coords:
            if abs(nx - cx) > 1e-3:
                neighbors.append((nx, cy, "H", abs(nx - cx)))
        # Vertical moves
        for ny in y_coords:
            if abs(ny - cy) > 1e-3:
                neighbors.append((cx, ny, "V", abs(ny - cy)))

        for nx, ny, move_dir, dist in neighbors:
            # Check port exit constraints on first step
            if len(path) == 1:
                if s_port_name == "right" and (move_dir != "H" or nx < sx + BUFFER):
                    continue
                if s_port_name == "left" and (move_dir != "H" or nx > sx - BUFFER):
                    continue
                if s_port_name == "bottom" and (move_dir != "V" or ny < sy + BUFFER):
                    continue
                if s_port_name == "top" and (move_dir != "V" or ny > sy - BUFFER):
                    continue

            # Check collision of this segment
            next_pt = Waypoint(nx, ny)
            if any(_segment_intersects_box(path[-1], next_pt, obs) for obs in obstacles):
                continue

            new_bends = bends + (1 if last_dir is not None and last_dir != move_dir else 0)
            if new_bends > 5:
                continue

            bend_penalty = 120.0 if last_dir is not None and last_dir != move_dir else 0.0
            new_g = g_score + dist + bend_penalty
            heuristic = abs(nx - tx) + abs(ny - ty)
            new_f = new_g + heuristic
            entry_id += 1
            heapq.heappush(
                queue,
                (new_f, new_g, new_bends, entry_id, (nx, ny, move_dir), path + [next_pt]),
            )

    return None


def _route_sequence_flow(
    source_node: FlowNode,
    target_node: FlowNode,
    obstacles: List[Bounds],
    used_incoming_ports: set,
    used_outgoing_ports: set,
) -> List[Waypoint]:
    """Compute optimal collision-free orthogonal waypoints between source and target nodes,
    respecting BPMN gateway port semantics and incoming/outgoing port occupancy.
    """
    sb = source_node.bounds
    tb = target_node.bounds
    s_ports = _get_node_ports(sb)
    t_ports = _get_node_ports(tb)

    s_cx, s_cy = sb.x + sb.width / 2.0, sb.y + sb.height / 2.0
    t_cx, t_cy = tb.x + tb.width / 2.0, tb.y + tb.height / 2.0

    # 1. Determine preferred order for source exit port
    is_s_gw = "Gateway" in source_node.element_type
    is_t_gw = "Gateway" in target_node.element_type

    if is_s_gw and len(source_node.outgoing) > 1:
        # Splitting Gateway
        if t_cy < s_cy - 15.0:
            s_pref = ["top", "right", "bottom", "left"]
        elif t_cy > s_cy + 15.0:
            s_pref = ["bottom", "right", "top", "left"]
        else:
            s_pref = ["right", "top", "bottom", "left"]
    elif is_s_gw and len(source_node.incoming) > 1:
        # Joining Gateway (outgoing flow MUST prefer right port, avoid input ports)
        if tb.x >= (sb.x + sb.width - 10.0):
            s_pref = ["right", "top", "bottom", "left"]
        elif tb.y >= (sb.y + sb.height):
            s_pref = ["right", "bottom", "top", "left"]
        else:
            s_pref = ["right", "bottom", "top", "left"]
    else:
        # Task / Event / Subprocess / 1-to-1 Gateway
        if tb.x >= (sb.x + sb.width - 10.0):
            s_pref = ["right", "top", "bottom", "left"]
        elif tb.y >= (sb.y + sb.height):
            s_pref = ["bottom", "right", "top", "left"]
        else:
            s_pref = ["bottom", "left", "top", "right"]

    # Ports NOT used by incoming flows come first; ports used by incoming flows are last resort only!
    s_candidates = [p for p in s_pref if p not in used_incoming_ports] + [
        p for p in s_pref if p in used_incoming_ports
    ]

    # 2. Determine preferred order for target ingress port
    if is_t_gw and len(target_node.incoming) > 1:
        # Joining Gateway
        if s_cy < t_cy - 15.0:
            t_pref = ["top", "left", "bottom", "right"]
        elif s_cy > t_cy + 15.0:
            t_pref = ["bottom", "left", "top", "right"]
        else:
            t_pref = ["left", "top", "bottom", "right"]
    else:
        # Splitting Gateway (1 incoming) or Task/Event: enter via left port first
        if sb.x + sb.width <= tb.x + 10.0:
            t_pref = ["left", "top", "bottom", "right"]
        elif sb.y + sb.height <= tb.y:
            t_pref = ["top", "left", "bottom", "right"]
        elif sb.y >= tb.y + tb.height:
            t_pref = ["bottom", "left", "top", "right"]
        else:
            t_pref = ["bottom", "top", "left", "right"]

    t_candidates = [p for p in t_pref if p not in used_outgoing_ports] + [
        p for p in t_pref if p in used_outgoing_ports
    ]

    # Try port pairs in preference order to find collision-free path
    for s_port in s_candidates:
        s_pt = s_ports[s_port]
        for t_port in t_candidates:
            t_pt = t_ports[t_port]
            path = _find_collision_free_path(s_port, s_pt, t_port, t_pt, obstacles)
            if path is not None:
                return path

    # Fallback to direct routing if all search failed
    return _route_orthogonal_edge(
        sb, tb, is_source_gateway=is_s_gw, is_target_gateway=is_t_gw
    )


def _route_orthogonal_edge(
    source_bounds: Bounds,
    target_bounds: Bounds,
    is_source_gateway: bool = False,
    is_target_gateway: bool = False,
) -> List[Waypoint]:
    """Compute orthogonal waypoints between source and target bounding boxes (heuristic baseline)."""
    s_cx = round(source_bounds.x + source_bounds.width / 2.0, 1)
    s_cy = round(source_bounds.y + source_bounds.height / 2.0, 1)
    s_right = round(source_bounds.x + source_bounds.width, 1)
    s_bottom = round(source_bounds.y + source_bounds.height, 1)
    s_top = round(source_bounds.y, 1)
    s_left = round(source_bounds.x, 1)

    t_cx = round(target_bounds.x + target_bounds.width / 2.0, 1)
    t_cy = round(target_bounds.y + target_bounds.height / 2.0, 1)
    t_left = round(target_bounds.x, 1)
    t_right = round(target_bounds.x + target_bounds.width, 1)
    t_top = round(target_bounds.y, 1)
    t_bottom = round(target_bounds.y + target_bounds.height, 1)

    # Case 1: Target is strictly to the right
    if t_left >= s_right:
        if abs(s_cy - t_cy) <= 8.0:
            return [
                Waypoint(x=s_right, y=s_cy),
                Waypoint(x=t_left, y=s_cy),
            ]

        if is_source_gateway:
            if t_cy < s_cy - 15.0:
                return [
                    Waypoint(x=s_cx, y=s_top),
                    Waypoint(x=s_cx, y=t_cy),
                    Waypoint(x=t_left, y=t_cy),
                ]
            elif t_cy > s_cy + 15.0:
                return [
                    Waypoint(x=s_cx, y=s_bottom),
                    Waypoint(x=s_cx, y=t_cy),
                    Waypoint(x=t_left, y=t_cy),
                ]

        if is_target_gateway:
            if s_cy < t_cy - 15.0:
                return [
                    Waypoint(x=s_right, y=s_cy),
                    Waypoint(x=t_cx, y=s_cy),
                    Waypoint(x=t_cx, y=t_top),
                ]
            elif s_cy > t_cy + 15.0:
                return [
                    Waypoint(x=s_right, y=s_cy),
                    Waypoint(x=t_cx, y=s_cy),
                    Waypoint(x=t_cx, y=t_bottom),
                ]

        mid_x = round((s_right + t_left) / 2.0, 1)
        return [
            Waypoint(x=s_right, y=s_cy),
            Waypoint(x=mid_x, y=s_cy),
            Waypoint(x=mid_x, y=t_cy),
            Waypoint(x=t_left, y=t_cy),
        ]

    # Case 2: Target is below or backwards
    if t_top >= s_bottom:
        if abs(s_cx - t_cx) <= 15.0:
            return [
                Waypoint(x=s_cx, y=s_bottom),
                Waypoint(x=s_cx, y=t_top),
            ]
        mid_y = round((s_bottom + t_top) / 2.0, 1)
        return [
            Waypoint(x=s_cx, y=s_bottom),
            Waypoint(x=s_cx, y=mid_y),
            Waypoint(x=t_cx, y=mid_y),
            Waypoint(x=t_cx, y=t_top),
        ]

    # Case 3: General backward / loop flow
    route_y = round(max(s_bottom, t_bottom) + 30.0, 1)
    return [
        Waypoint(x=s_cx, y=s_bottom),
        Waypoint(x=s_cx, y=route_y),
        Waypoint(x=t_cx, y=route_y),
        Waypoint(x=t_cx, y=t_bottom),
    ]


def _get_element_ports(
    bounds: Bounds,
    kind: str,
    other_bounds: Optional[Bounds] = None,
) -> Dict[str, Waypoint]:
    """Get the 4 standard cardinal ports for a node or participant pool.
    For participants / black-box pools, ports align dynamically with the other element's center.
    """
    cx = round(bounds.x + bounds.width / 2.0, 1)
    cy = round(bounds.y + bounds.height / 2.0, 1)
    left_x = round(bounds.x, 1)
    right_x = round(bounds.x + bounds.width, 1)
    top_y = round(bounds.y, 1)
    bottom_y = round(bounds.y + bounds.height, 1)

    if kind in ("blackbox_participant", "process_participant") and other_bounds is not None:
        other_cx = round(other_bounds.x + other_bounds.width / 2.0, 1)
        other_cy = round(other_bounds.y + other_bounds.height / 2.0, 1)
        clamped_x = round(max(bounds.x + 30.0, min(bounds.x + bounds.width - 30.0, other_cx)), 1)
        clamped_y = round(max(bounds.y + 20.0, min(bounds.y + bounds.height - 20.0, other_cy)), 1)
        return {
            "left": Waypoint(x=left_x, y=clamped_y),
            "right": Waypoint(x=right_x, y=clamped_y),
            "top": Waypoint(x=clamped_x, y=top_y),
            "bottom": Waypoint(x=clamped_x, y=bottom_y),
        }

    return {
        "left": Waypoint(x=left_x, y=cy),
        "right": Waypoint(x=right_x, y=cy),
        "top": Waypoint(x=cx, y=top_y),
        "bottom": Waypoint(x=cx, y=bottom_y),
    }


def _route_message_flow(
    source_bounds: Bounds,
    target_bounds: Bounds,
    source_kind: str,
    target_kind: str,
    obstacles: Optional[List[Bounds]] = None,
) -> List[Waypoint]:
    """Compute clean orthogonal waypoints for message flows connecting tasks, subprocesses,
    events, and other participants (black-box pools or tasks in other pools) with full
    S-curve routing and obstacle avoidance identical to sequence flows.
    """
    if obstacles is None:
        obstacles = []

    s_ports = _get_element_ports(source_bounds, source_kind, other_bounds=target_bounds)
    t_ports = _get_element_ports(target_bounds, target_kind, other_bounds=source_bounds)

    s_cx = source_bounds.x + source_bounds.width / 2.0
    s_cy = source_bounds.y + source_bounds.height / 2.0
    t_cx = target_bounds.x + target_bounds.width / 2.0
    t_cy = target_bounds.y + target_bounds.height / 2.0

    # Determine port preferences based on relative spatial position
    if t_cy >= (s_cy + 25.0):
        # Target is below Source
        s_pref = ["bottom", "right", "left", "top"]
        t_pref = ["top", "left", "right", "bottom"]
    elif t_cy <= (s_cy - 25.0):
        # Target is above Source
        s_pref = ["top", "right", "left", "bottom"]
        t_pref = ["bottom", "left", "right", "top"]
    elif t_cx >= (s_cx + 25.0):
        # Target is strictly to the right
        s_pref = ["right", "bottom", "top", "left"]
        t_pref = ["left", "top", "bottom", "right"]
    else:
        # Target is strictly to the left
        s_pref = ["left", "bottom", "top", "right"]
        t_pref = ["right", "top", "bottom", "left"]

    # Try port pairs in preference order to find collision-free path with S-curves / corridors
    for s_port in s_pref:
        s_pt = s_ports[s_port]
        for t_port in t_pref:
            t_pt = t_ports[t_port]
            path = _find_collision_free_path(s_port, s_pt, t_port, t_pt, obstacles)
            if path is not None:
                return path

    # Fallback to direct routing if search failed
    return _route_orthogonal_edge(source_bounds, target_bounds)


def _route_association(
    source_bounds: Bounds,
    target_bounds: Bounds,
) -> List[Waypoint]:
    """Compute waypoints for text annotation associations connecting closest edges."""
    s_cx = round(source_bounds.x + source_bounds.width / 2.0, 1)
    s_cy = round(source_bounds.y + source_bounds.height / 2.0, 1)
    t_cx = round(target_bounds.x + target_bounds.width / 2.0, 1)
    t_cy = round(target_bounds.y + target_bounds.height / 2.0, 1)

    # If annotation is above source
    if (target_bounds.y + target_bounds.height) <= (source_bounds.y + 5.0):
        return [
            Waypoint(x=s_cx, y=source_bounds.y),
            Waypoint(x=t_cx, y=round(target_bounds.y + target_bounds.height, 1)),
        ]

    # If annotation is below source
    if target_bounds.y >= (source_bounds.y + source_bounds.height - 5.0):
        return [
            Waypoint(x=s_cx, y=round(source_bounds.y + source_bounds.height, 1)),
            Waypoint(x=t_cx, y=target_bounds.y),
        ]

    # If annotation is to the right
    if target_bounds.x >= (source_bounds.x + source_bounds.width - 5.0):
        return [
            Waypoint(x=round(source_bounds.x + source_bounds.width, 1), y=s_cy),
            Waypoint(x=target_bounds.x, y=t_cy),
        ]

    # If annotation is to the left
    if (target_bounds.x + target_bounds.width) <= (source_bounds.x + 5.0):
        return [
            Waypoint(x=source_bounds.x, y=s_cy),
            Waypoint(x=round(target_bounds.x + target_bounds.width, 1), y=t_cy),
        ]

    return [
        Waypoint(x=s_cx, y=s_cy),
        Waypoint(x=t_cx, y=t_cy),
    ]


def _get_participant_for_ref(model: BPMNModel, elem_id: str) -> Optional[Participant]:
    """Resolve a node or participant reference to its enclosing Participant."""
    if not model.collaboration:
        return None
    part = model.get_participant(elem_id)
    if part:
        return part
    proc = model.get_process_for_node(elem_id)
    if proc:
        return model.get_participant_for_process(proc.id)
    return None


def _get_lane_index_for_node(proc: Process, node_id: str) -> Optional[int]:
    """Get 0-based index of the lane containing node_id."""
    for i, lane in enumerate(proc.lanes):
        if node_id in lane.flow_node_refs:
            return i
    return None


def _determine_optimal_pool_order(model: BPMNModel) -> List[Participant]:
    """Dynamically determine optimal vertical placement order of pools and participants
    to minimize message flow wire length, prevent crossings, and align lanes.
    """
    if not model.collaboration or len(model.collaboration.participants) <= 1:
        return list(model.collaboration.participants) if model.collaboration else []

    participants = list(model.collaboration.participants)
    if len(participants) > 8:
        return participants

    import itertools
    best_perm = participants
    best_cost = float("inf")

    orig_indices = {p.id: i for i, p in enumerate(participants)}

    for perm in itertools.permutations(participants):
        pos_map = {p.id: i for i, p in enumerate(perm)}
        cost = 0.0

        for mf in model.collaboration.message_flows:
            s_part = _get_participant_for_ref(model, mf.source_ref)
            t_part = _get_participant_for_ref(model, mf.target_ref)
            if s_part and t_part and s_part.id in pos_map and t_part.id in pos_map:
                s_idx = pos_map[s_part.id]
                t_idx = pos_map[t_part.id]
                dist = abs(s_idx - t_idx)

                # Heavily penalize non-adjacent pools
                cost += 1000.0 * max(0, dist - 1) ** 2 + 100.0 * dist

                # Small penalty for backward flow
                if s_idx > t_idx:
                    cost += 15.0

                # Lane alignment scoring
                s_node = model.get_flow_node(mf.source_ref)
                if s_node and s_part.process_ref:
                    s_proc = model.get_process(s_part.process_ref)
                    if s_proc and s_proc.lanes:
                        lane_idx = _get_lane_index_for_node(s_proc, s_node.id)
                        num_lanes = len(s_proc.lanes)
                        if lane_idx is not None and num_lanes > 1:
                            if t_idx < s_idx:
                                cost += 30.0 * lane_idx
                            else:
                                cost += 30.0 * (num_lanes - 1 - lane_idx)

                t_node = model.get_flow_node(mf.target_ref)
                if t_node and t_part.process_ref:
                    t_proc = model.get_process(t_part.process_ref)
                    if t_proc and t_proc.lanes:
                        lane_idx = _get_lane_index_for_node(t_proc, t_node.id)
                        num_lanes = len(t_proc.lanes)
                        if lane_idx is not None and num_lanes > 1:
                            if s_idx < t_idx:
                                cost += 30.0 * lane_idx
                            else:
                                cost += 30.0 * (num_lanes - 1 - lane_idx)

        # Structure / Start ordering: if all participants have processes (no black boxes),
        # apply order_lanes_and_pools logic: pool with smaller min_x should be placed above pool with larger min_x
        has_black_box = any(p.is_black_box for p in participants)
        if not has_black_box:
            part_min_x = {}
            for part in participants:
                if part.process_ref:
                    pr = model.get_process(part.process_ref)
                    if pr and pr.flow_nodes:
                        part_min_x[part.id] = min(n.bounds.x for n in pr.flow_nodes.values())

            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    p_i, p_j = perm[i], perm[j]
                    if p_i.id in part_min_x and p_j.id in part_min_x:
                        if part_min_x[p_i.id] > part_min_x[p_j.id]:
                            cost += 50.0

        # Tie-breaker: preserve original order
        inversions = sum(
            1 for i in range(len(perm)) for j in range(i + 1, len(perm))
            if orig_indices[perm[i].id] > orig_indices[perm[j].id]
        )
        cost += inversions * 0.1

        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    return best_perm


def _layout_node_labels(model: BPMNModel) -> None:
    """Dynamically layout external labels for Events and Gateways after all nodes and edges
    are positioned and routed. Evaluates candidate positions (bottom, top, right, left,
    and diagonal corners) to avoid overlap with incoming/outgoing flows and other elements.
    """
    # 1. Collect all edge segments across sequence flows, message flows, and associations
    all_edges: List[FlowEdge] = []
    for proc in model.processes:
        all_edges.extend(proc.sequence_flows)
    if model.collaboration:
        all_edges.extend(model.collaboration.message_flows)
    all_edges.extend(model.get_all_associations())

    edge_segments: List[Tuple[Waypoint, Waypoint]] = []
    for edge in all_edges:
        if edge.waypoints and len(edge.waypoints) >= 2:
            for i in range(len(edge.waypoints) - 1):
                edge_segments.append((edge.waypoints[i], edge.waypoints[i + 1]))

    # 2. Collect base obstacles (all flow nodes, text annotations, black-box participants)
    base_obstacles: List[Bounds] = [
        n.bounds
        for proc in model.processes
        for n in proc.flow_nodes.values()
    ]
    base_obstacles.extend(annot.bounds for annot in model.get_all_text_annotations())
    if model.collaboration:
        base_obstacles.extend(
            part.bounds
            for part in model.collaboration.participants
            if part.is_black_box and part.bounds.width > 0
        )

    placed_label_bounds: List[Bounds] = []

    # 3. Identify all events and gateways with non-empty names
    nodes_to_label: List[FlowNode] = [
        n
        for proc in model.processes
        for n in proc.flow_nodes.values()
        if ("Event" in n.element_type or "Gateway" in n.element_type) and n.name and n.name.strip()
    ]
    # Sort deterministically by (x, y)
    nodes_to_label.sort(key=lambda n: (n.bounds.x, n.bounds.y))

    # Mapping node ID to containing lane
    node_to_lane: Dict[str, Lane] = {}
    for proc in model.processes:
        for lane in proc.lanes:
            for nid in lane.flow_node_refs:
                node_to_lane[nid] = lane

    for node in nodes_to_label:
        nb = node.bounds
        x, y, w, h = nb.x, nb.y, nb.width, nb.height
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Find connected edges and identify occupied cardinal ports
        occupied_ports: set[str] = set()
        for edge in all_edges:
            if not edge.waypoints or len(edge.waypoints) < 2:
                continue

            if edge.source_ref == node.id:
                p0 = edge.waypoints[0]
                p1 = edge.waypoints[1]
                d_top = abs(p0.y - y)
                d_bottom = abs(p0.y - (y + h))
                d_left = abs(p0.x - x)
                d_right = abs(p0.x - (x + w))
                min_d = min(d_top, d_bottom, d_left, d_right)
                if min_d <= 6.0:
                    if d_top == min_d:
                        occupied_ports.add("top")
                    elif d_bottom == min_d:
                        occupied_ports.add("bottom")
                    elif d_left == min_d:
                        occupied_ports.add("left")
                    elif d_right == min_d:
                        occupied_ports.add("right")
                else:
                    dx = p1.x - p0.x
                    dy = p1.y - p0.y
                    if abs(dy) >= abs(dx):
                        occupied_ports.add("bottom" if dy > 0 else "top")
                    else:
                        occupied_ports.add("right" if dx > 0 else "left")

            if edge.target_ref == node.id:
                p_end = edge.waypoints[-1]
                p_pen = edge.waypoints[-2]
                d_top = abs(p_end.y - y)
                d_bottom = abs(p_end.y - (y + h))
                d_left = abs(p_end.x - x)
                d_right = abs(p_end.x - (x + w))
                min_d = min(d_top, d_bottom, d_left, d_right)
                if min_d <= 6.0:
                    if d_top == min_d:
                        occupied_ports.add("top")
                    elif d_bottom == min_d:
                        occupied_ports.add("bottom")
                    elif d_left == min_d:
                        occupied_ports.add("left")
                    elif d_right == min_d:
                        occupied_ports.add("right")
                else:
                    dx = p_end.x - p_pen.x
                    dy = p_end.y - p_pen.y
                    if abs(dy) >= abs(dx):
                        occupied_ports.add("top" if dy > 0 else "bottom")
                    else:
                        occupied_ports.add("left" if dx > 0 else "right")

        name_text = node.name.strip()
        label_w = round(max(w + 20.0, min(85.0, len(name_text) * 6.5)), 1)
        label_h = 14.0

        candidates: Dict[str, Bounds] = {
            "bottom": Bounds(
                x=round(cx - label_w / 2.0, 1),
                y=round(y + h + LABEL_CARDINAL_GAP, 1),
                width=label_w,
                height=label_h,
            ),
            "top": Bounds(
                x=round(cx - label_w / 2.0, 1),
                y=round(y - LABEL_CARDINAL_GAP - label_h, 1),
                width=label_w,
                height=label_h,
            ),
            "right": Bounds(
                x=round(x + w + LABEL_CARDINAL_GAP, 1),
                y=round(cy - label_h / 2.0, 1),
                width=label_w,
                height=label_h,
            ),
            "left": Bounds(
                x=round(x - LABEL_CARDINAL_GAP - label_w, 1),
                y=round(cy - label_h / 2.0, 1),
                width=label_w,
                height=label_h,
            ),
            "top-right": Bounds(
                x=round(x + w + LABEL_CORNER_MARGIN_X, 1),
                y=round(y - label_h - LABEL_CORNER_MARGIN_Y, 1),
                width=label_w,
                height=label_h,
            ),
            "bottom-right": Bounds(
                x=round(x + w + LABEL_CORNER_MARGIN_X, 1),
                y=round(y + h + LABEL_CORNER_MARGIN_Y, 1),
                width=label_w,
                height=label_h,
            ),
            "top-left": Bounds(
                x=round(x - label_w - LABEL_CORNER_MARGIN_X, 1),
                y=round(y - label_h - LABEL_CORNER_MARGIN_Y, 1),
                width=label_w,
                height=label_h,
            ),
            "bottom-left": Bounds(
                x=round(x - label_w - LABEL_CORNER_MARGIN_X, 1),
                y=round(y + h + LABEL_CORNER_MARGIN_Y, 1),
                width=label_w,
                height=label_h,
            ),
        }

        is_event = "Event" in node.element_type
        base_preference = {
            "bottom": 0.0,
            "top": 10.0 if is_event else 5.0,
            "right": 20.0 if is_event else 10.0,
            "left": 30.0 if is_event else 15.0,
            "top-right": 100.0,
            "bottom-right": 105.0,
            "top-left": 110.0,
            "bottom-left": 115.0,
        }

        lane = node_to_lane.get(node.id)

        candidate_scores: Dict[str, float] = {}
        for d_name, cand in candidates.items():
            score = base_preference[d_name]

            # A. Direct connected port occupation check
            if d_name in occupied_ports:
                score += 100000.0
            if d_name == "top-right" and ("top" in occupied_ports or "right" in occupied_ports):
                score += 40000.0
            if d_name == "bottom-right" and ("bottom" in occupied_ports or "right" in occupied_ports):
                score += 40000.0
            if d_name == "top-left" and ("top" in occupied_ports or "left" in occupied_ports):
                score += 40000.0
            if d_name == "bottom-left" and ("bottom" in occupied_ports or "left" in occupied_ports):
                score += 40000.0

            # B. Edge segment collision check
            for p1, p2 in edge_segments:
                if _segment_intersects_box(p1, p2, cand, margin=2.0):
                    score += 50000.0

            # C. Obstacle shape collision check
            for obs in base_obstacles:
                if obs is not node.bounds and _check_box_overlap(cand, obs, margin=2.0):
                    score += 20000.0

            # D. Placed label collision check
            for obs in placed_label_bounds:
                if _check_box_overlap(cand, obs, margin=2.0):
                    score += 20000.0

            # E. Canvas and lane boundary checks
            if cand.x < 15.0 or cand.y < 15.0:
                score += 30000.0
            if lane and lane.bounds:
                if cand.x < lane.bounds.x + 35.0:
                    score += 15000.0
                if cand.y < lane.bounds.y + 5.0:
                    score += 10000.0
                if (cand.y + cand.height) > (lane.bounds.y + lane.bounds.height - 5.0):
                    score += 10000.0

            candidate_scores[d_name] = score

        best_dir = min(candidates.keys(), key=lambda d: candidate_scores[d])
        node.label_bounds = candidates[best_dir]
        placed_label_bounds.append(node.label_bounds)


def _align_horizontal_flow_nodes(model: BPMNModel, max_diff: float = 30.0) -> None:
    """Align horizontally connected flow nodes in each process/lane to the exact same vertical centerline,
    eliminating tiny kinks and non-straight arcs along horizontal paths.
    """
    for proc in model.processes:
        if not proc.flow_nodes:
            continue

        node_lane = {}
        if proc.lanes:
            for lane in proc.lanes:
                for nid in lane.flow_node_refs:
                    node_lane[nid] = lane.id
        else:
            for nid in proc.flow_nodes:
                node_lane[nid] = "default"

        tasks, gateways, events = [], [], []
        for n in proc.flow_nodes.values():
            el_type = n.element_type.lower()
            if "task" in el_type or "subprocess" in el_type or "activity" in el_type:
                tasks.append(n)
            elif "gateway" in el_type:
                gateways.append(n)
            else:
                events.append(n)

        successors = {nid: [] for nid in proc.flow_nodes}
        predecessors = {nid: [] for nid in proc.flow_nodes}
        for sf in proc.sequence_flows:
            if sf.source_ref in proc.flow_nodes and sf.target_ref in proc.flow_nodes:
                successors[sf.source_ref].append(sf.target_ref)
                predecessors[sf.target_ref].append(sf.source_ref)

        aligned = set()
        tasks_sorted = sorted(tasks, key=lambda t: t.bounds.x)
        for t in tasks_sorted:
            aligned.add(t.id)

        def try_align(anchor_node: FlowNode, target_node: FlowNode) -> bool:
            if target_node.id in aligned:
                return False
            if node_lane.get(anchor_node.id) != node_lane.get(target_node.id):
                return False
            a_cy = anchor_node.bounds.y + anchor_node.bounds.height / 2.0
            t_cy = target_node.bounds.y + target_node.bounds.height / 2.0
            if abs(a_cy - t_cy) <= max_diff:
                new_y = round(a_cy - target_node.bounds.height / 2.0, 1)
                new_box = Bounds(target_node.bounds.x, new_y, target_node.bounds.width, target_node.bounds.height)
                for other in proc.flow_nodes.values():
                    if other.id != target_node.id and _check_box_overlap(new_box, other.bounds, margin=2.0):
                        return False
                target_node.bounds.y = new_y
                aligned.add(target_node.id)
                return True
            return False

        # Phase 1: Align gateways connected to aligned tasks
        for _ in range(3):
            for gw in gateways:
                if gw.id in aligned:
                    continue
                for pred_id in predecessors[gw.id]:
                    if pred_id in aligned and try_align(proc.flow_nodes[pred_id], gw):
                        break
                if gw.id not in aligned:
                    for succ_id in successors[gw.id]:
                        if succ_id in aligned and try_align(proc.flow_nodes[succ_id], gw):
                            break

        # Phase 2: Align events connected to aligned tasks or gateways
        for _ in range(3):
            for ev in events:
                if ev.id in aligned:
                    continue
                for pred_id in predecessors[ev.id]:
                    if pred_id in aligned and try_align(proc.flow_nodes[pred_id], ev):
                        break
                if ev.id not in aligned:
                    for succ_id in successors[ev.id]:
                        if succ_id in aligned and try_align(proc.flow_nodes[succ_id], ev):
                            break

        # Phase 3: Align remaining connected linear chains
        for _ in range(3):
            for sf in proc.sequence_flows:
                sn = proc.flow_nodes.get(sf.source_ref)
                tn = proc.flow_nodes.get(sf.target_ref)
                if sn and tn:
                    if sn.id in aligned and tn.id not in aligned:
                        try_align(sn, tn)
                    elif tn.id in aligned and sn.id not in aligned:
                        try_align(tn, sn)


def post_process_layout(model: BPMNModel) -> BPMNModel:
    """Post-process node positions, dynamically place pools/participants in optimal order,
    compute pool and lane bounds based strictly on flow nodes, places text annotations
    with collision avoidance (considering element name labels), and generates edge waypoints.
    """
    # 1. Precompute per-process content metrics
    process_info = {}
    max_pool_width = 0.0
    pool_header_w = POOL_HEADER_WIDTH if model.collaboration else 0.0

    all_model_nodes = [n for proc in model.processes for n in proc.flow_nodes.values()]
    if all_model_nodes:
        global_min_x = min(n.bounds.x for n in all_model_nodes)
        global_max_x = max(n.bounds.x + n.bounds.width for n in all_model_nodes)
        global_content_w = global_max_x - global_min_x
        global_lane_w = round(global_content_w + LANE_PADDING_LEFT + LANE_PADDING_RIGHT + 60.0, 1)
        max_pool_width = round(pool_header_w + global_lane_w, 1)
    else:
        global_min_x = 0.0
        global_lane_w = 600.0
        max_pool_width = round(pool_header_w + 600.0, 1)

    for proc in model.processes:
        all_nodes = list(proc.flow_nodes.values())
        if not all_nodes:
            continue
        if model.collaboration:
            use_min_x = global_min_x
            use_lane_w = global_lane_w
        else:
            min_all_x = min(n.bounds.x for n in all_nodes)
            max_all_x = max(n.bounds.x + n.bounds.width for n in all_nodes)
            content_width = max_all_x - min_all_x
            use_lane_w = round(content_width + LANE_PADDING_LEFT + LANE_PADDING_RIGHT + 60.0, 1)
            use_min_x = min_all_x
            max_pool_width = max(max_pool_width, round(pool_header_w + use_lane_w, 1))

        process_info[proc.id] = {
            "min_all_x": use_min_x,
            "lane_width": use_lane_w,
            "all_nodes": all_nodes,
        }

    # Ensure all pools have matching width
    for pinfo in process_info.values():
        pinfo["lane_width"] = round(max_pool_width - pool_header_w, 1)

    current_pool_y = POOL_MARGIN_Y
    pool_x = POOL_MARGIN_X
    lane_x = pool_x + pool_header_w

    # 2. Position Pools & Participants in Optimal Order
    ordered_participants = _determine_optimal_pool_order(model)

    if ordered_participants:
        for part in ordered_participants:
            if part.process_ref:
                proc = model.get_process(part.process_ref)
                if proc and proc.id in process_info:
                    pinfo = process_info[proc.id]
                    min_all_x = pinfo["min_all_x"]
                    lane_width = pinfo["lane_width"]
                    all_nodes = pinfo["all_nodes"]
                    current_lane_y = current_pool_y

                    if proc.lanes:
                        for lane in proc.lanes:
                            lane_nodes = [proc.flow_nodes[nid] for nid in lane.flow_node_refs if nid in proc.flow_nodes]
                            if lane_nodes:
                                min_ly = min(n.bounds.y for n in lane_nodes)
                                max_ly = max(n.bounds.y + n.bounds.height for n in lane_nodes)
                                lane_content_h = max_ly - min_ly
                                lane_h = round(max(130.0, lane_content_h + LANE_PADDING_TOP + LANE_PADDING_BOTTOM), 1)

                                lane.bounds = Bounds(
                                    x=lane_x,
                                    y=current_lane_y,
                                    width=lane_width,
                                    height=lane_h,
                                )

                                for n in lane_nodes:
                                    n.bounds.x = round(lane_x + LANE_PADDING_LEFT + (n.bounds.x - min_all_x), 1)
                                    n.bounds.y = round(current_lane_y + LANE_PADDING_TOP + (n.bounds.y - min_ly), 1)

                                current_lane_y += lane_h
                            else:
                                lane.bounds = Bounds(x=lane_x, y=current_lane_y, width=lane_width, height=120.0)
                                current_lane_y += 120.0
                    else:
                        min_py = min(n.bounds.y for n in all_nodes)
                        max_py = max(n.bounds.y + n.bounds.height for n in all_nodes)
                        proc_h = round(max_py - min_py + LANE_PADDING_TOP + LANE_PADDING_BOTTOM, 1)
                        base_x = lane_x + LANE_PADDING_LEFT
                        base_y = current_pool_y + LANE_PADDING_TOP

                        for n in all_nodes:
                            n.bounds.x = round(base_x + (n.bounds.x - min_all_x), 1)
                            n.bounds.y = round(base_y + (n.bounds.y - min_py), 1)

                        current_lane_y += proc_h

                    total_pool_h = round(current_lane_y - current_pool_y, 1)
                    part.bounds = Bounds(
                        x=pool_x,
                        y=current_pool_y,
                        width=max_pool_width,
                        height=total_pool_h,
                    )
                    current_pool_y = current_lane_y + 40.0
            elif part.is_black_box:
                bb_dim = ELEMENT_BOUNDS.get("blackBoxParticipant", {"width": 600, "height": 60})
                connected_x_coords = []
                for mf in model.collaboration.message_flows:
                    other_id = None
                    if mf.source_ref == part.id:
                        other_id = mf.target_ref
                    elif mf.target_ref == part.id:
                        other_id = mf.source_ref
                    if other_id:
                        node = model.get_flow_node(other_id)
                        if node and node.bounds.width > 0:
                            connected_x_coords.append(node.bounds.x + node.bounds.width / 2.0)

                if connected_x_coords:
                    min_cx = min(connected_x_coords)
                    max_cx = max(connected_x_coords)
                    span_w = max_cx - min_cx
                    bb_width = round(max(bb_dim["width"], span_w + 140.0), 1)
                    center_cx = (min_cx + max_cx) / 2.0
                    bb_x = round(max(POOL_MARGIN_X, center_cx - bb_width / 2.0), 1)
                else:
                    bb_width = bb_dim["width"]
                    bb_x = POOL_MARGIN_X + 150.0

                part.bounds = Bounds(
                    x=bb_x,
                    y=round(current_pool_y, 1),
                    width=bb_width,
                    height=bb_dim["height"],
                )
                current_pool_y += bb_dim["height"] + 40.0
    else:
        for proc in model.processes:
            if proc.id in process_info:
                pinfo = process_info[proc.id]
                min_all_x = pinfo["min_all_x"]
                lane_width = pinfo["lane_width"]
                all_nodes = pinfo["all_nodes"]
                current_lane_y = current_pool_y

                if proc.lanes:
                    for lane in proc.lanes:
                        lane_nodes = [proc.flow_nodes[nid] for nid in lane.flow_node_refs if nid in proc.flow_nodes]
                        if lane_nodes:
                            min_ly = min(n.bounds.y for n in lane_nodes)
                            max_ly = max(n.bounds.y + n.bounds.height for n in lane_nodes)
                            lane_content_h = max_ly - min_ly
                            lane_h = round(max(130.0, lane_content_h + LANE_PADDING_TOP + LANE_PADDING_BOTTOM), 1)

                            lane.bounds = Bounds(
                                x=lane_x,
                                y=current_lane_y,
                                width=lane_width,
                                height=lane_h,
                            )

                            for n in lane_nodes:
                                n.bounds.x = round(lane_x + LANE_PADDING_LEFT + (n.bounds.x - min_all_x), 1)
                                n.bounds.y = round(current_lane_y + LANE_PADDING_TOP + (n.bounds.y - min_ly), 1)

                            current_lane_y += lane_h
                        else:
                            lane.bounds = Bounds(x=lane_x, y=current_lane_y, width=lane_width, height=120.0)
                            current_lane_y += 120.0
                else:
                    min_py = min(n.bounds.y for n in all_nodes)
                    max_py = max(n.bounds.y + n.bounds.height for n in all_nodes)
                    proc_h = round(max_py - min_py + LANE_PADDING_TOP + LANE_PADDING_BOTTOM, 1)
                    base_x = lane_x + LANE_PADDING_LEFT
                    base_y = current_pool_y + LANE_PADDING_TOP

                    for n in all_nodes:
                        n.bounds.x = round(base_x + (n.bounds.x - min_all_x), 1)
                        n.bounds.y = round(base_y + (n.bounds.y - min_py), 1)

                    current_lane_y += proc_h

                current_pool_y = current_lane_y + 40.0

    # Align horizontally connected flow nodes to share exact centerline
    _align_horizontal_flow_nodes(model)

    # 3. Collision-free Placement of Text Annotations
    all_annotations = model.get_all_text_annotations()
    all_associations = model.get_all_associations()

    def get_associated_node(annot: TextAnnotation) -> Optional[FlowNode]:
        for assoc in all_associations:
            if assoc.target_ref == annot.id:
                return model.get_flow_node(assoc.source_ref)
            elif assoc.source_ref == annot.id:
                return model.get_flow_node(assoc.target_ref)
        return None

    # Collect obstacles including element name labels
    obstacles: List[Bounds] = [
        _get_node_obstacle_bounds(n)
        for proc in model.processes
        for n in proc.flow_nodes.values()
    ]

    for annot in all_annotations:
        node = get_associated_node(annot)
        aw = annot.bounds.width if annot.bounds.width > 0 else 120.0
        ah = annot.bounds.height if annot.bounds.height > 0 else 40.0
        annot.bounds.width = aw
        annot.bounds.height = ah

        if node:
            nb = node.bounds
            obs_b = _get_node_obstacle_bounds(node)

            candidates = []
            for d in [20.0, 35.0, 50.0, 75.0, 100.0, 130.0, 160.0]:
                # Top (above node)
                if (nb.y - ah - d) >= 20.0:
                    candidates.append((nb.x + (nb.width - aw) / 2.0, nb.y - ah - d))
                # Top-Right
                if (nb.y - ah - d) >= 20.0:
                    candidates.append((obs_b.x + obs_b.width + d, nb.y - ah - d))
                # Bottom (below obstacle including label)
                candidates.append((nb.x + (nb.width - aw) / 2.0, obs_b.y + obs_b.height + d))
                # Bottom-Right
                candidates.append((obs_b.x + obs_b.width + d, obs_b.y + obs_b.height + d))
                # Right (to the right of obstacle)
                candidates.append((obs_b.x + obs_b.width + d, nb.y + (nb.height - ah) / 2.0))
                # Top-Left
                if (obs_b.x - aw - d) >= 20.0 and (nb.y - ah - d) >= 20.0:
                    candidates.append((obs_b.x - aw - d, nb.y - ah - d))
                # Bottom-Left
                if (obs_b.x - aw - d) >= 20.0:
                    candidates.append((obs_b.x - aw - d, obs_b.y + obs_b.height + d))
                # Left (fallback only)
                if (obs_b.x - aw - d) >= 20.0:
                    candidates.append((obs_b.x - aw - d, nb.y + (nb.height - ah) / 2.0))

            chosen = None
            for cx, cy in candidates:
                cand = Bounds(x=round(cx, 1), y=round(cy, 1), width=aw, height=ah)
                if not any(_check_box_overlap(cand, obs) for obs in obstacles):
                    chosen = cand
                    break

            if chosen is None:
                # Fallback to the right with clearance
                chosen = Bounds(
                    x=round(obs_b.x + obs_b.width + 30.0, 1),
                    y=round(nb.y, 1),
                    width=aw,
                    height=ah,
                )

            annot.bounds = chosen
            obstacles.append(chosen)
        else:
            cand = Bounds(x=POOL_MARGIN_X + 200.0, y=50.0, width=aw, height=ah)
            annot.bounds = cand
            obstacles.append(cand)

    # 4. Compute Edge Waypoints
    # Sequence Flows
    for proc in model.processes:
        incoming_ports, outgoing_ports = _determine_node_port_usage(proc)
        proc_node_obstacles = {
            nid: _get_node_obstacle_bounds(n)
            for nid, n in proc.flow_nodes.items()
        }

        for sf in proc.sequence_flows:
            s_node = proc.flow_nodes.get(sf.source_ref)
            t_node = proc.flow_nodes.get(sf.target_ref)
            if s_node and t_node:
                obstacles = [
                    obs for nid, obs in proc_node_obstacles.items()
                    if nid != s_node.id and nid != t_node.id
                ]

                used_in = incoming_ports.get(s_node.id, set())
                used_out = outgoing_ports.get(t_node.id, set())

                sf.waypoints = _route_sequence_flow(
                    s_node,
                    t_node,
                    obstacles,
                    used_incoming_ports=used_in,
                    used_outgoing_ports=used_out,
                )

    # Message Flows
    if model.collaboration:
        all_flow_obstacles = [
            _get_node_obstacle_bounds(n)
            for proc in model.processes
            for n in proc.flow_nodes.values()
        ] + [
            annot.bounds
            for annot in model.get_all_text_annotations()
        ]

        for mf in model.collaboration.message_flows:
            s_bounds, s_kind = model.get_element_bounds_and_kind(mf.source_ref)
            t_bounds, t_kind = model.get_element_bounds_and_kind(mf.target_ref)
            if s_bounds and t_bounds:
                s_node = model.get_flow_node(mf.source_ref)
                t_node = model.get_flow_node(mf.target_ref)
                excluded_ids = set()
                if s_node:
                    excluded_ids.add(s_node.id)
                if t_node:
                    excluded_ids.add(t_node.id)

                mf_obstacles = [
                    _get_node_obstacle_bounds(n)
                    for proc in model.processes
                    for n in proc.flow_nodes.values()
                    if n.id not in excluded_ids
                ] + [
                    annot.bounds
                    for annot in model.get_all_text_annotations()
                    if annot.id not in excluded_ids
                ]

                mf.waypoints = _route_message_flow(
                    s_bounds,
                    t_bounds,
                    source_kind=s_kind,
                    target_kind=t_kind,
                    obstacles=mf_obstacles,
                )

    # Associations (in collaboration and in processes)
    for assoc in all_associations:
        s_bounds = None
        t_bounds = None
        for annot in all_annotations:
            if annot.id == assoc.source_ref:
                s_bounds = annot.bounds
            if annot.id == assoc.target_ref:
                t_bounds = annot.bounds

        if not s_bounds:
            s_node = model.get_flow_node(assoc.source_ref)
            if s_node:
                s_bounds = s_node.bounds
        if not t_bounds:
            t_node = model.get_flow_node(assoc.target_ref)
            if t_node:
                t_bounds = t_node.bounds

        if s_bounds and t_bounds:
            assoc.waypoints = _route_association(s_bounds, t_bounds)

    # 5. Collision-Free Placement of Event and Gateway Labels
    _layout_node_labels(model)

    return model