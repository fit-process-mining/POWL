from pm4py.objects.process_tree.obj import Operator as PTOperator, ProcessTree

from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.base import TaggedPOWL
from powl.objects.tagged_powl.builders import loop, sequence, xor
from powl.objects.tagged_powl.partial_order import PartialOrder


def apply(tree: ProcessTree) -> TaggedPOWL:

    nodes = []

    for c in tree.children:
        nodes.append(apply(c))

    if tree.operator is None:
        if tree.label is not None:
            powl = Activity(label=tree.label)
        else:
            powl = Activity(label=None)
    elif tree.operator == PTOperator.XOR:
        powl = xor(nodes)
    elif tree.operator == PTOperator.LOOP:
        if len(nodes) < 2:
            raise ValueError("Loop nodes are missing")
        powl = loop(nodes[0], nodes[1] if len(nodes) == 2 else xor(nodes[1:]).normalize())
    elif tree.operator == PTOperator.PARALLEL:
        powl = PartialOrder(nodes=nodes)
    elif tree.operator == PTOperator.SEQUENCE:
        powl = sequence(nodes)
    else:
        raise Exception("Unsupported process tree!")

    powl = powl.normalize()

    return powl
