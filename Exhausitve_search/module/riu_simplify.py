from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.utils import EdgeType, VertexType, toggle_edge, vertex_is_zx
from fractions import Fraction
import pyzx as zx

#Riu さんのやつは、強化学習に特化？？して、match を、node でとるみたい
# 注意点としては、pivot, gf だけ、それぞれの info_dict を rewrite 関数内で

# lcomp は、pyzx と割とちがうかなあ、と思ったけど、うん違うんだけど、うーんん

MatchLcompType = Tuple[VT,Tuple[VT,...]]
def match_lcomp(
    g: BaseGraph,
    vertexf: Optional[Callable[[VT],bool]] = None, 
    num: int = -1, 
    check_edge_types: bool = True,
    allow_interacting_matches: bool = False
    ) -> List[MatchLcompType[VT]]:
    """Finds matches of the local complementation rule.
    
    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
    tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
    are of the correct type (Hadamard edges).
    :param vertexf: An optional filtering function for candidate vertices, should
    return True if a vertex should be considered as a match. Passing None will
    consider all vertices.
    :param allow_interacting_matches: Whether or not to allow matches which overlap,
        hence can not all be applied at once. Defaults to False.
    :rtype: List of vertex that can be applied.
    """
    if vertexf is not None: candidates = set([v for v in g.vertices() if vertexf(v)])
    else: candidates = g.vertex_set()
    
    phases = g.phases()
    types = g.types()
    
    i = 0
    m: List[MatchLcompType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        
        if types[v] != VertexType.Z: continue
        if phases[v] not in (Fraction(1,2), Fraction(3,2)): continue # z node ∧ Pauli phase
        if g.is_ground(v): continue

        if check_edge_types and not (
            all(g.edge_type(e) == EdgeType.HADAMARD for e in g.incident_edges(v))
            ): continue # only hadamard edge 

        vn = list(g.neighbors(v))
        if any(types[n] != VertexType.Z for n in vn): continue
        # neighboring nodes also should be z node.
        
        #m.append((v,tuple(vn)))
        if len(g.neighbors(v)) == 1:  #Phase gadget of pi/2 can not be selected
            continue
        flag = False
        for neigh_pg in g.neighbors(v): #If root node of phase gadget is a neighbor of candidate node, node can not be selected.
            for neigh_pg2 in g.neighbors(neigh_pg):
                if len(g.neighbors(neigh_pg2))==1:
                    flag = True
        if flag:
            continue
        m.append(v)
        i += 1
        
        if allow_interacting_matches: continue
        for n in vn: candidates.discard(n)
    return m

def lcomp(g : BaseGraph, node:int):
    phase = g.phase(node)
    neighbors = list(g.neighbors(node))
    edge_table = dict()
    vertice = []
    vertice.append(node)
    n = len(neighbors)
    if phase.numerator == 1:
        g.scalar.add_phase(Fraction(1, 4))
    else:
        g.scalar.add_phase(Fraction(7, 4))
    g.scalar.add_power((n - 2) * (n - 1) // 2)
    for i in range(n):
        g.add_to_phase(neighbors[i], -phase)
        for j in range(i + 1, n):
            edge_neigh = g.edge(neighbors[i], neighbors[j])  # edge type between neighbours
            he = edge_table.get(edge_neigh, [0, 0])[1]
            edge_table[edge_neigh] = [0, he + 1]

    return (edge_table, vertice, [], True)


# pivot は正直ぜんぜん違う感じ

RewriteOutputType = Tuple[Dict[ET, List[int]], List[VT], List[ET], bool]
MatchPivotType = Tuple[VT, VT, Tuple[VT, ...], Tuple[VT, ...]]

def match_pivot_parallel(
    g : BaseGraph,
    matchf: Optional[Callable[[ET], bool]] = None,
    num: int = -1,
    check_edge_types: bool = True,
    allow_interacting_matches: bool = False,
) -> List[MatchPivotType[VT]]:
    """Finds matches of the pivot rule.

    :param g: An instance of a ZX-graph.
    :param num: Maximal amount of matchings to find. If -1 (the default)
    tries to find as many as possible.
    :param check_edge_types: Whether the method has to check if all the edges involved
    are of the correct type (Hadamard edges).
    :param matchf: An optional filtering function for candidate edge, should
    return True if a edge should considered as a match. Passing None will
    consider all edges.
    :param allow_interacting_matches: Whether or not to allow matches which overlap,
        hence can not all be applied at once. Defaults to False.
    :rtype: List of 4-tuples. See :func:`pivot` for the details.
    """
    if matchf is not None:
        candidates = set([e for e in g.edges() if matchf(e)])
    else:
        candidates = g.edge_set()

    types = g.types()
    phases = g.phases()
    matches_dict = {}
    i = 0
    m: List[MatchPivotType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        if check_edge_types and g.edge_type(e) != EdgeType.HADAMARD:
            continue

        v0, v1 = g.edge_st(e)
        if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z):
            continue
        if any(phases[v] not in (0, 1) for v in (v0, v1)):
            continue
        if g.is_ground(v0) or g.is_ground(v1):
            continue

        invalid_edge = False
        v0n = list(g.neighbors(v0))
        v0b = []
        for n in v0n:
            if types[n] == VertexType.Z and g.edge_type(g.edge(v0, n)) == EdgeType.HADAMARD:
                pass
            elif types[n] == VertexType.BOUNDARY:
                v0b.append(n)
            else:
                invalid_edge = True
                break
        if invalid_edge:
            continue

        v1n = list(g.neighbors(v1))
        v1b = []
        for n in v1n:
            if types[n] == VertexType.Z and g.edge_type(g.edge(v1, n)) == EdgeType.HADAMARD:
                pass
            elif types[n] == VertexType.BOUNDARY:
                v1b.append(n)
            else:
                invalid_edge = True
                break
        if invalid_edge:
            continue
        if len(v0b) + len(v1b) > 1:
            continue

        m.append((v0, v1, tuple(v0b), tuple(v1b)))
        matches_dict[(v0, v1)] = (tuple(v0b), tuple(v1b), 0)
        i += 1

    return matches_dict

def match_pivot_gadget(
    g:BaseGraph, matchf: Optional[Callable[[ET], bool]] = None, num: int = -1, allow_interacting_matches: bool = False
    ) -> List[MatchPivotType[VT]]:
    """Like :func:`match_pivot_parallel`, but except for pairings of
    Pauli vertices, it looks for a pair of an interior Pauli vertex and an
    interior non-Clifford vertex in order to gadgetize the non-Clifford vertex."""

    if matchf is not None:
        candidates = set([e for e in g.edges() if matchf(e)])
    else:
        candidates = g.edge_set()

    types = g.types()
    phases = g.phases()
    matches_dict = {}
    i = 0
    m: List[MatchPivotType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        e = candidates.pop()
        v0, v1 = g.edge_st(e)
        if not all(types[v] == VertexType.Z for v in (v0, v1)):
            continue

        if phases[v0] not in (0, 1):
            if phases[v1] in (0, 1):
                v0, v1 = v1, v0
            else:
                continue
        elif phases[v1] in (0, 1):
            continue  # Now v0 has a Pauli phase and v1 has a non-Pauli phase

        if g.is_ground(v0):
            continue

        v0n = list(g.neighbors(v0))
        v1n = list(g.neighbors(v1))
        if len(v1n) == 1:
            continue  # It is a phase gadget
        if any(types[n] != VertexType.Z for vn in (v0n, v1n) for n in vn):
            continue

        bad_match = False
        edges_to_discard = []
        for i, neighbors in enumerate((v0n, v1n)):
            for n in neighbors:
                if types[n] != VertexType.Z:
                    bad_match = True
                    break
                ne = list(g.incident_edges(n))
                if i == 0 and len(ne) == 1 and not (e == ne[0]):  # v0 is a phase gadget
                    bad_match = True
                    break
                edges_to_discard.extend(ne)
            if bad_match:
                break
        if bad_match:
            continue

        m.append((v0, v1, tuple(), tuple()))
        matches_dict[(v0, v1)] = (tuple(), tuple(), 2)
        i += 1

    return matches_dict

def match_pivot_boundary(
    g:BaseGraph, matchf: Optional[Callable[[VT], bool]] = None, num: int = -1, allow_interacting_matches: bool = False
    ) -> List[MatchPivotType[VT]]:
    """Like :func:`match_pivot_parallel`, but except for pairings of
    Pauli vertices, it looks for a pair of an interior Pauli vertex and a
    boundary non-Pauli vertex in order to gadgetize the non-Pauli vertex."""
    if matchf is not None:
        candidates = set([v for v in g.vertices() if matchf(v)])
    else:
        candidates = g.vertex_set()

    phases = g.phases()
    types = g.types()
    matches_dict = {}
    i = 0
    consumed_vertices: Set[VT] = set()
    m: List[MatchPivotType[VT]] = []
    while (num == -1 or i < num) and len(candidates) > 0:
        v = candidates.pop()
        if types[v] != VertexType.Z or phases[v] not in (0, 1) or g.is_ground(v):
            continue

        good_vert = True
        w = None
        bound = None
        for n in g.neighbors(v):
            if (
                types[n] != VertexType.Z
                or len(g.neighbors(n)) == 1
                or n in consumed_vertices
                or g.is_ground(n)
            ):
                good_vert = False
                break

            boundaries = []
            wrong_match = False
            for b in g.neighbors(n):
                if types[b] == VertexType.BOUNDARY:
                    boundaries.append(b)
                elif types[b] != VertexType.Z:
                    wrong_match = True
            if len(boundaries) != 1 or wrong_match:
                continue  # n is not on the boundary or has too many boundaries or has neighbors of wrong type
            if phases[n] and hasattr(phases[n], "denominator") and phases[n].denominator == 2:
                w = n
                bound = boundaries[0]
            if not w:
                w = n
                bound = boundaries[0]
        if not good_vert or w is None:
            continue
        assert bound is not None

        m.append((v, w, tuple(), tuple([bound])))
        matches_dict[(v, w)] = (tuple(), tuple([bound]), 1)
        i += 1
    return matches_dict

# pivot_info_dict = match_pivot_parallel() | match_pivot_boundary() | match_pivot_gadget()

def create_pivot_info_dict(g:BaseGraph): # 意図としては、gdget をかけようという感じか？
    info_dict = match_pivot_parallel(g) | match_pivot_boundary(g) | match_pivot_gadget(g)
    return info_dict

# とりあえず、pivot_info_dict は、引数として入れてあげることにする

def pivot(g:BaseGraph, v0, v1, pivot_info_dict:dict) -> RewriteOutputType[ET, VT]:
    """Perform a pivoting rewrite, given a list of matches as returned by
    ``match_pivot(_parallel)``. A match is itself a list where:

    ``m[0]`` : first vertex in pivot.
    ``m[1]`` : second vertex in pivot.
    ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
    ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
    """
    rem_verts: List[VT] = []
    rem_edges: List[ET] = []
    etab: Dict[ET, List[int]] = dict()
    m = [0, 0, 0, 0, 0]

    m[0], m[1] = v0, v1
    m[2], m[3], _ = pivot_info_dict[(v0, v1)]
    phases = g.phases()

    n = [set(g.neighbors(m[0])), set(g.neighbors(m[1]))]
    for i in range(2):
        n[i].remove(m[1 - i])  # type: ignore # Really complex typing situation
        if len(m[i + 2]) == 1:
            n[i].remove(m[i + 2][0])  # type: ignore

    n.append(n[0] & n[1])  #  n[2] <- non-boundary neighbors of m[0] and m[1]
    n[0] = n[0] - n[2]  #  n[0] <- non-boundary neighbors of m[0] only
    n[1] = n[1] - n[2]  #  n[1] <- non-boundary neighbors of m[1] only

    es = (
        [g.edge(s, t) for s in n[0] for t in n[1]]
        + [g.edge(s, t) for s in n[1] for t in n[2]]
        + [g.edge(s, t) for s in n[0] for t in n[2]]
    )
    k0, k1, k2 = len(n[0]), len(n[1]), len(n[2])
    g.scalar.add_power(k0 * k2 + k1 * k2 + k0 * k1)

    for v in n[2]:
        if not g.is_ground(v):
            g.add_to_phase(v, 1)

    if phases[m[0]] and phases[m[1]]:
        g.scalar.add_phase(Fraction(1))
    if not m[2] and not m[3]:
        g.scalar.add_power(-(k0 + k1 + 2 * k2 - 1))
    elif not m[2]:
        g.scalar.add_power(-(k1 + k2))
    else:
        g.scalar.add_power(-(k0 + k2))

    for i in range(2):  # if m[i] has a phase, it will get copied on to the neighbors of m[1-i]:
        a = phases[m[i]]  # type: ignore
        if a:
            for v in n[1 - i]:
                if not g.is_ground(v):
                    g.add_to_phase(v, a)
            for v in n[2]:
                if not g.is_ground(v):
                    g.add_to_phase(v, a)

        if not m[i + 2]:
            rem_verts.append(m[1 - i])  # type: ignore # if there is no boundary, the other vertex is destroyed
        else:
            e = g.edge(m[i], m[i + 2][0])  # type: ignore # if there is a boundary, toggle whether it is an h-edge or a normal edge
            new_e = g.edge(m[1 - i], m[i + 2][0])  # type: ignore # and point it at the other vertex
            ne, nhe = etab.get(new_e, [0, 0])
            if g.edge_type(e) == EdgeType.SIMPLE:
                nhe += 1
            elif g.edge_type(e) == EdgeType.HADAMARD:
                ne += 1
            etab[new_e] = [ne, nhe]
            rem_edges.append(e)

    for e in es:
        nhe = etab.get(e, (0, 0))[1]
        etab[e] = [0, nhe + 1]

    return (etab, rem_verts, rem_edges, True)


def pivot_gadget(g:BaseGraph, v0, v1) -> RewriteOutputType[ET, VT]:
    """Performs the gadgetizations required before applying pivots.
    ``m[0]`` : interior pauli vertex
    ``m[1]`` : interior non-pauli vertex to gadgetize
    ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
    ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
    """
    vertices_to_gadgetize = v1
    gadgetize(g, vertices_to_gadgetize)
    return pivot(v0, v1)

def gadgetize(g:BaseGraph, vertices: VT) -> None:
    """Helper function which pulls out a list of vertices into gadgets"""
    edge_list = []

    inputs = g.inputs()
    phases = g.phases()
    v = vertices
    if any(n in inputs for n in g.neighbors(v)):
        mod = 0.5
    else:
        mod = -0.5

    vp = g.add_vertex(VertexType.Z, -2, g.row(v) + mod, phases[v])
    v0 = g.add_vertex(VertexType.Z, -1, g.row(v) + mod, 0)
    g.set_phase(v, 0)

    edge_list.append(g.edge(v, v0))
    edge_list.append(g.edge(v0, vp))

    if g.phase_tracking:
        g.unfuse_vertex(vp, v)

    g.add_edges(edge_list, EdgeType.HADAMARD)
    return


# Riu さんの id_match は、確認した感じ、pyzx と、関数の引数の入れ方以外差異なし
MatchIdType = Tuple[VT,VT,VT,EdgeType]


def match_ids(g: BaseGraph):
    candidates = g.vertex_set()
    types = g.types()
    phases = g.phases()
    m = []
    while len(candidates) > 0:
        v = candidates.pop()
        if phases[v] != 0 or not zx.utils.vertex_is_zx(types[v]) or g.is_ground(v):
            continue
        neigh = g.neighbors(v)
        if len(neigh) != 2:
            continue
        v0, v1 = neigh
        if (
            g.is_ground(v0)
            and types[v1] == zx.VertexType.BOUNDARY
            or g.is_ground(v1)
            and types[v0] == zx.VertexType.BOUNDARY
        ):
            # Do not put ground spiders on the boundary
            continue
        m.append(v)
    return m


def remove_ids(g:BaseGraph, node):
    neigh = g.neighbors(node)
    v0, v1 = neigh
    if g.edge_type(g.edge(node, v0)) != g.edge_type(
        g.edge(node, v1)
    ):  # exactly one of them is a hadamard edge
        et = zx.EdgeType.HADAMARD
    else:
        et = zx.EdgeType.SIMPLE
    # create dict, rem_vertexs
    etab = dict()
    e = g.edge(v0, v1)
    if not e in etab:
        etab[e] = [0, 0]
    if et == zx.EdgeType.SIMPLE:
        etab[e][0] += 1
    else:
        etab[e][1] += 1
    return (etab, [node], [], False)


"""
from copy import deepcopy
import random
qubit = 5
depth = 20
random.seed(20)
circ = zx.generate.cliffordT(qubit, depth)
zx.to_gh(circ)
zx.spider_simp(circ, quiet=True)
g = deepcopy(circ)

match = match_lcomp(g)
print(match)"""

# Gadget Fusion

MatchGadgetType = Tuple[VT, int, List[VT], Dict[VT, VT]]


def match_phase_gadgets(g: BaseGraph, vertexf: Optional[Callable[[VT], bool]] = None) -> List[MatchGadgetType[VT]]:
    """Determines which phase gadgets act on the same vertices, so that they can be fused together.

    :param g: An instance of a ZX-graph.
    :rtype: List of 4-tuples ``(leaf, parity_length, other axels with same targets, leaf dictionary)``.
    1.leaf is a vertex that represents a phase gadget
    2.parity_length is the number of vertices that the phase gadget acts on
    3.other_axels is a list of other phase gadgets that act on the same vertices as leaf
    4.leaf_dict is a dictionary that maps each phase gadget to its corresponding phase node
    """
    if vertexf is not None:
        candidates = set([v for v in g.vertices() if vertexf(v)])
    else:
        candidates = g.vertex_set()
    gadget_info_dict = {}
    phases = g.phases()

    parities: Dict[FrozenSet[VT], List[VT]] = dict()
    gadgets: Dict[VT, VT] = dict()
    inputs = g.inputs()
    outputs = g.outputs()
    # First we find all the phase-gadgets, and the list of vertices they act on
    for v in candidates:
        non_clifford = phases[v] != 0 and getattr(phases[v], "denominator", 1) > 2
        if isinstance(phases[v], Poly):
            non_clifford = True
        if non_clifford and len(list(g.neighbors(v))) == 1:
            n = list(g.neighbors(v))[0]
            if phases[n] not in (0, 1):
                continue  # Not a real phase gadget (happens for scalar diagrams)
            if n in gadgets:
                continue  # Not a real phase gadget (happens for scalar diagrams)
            if n in inputs or n in outputs:
                continue  # Not a real phase gadget (happens for non-unitary diagrams)
            gadgets[n] = v
            par = frozenset(set(g.neighbors(n)).difference({v}))
            if par in parities:
                parities[par].append(n)
            else:
                parities[par] = [n]

    for par, gad in parities.items():
        if len(gad) == 1:
            n = gad[0]
            if phases[n] != 0:
                continue
        else:
            # n = gad.pop()
            gadget_info_dict[tuple(gad)] = len(par)

    return gadget_info_dict, gadgets

def merge_phase_gadgets(g:BaseGraph, vertexs: Tuple[VT]) -> RewriteOutputType[ET, VT]:
    """v0,v1"""
    """Given the output of :func:``match_phase_gadgets``, removes phase gadgets that act on the same set of targets."""
    rem = []
    phases = g.phases()
    gadget_info_dict, graph_gadgets = match_phase_gadgets(g)
    par_num = gadget_info_dict[vertexs]
    n = vertexs[0]
    gad = list(vertexs[1:])
    gadgets = graph_gadgets  # self.gadgets

    v = gadgets[n]
    if len(gad) == 0:
        if phases[n] != 0:
            g.scalar.add_phase(phases[v])
            if g.phase_tracking:
                g.phase_negate(v)
            phase = -phases[v]
    else:
        phase = sum((1 if phases[w] == 0 else -1) * phases[gadgets[w]] for w in gad + [n]) % 2
        for w in gad + [n]:
            if phases[w] != 0:
                g.scalar.add_phase(phases[gadgets[w]])
                if g.phase_tracking:
                    g.phase_negate(gadgets[w])
        g.scalar.add_power(-((par_num - 1) * len(gad)))
    g.set_phase(v, phase)
    g.set_phase(n, 0)
    othertargets = [gadgets[w] for w in gad]
    rem.extend(gad)
    rem.extend(othertargets)
    for w in othertargets:
        if g.phase_tracking:
            g.fuse_phases(v, w)
        if g.merge_vdata is not None:
            g.merge_vdata(v, w)
    return ({}, rem, [], False)

def apply_rule(g:BaseGraph, edge_table, rem_vert, rem_edge, check_isolated_vertices):
    g.add_edge_table(edge_table)
    g.remove_edges(rem_edge)
    g.remove_vertices(rem_vert)
    if check_isolated_vertices:
        g.remove_isolated_vertices()

def spider_fusion(g:BaseGraph, neighs):
    rem_verts = []
    etab = dict()

    if g.row(neighs[0]) == 0:
        v0, v1 = neighs[1], neighs[0]
    else:
        v0, v1 = neighs[0], neighs[1]
    ground = g.is_ground(v0) or g.is_ground(v1)
    if ground:
        g.set_phase(v0, 0)
        g.set_ground(v0)
    else:
        g.add_to_phase(v0, g.phase(v1))
    if g.phase_tracking:
        g.fuse_phases(v0, v1)
    # always delete the second vertex in the match
    rem_verts.append(v1)
    # edges from the second vertex are transferred to the first
    for w in g.neighbors(v1):
        if v0 == w:
            continue
        e = g.edge(v0, w)
        if e not in etab:
            etab[e] = [0, 0]
        etab[e][g.edge_type(g.edge(v1, w)) - 1] += 1
    return (etab, rem_verts, [], True)
