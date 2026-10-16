"""Linking charge of the twisted quantum double: histories, twisted errors, c_0.

Everything here follows Appendix A of arxiv/2604.02033 on the explicit cubic
lattice of this package, with no cohomology left implicit:

- the flux b of one color is the residual X loop configuration of that color
  (one bit per X decoding-graph edge), placed on the faces of a primal
  "history" lattice by a fixed dual-lattice shift (DUAL_SHIFT below);
- a history a is one bit per edge of that lattice (data-qubit values on the
  spatial edges, A-ancilla branches on the vertical ones), and da = b says the
  history is consistent with the flux (sample_history draws one uniformly);
- the twisted errors are M~^b v for random coins v, with M~^b read off Eq. (41)
  cube by cube (TwistGeometry.twisted_matrix);
- M^b = d^T M~^b, K^b spans ker M^b, kappa^b_u = <a, M~^b u> + Q^b(u) with
  Q^b(u) = sum_{i<j} M^b_ij u_i u_j (Eq. (37); Q has no single-colour terms and
  its polarization is M^b), and c_0 is any short Z chain with
  K^b d^T c_0 = kappa^b (linking_charge_errors).

Index conventions (shared with lattice.py): node n = x + y L + t L^2, edge
3 n + k with k = 0, 1, 2 for x, y, t; the X and Z decoding graphs use the same
indexing. The Z graph *is* the primal lattice: Z node n is primal vertex
(x, y, t). The X graph of colour lam is the dual lattice, X node q being the
dual vertex q + (DUAL_SHIFT[lam] - 1/2)(1, 1, 1); the X edge (q; k) therefore
crosses the primal face normal to k with base corner
    q + (DUAL_SHIFT[lam] - 1)(1, 1, 1) + e_k.

DUAL_SHIFT: the legacy twist rules (twisted.generate_twisted_z_errors) pair
each twisted Z edge with the X-loop bit at the same index. Read against
Eq. (41) this pins red to shift 0 and blue to shift 1, but the two rules that
read the *green* flux disagree by one diagonal step: the blue-vertex rule
(red edges leaving node - (1,1,1)) needs shift 1, the red-vertex rule (blue
edges arriving at node + (1,1,1)) needs shift 0. With a single inconsistent
convention M^b is not symmetric and kappa^b is not defined, so the default
here takes green = 1. That keeps the blue-vertex rule and moves the twisted
blue edges of the red-vertex rule by one diagonal step. Both choices are
statistically equivalent (the green X graph is just relabelled); they differ
only at the time boundaries. legacy_twist_rules=True everywhere keeps the old
rules exactly.

Time boundaries. OBC: the history lattice runs over times -1..T (one slice
beyond the Z graph on each side, where the flux of the shift-0 colours can sit),
and it is open in time. The flux vanishes on its first and last slices, so
da = b always has a solution. PBC (the 3+0D protocol, T = L): the history
lattice is the Z graph itself, periodic in time, and the twist rules wrap in
time too. A flux whose time-slice parity is odd (a loop winding around the
periodic time direction) has no history; such a configuration is reported as
`undefined`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix, triu
from scipy.sparse.csgraph import connected_components

from .geometry import DIMENSIONS
from .lattice import build_incidence_matrix

# Colour order of the cup product x^r U x^g U x^b; global coin / edge indices
# are COLOR_INDEX[colour] * (num_nodes or num_edges) + local index.
ORDER: Tuple[str, ...] = ("r", "g", "b")
COLOR_INDEX: Dict[str, int] = {color: index for index, color in enumerate(ORDER)}

# X node q of colour lam <-> dual vertex q + (DUAL_SHIFT[lam] - 1/2)(1,1,1).
DUAL_SHIFT: Dict[str, int] = {"r": 0, "g": 1, "b": 1}

# The six terms of Eq. (41), per cube with base corner c and direction k:
#   (coin colour, coin offset, edge colour, edge base offset, face colour, face base offset)
# offsets are symbolic: "0" = c, "k" = c + e_k, "1" = c + (1,1,1), "1-k" = c + (1,1,1) - e_k.
# Edges run along +k from their base; faces are normal to k at their base corner.
#   A: x^r U b^g U v^b   B: b^r U x^g U v^b   C: x^r U v^g U b^b
#   D: b^r U v^g U x^b   E: v^r U x^g U b^b   F: v^r U b^g U x^b
EQ41_TERMS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("b", "1", "r", "0", "g", "k"),
    ("b", "1", "g", "1-k", "r", "0"),
    ("g", "k", "r", "0", "b", "k"),
    ("g", "1-k", "b", "1-k", "r", "0"),
    ("r", "0", "g", "0", "b", "k"),
    ("r", "0", "b", "1-k", "g", "0"),
)


def _offset(symbol: str, axis: int) -> np.ndarray:
    unit = np.zeros(DIMENSIONS, dtype=np.int64)
    unit[axis] = 1
    ones = np.ones(DIMENSIONS, dtype=np.int64)
    return {"0": 0 * ones, "k": unit, "1": ones, "1-k": ones - unit}[symbol]


@dataclass
class TwistGeometry:
    """Index tables of one (L, T, boundary): twist terms, flux faces, history lattice.

    Built once per lattice (get_geometry caches it) and shared by every shot.
    """

    linear_size: int
    time_depth: int
    boundary: str
    num_nodes: int = field(init=False)
    num_edges: int = field(init=False)
    periodic_time: bool = field(init=False)
    history_depth: int = field(init=False)   # time slices of the history lattice
    history_offset: int = field(init=False)  # history slice of Z time t is t + offset
    incidence3: csr_matrix = field(init=False, repr=False)
    z_incidence: csr_matrix = field(init=False, repr=False)
    # Valid Z (= X) edges of the decoding graph.
    valid_edge: np.ndarray = field(init=False, repr=False)
    # Z edge index -> history-lattice edge index (valid edges only).
    z_to_history_edge: np.ndarray = field(init=False, repr=False)
    # Per colour: (X edge indices, history-face indices) of the valid X edges.
    flux_faces: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(init=False, repr=False)
    # Eq. (41) entries: coin (global), twisted edge (global), face colour index, face index.
    term_coin: np.ndarray = field(init=False, repr=False)
    term_edge: np.ndarray = field(init=False, repr=False)
    term_face_color: np.ndarray = field(init=False, repr=False)
    term_face: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        L, T = self.linear_size, self.time_depth
        if self.boundary not in ("OBC", "PBC"):
            raise ValueError(f"Unsupported boundary {self.boundary!r}")
        self.num_nodes = L * L * T
        self.num_edges = DIMENSIONS * self.num_nodes
        self.periodic_time = self.boundary == "PBC"
        self.history_depth = T if self.periodic_time else T + 2
        self.history_offset = 0 if self.periodic_time else 1

        incidence = build_incidence_matrix(L, T, self.boundary).tocsr()
        self.z_incidence = incidence
        self.incidence3 = block_diag([incidence] * len(ORDER), format="csr")

        edge = np.arange(self.num_edges)
        axis = edge % DIMENSIONS
        node = edge // DIMENSIONS
        t = node // (L * L)
        self.valid_edge = ~((axis == 2) & (t == T - 1)) if not self.periodic_time else np.ones(self.num_edges, bool)

        x, y = node % L, (node // L) % L
        self.z_to_history_edge = np.where(
            self.valid_edge,
            self._history_index(x, y, t + self.history_offset) * DIMENSIONS + axis,
            -1,
        )

        self.flux_faces = {}
        for color in ORDER:
            shift = DUAL_SHIFT[color] - 1
            valid = np.flatnonzero(self.valid_edge)
            k = axis[valid]
            fx = x[valid] + shift + (k == 0)
            fy = y[valid] + shift + (k == 1)
            ft = t[valid] + shift + (k == 2) + self.history_offset
            self.flux_faces[color] = (valid, self._history_index(fx, fy, ft) * DIMENSIONS + k)

        self._build_terms()

    # -- index helpers ---------------------------------------------------------

    def _history_index(self, x, y, tau) -> np.ndarray:
        L = self.linear_size
        tau = np.asarray(tau)
        if self.periodic_time:
            tau = tau % self.history_depth
        elif np.any((tau < 0) | (tau >= self.history_depth)):
            raise IndexError("history slice out of range")
        return (tau * L + np.asarray(y) % L) * L + np.asarray(x) % L

    def _z_node(self, x, y, t) -> Tuple[np.ndarray, np.ndarray]:
        """Z node index and validity of primal vertex (x, y, t)."""
        L, T = self.linear_size, self.time_depth
        if self.periodic_time:
            valid = np.ones(np.shape(t), dtype=bool)
            t = t % T
        else:
            valid = (t >= 0) & (t < T)
        return (np.where(valid, t, 0) * L + y % L) * L + x % L, valid

    def _build_terms(self) -> None:
        """Expand Eq. (41) over every cube into flat (coin, edge, face) tables."""
        L, T = self.linear_size, self.time_depth
        t_range = np.arange(T) if self.periodic_time else np.arange(-1, T)
        cy, cx, ct = np.meshgrid(np.arange(L), np.arange(L), t_range, indexing="ij")
        cx, cy, ct = cx.ravel(), cy.ravel(), ct.ravel()
        corner = np.stack([cx, cy, ct])

        coins, edges, face_colors, faces = [], [], [], []
        for coin_color, coin_off, edge_color, edge_off, face_color, face_off in EQ41_TERMS:
            for k in range(DIMENSIONS):
                coin_pos = corner + _offset(coin_off, k)[:, None]
                edge_pos = corner + _offset(edge_off, k)[:, None]
                face_pos = corner + _offset(face_off, k)[:, None]

                coin_node, coin_ok = self._z_node(*coin_pos)
                edge_node, edge_ok = self._z_node(*edge_pos)
                edge_index = edge_node * DIMENSIONS + k
                edge_ok &= self.valid_edge[edge_index]

                tau = face_pos[2] + self.history_offset
                if self.periodic_time:
                    face_ok = np.ones_like(coin_ok)
                else:
                    last = self.history_depth - (1 if k == 2 else 2)
                    face_ok = (tau >= 0) & (tau <= last)
                tau = np.where(face_ok, tau, 0)
                face_index = self._history_index(face_pos[0], face_pos[1], tau) * DIMENSIONS + k

                keep = coin_ok & edge_ok & face_ok
                coins.append(COLOR_INDEX[coin_color] * self.num_nodes + coin_node[keep])
                edges.append(COLOR_INDEX[edge_color] * self.num_edges + edge_index[keep])
                face_colors.append(np.full(int(keep.sum()), COLOR_INDEX[face_color], dtype=np.int64))
                faces.append(face_index[keep])

        self.term_coin = np.concatenate(coins)
        self.term_edge = np.concatenate(edges)
        self.term_face_color = np.concatenate(face_colors)
        self.term_face = np.concatenate(faces)

    # -- flux and twisted errors ----------------------------------------------

    @property
    def num_history_faces(self) -> int:
        return DIMENSIONS * self.linear_size**2 * self.history_depth

    def flux_on_faces(self, x_loops: Dict[str, np.ndarray]) -> np.ndarray:
        """(3, num_history_faces) uint8: the flux b of every colour on the history lattice."""
        flux = np.zeros((len(ORDER), self.num_history_faces), dtype=np.uint8)
        for color in ORDER:
            x_edges, faces = self.flux_faces[color]
            flux[COLOR_INDEX[color], faces] = np.asarray(x_loops[color])[x_edges] & 1
        return flux

    def twisted_matrix(self, flux: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """M~^b as a (3 E x 3 N) mod-2 matrix, and the flux colours behind every coin.

        Column (lam, v) holds the twisted Z edges coin (lam, v) flips. The second
        return value is a per-coin bitmask of the flux colours (1 << colour index)
        that triggered any of its twisted edges.
        """
        active = flux[self.term_face_color, self.term_face].astype(bool)
        rows = self.term_edge[active]
        cols = self.term_coin[active]
        matrix = coo_matrix(
            (np.ones(rows.size, dtype=np.int64), (rows, cols)),
            shape=(len(ORDER) * self.num_edges, len(ORDER) * self.num_nodes),
        ).tocsr()
        matrix.data %= 2
        matrix.eliminate_zeros()
        flux_mask = np.zeros(len(ORDER) * self.num_nodes, dtype=np.int64)
        np.bitwise_or.at(flux_mask, cols, 1 << self.term_face_color[active])
        return matrix, flux_mask

    def split(self, vector: np.ndarray, size: int) -> Dict[str, np.ndarray]:
        """Global (3 * size) vector -> {colour: size-vector} (uint8)."""
        return {color: (vector[COLOR_INDEX[color] * size:(COLOR_INDEX[color] + 1) * size] % 2).astype(np.uint8)
                for color in ORDER}

    def join(self, per_color: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(per_color[color], dtype=np.int64) for color in ORDER])


@lru_cache(maxsize=None)
def get_geometry(linear_size: int, time_depth: int, boundary: str) -> TwistGeometry:
    """Cached TwistGeometry of one lattice; read-only and shared by every shot."""
    return TwistGeometry(linear_size, time_depth, boundary)


def twisted_z_errors(
    geometry: TwistGeometry,
    x_loops: Dict[str, np.ndarray],
    coins: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Twisted Z errors M~^b v of Eq. (41): every (colour, vertex) coin flips with prob 1/2.

    `coins` overrides the random coins (length 3 N, 0/1); all ones gives the
    union M~^b 1 used by the Completing-the-Loop heralding. The random coins are
    drawn from numpy's global generator, like the rest of the noise.
    """
    matrix, _ = geometry.twisted_matrix(geometry.flux_on_faces(x_loops))
    if coins is None:
        coins = np.random.randint(0, 2, size=len(ORDER) * geometry.num_nodes)
    return geometry.split(matrix @ np.asarray(coins, dtype=np.int64), geometry.num_edges)


def heralded_twist_links(geometry: TwistGeometry, x_loops: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Every link some coin could twist, M~^b 1 (the new-rule counterpart of is_full=True)."""
    return twisted_z_errors(geometry, x_loops, coins=np.ones(len(ORDER) * geometry.num_nodes, dtype=np.int64))


# -----------------------------------------------------------------------------
# Histories: a with da = b
# -----------------------------------------------------------------------------


def _comb(plaquettes: np.ndarray) -> Optional[np.ndarray]:
    """Qubit values (y, x, k=0/1) on an L x L torus with the given plaquette parities.

    Comb-shaped spanning tree of the faces: in each row the y edge to the right
    of face x takes the prefix parity of the row, which leaves at most one
    defect per row in the last column; the x edges of that column clear those
    with the prefix parity of the row totals. None if the number of violated
    plaquettes is odd (no solution).
    """
    L = plaquettes.shape[0]
    plaquettes = plaquettes.astype(np.int64) & 1
    if plaquettes.sum() % 2:
        return None
    edges = np.zeros((L, L, 2), dtype=np.uint8)
    edges[:, 1:, 1] = np.cumsum(plaquettes, axis=1)[:, :-1] % 2          # y edge at x+1
    rows = plaquettes.sum(axis=1) % 2
    edges[1:, L - 1, 0] = np.cumsum(rows)[:-1] % 2                          # x edge at row y+1
    return edges


def history_coboundary(geometry: TwistGeometry, history: np.ndarray) -> np.ndarray:
    """da on the history lattice, (history faces,) uint8. Used by the checks."""
    L, D = geometry.linear_size, geometry.history_depth
    a = history.reshape(D, L, L, DIMENSIONS).astype(np.int64)
    ax, ay, at = a[..., 0], a[..., 1], a[..., 2]
    faces = np.zeros((D, L, L, DIMENSIONS), dtype=np.int64)
    faces[..., 2] = ax + np.roll(ax, -1, axis=1) + ay + np.roll(ay, -1, axis=2)
    next_ay = np.roll(ay, -1, axis=0)
    next_ax = np.roll(ax, -1, axis=0)
    faces[..., 0] = ay + next_ay + at + np.roll(at, -1, axis=1)   # normal x: over the y edge
    faces[..., 1] = ax + next_ax + at + np.roll(at, -1, axis=2)   # normal y: over the x edge
    if not geometry.periodic_time:
        faces[D - 1, :, :, :2] = 0  # no vertical faces above the last slice
    return (faces % 2).astype(np.uint8).reshape(-1)


def sample_history(
    geometry: TwistGeometry,
    flux_one_color: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """A uniformly random history a with da = b for one colour, or None if none exists.

    a = a_0 + dw + h: a_0 from the comb on the first slice and the temporal
    gauge (no A flips: a qubit changes only through the X on its vertical face),
    w a uniform bit per history vertex, h a uniform element of the
    non-contractible classes (x and y columns at every time; and the vertical
    edges of one round, if time is periodic). rng=None returns a_0 only.
    Returned flat over history edges, (time, y, x, k) order.
    """
    L, D = geometry.linear_size, geometry.history_depth
    b = flux_one_color.reshape(D, L, L, DIMENSIONS).astype(np.int64)
    first = _comb(b[0, :, :, 2])
    if first is None:
        return None
    a = np.zeros((D, L, L, DIMENSIONS), dtype=np.int64)
    # Vertical faces as increments of the edge below them: the face normal to y
    # sits over an x edge, the face normal to x over a y edge.
    step = np.stack([b[..., 1], b[..., 0]], axis=-1)
    a[..., :2] = np.cumsum(np.concatenate([first[None], step[:-1]]), axis=0) % 2
    if geometry.periodic_time:
        # Close the time loop: the last round's A flips must turn slice D-1 into slice 0.
        mismatch = (a[D - 1, :, :, :2] + step[D - 1] + first) % 2
        if mismatch[0, :, 0].sum() % 2 or mismatch[:, 0, 1].sum() % 2:
            return None
        flips = np.zeros((L, L), dtype=np.int64)
        flips[0, 1:] = np.cumsum(mismatch[0, :-1, 0]) % 2
        flips[1:, :] = (flips[0][None, :] + np.cumsum(mismatch[:-1, :, 1], axis=0)) % 2
        a[D - 1, :, :, 2] = flips

    if rng is not None:
        w = rng.integers(0, 2, size=(D, L, L))
        a[..., 0] += w + np.roll(w, -1, axis=2)
        a[..., 1] += w + np.roll(w, -1, axis=1)
        if geometry.periodic_time:
            a[..., 2] += w + np.roll(w, -1, axis=0)
        else:
            a[:-1, :, :, 2] += w[:-1] + w[1:]
        if rng.integers(2):
            a[:, :, 0, 0] += 1
        if rng.integers(2):
            a[:, 0, :, 1] += 1
        if geometry.periodic_time and rng.integers(2):
            a[0, :, :, 2] += 1
    return (a % 2).astype(np.uint8).reshape(-1)


def history_on_z_edges(geometry: TwistGeometry, history: np.ndarray) -> np.ndarray:
    """Restrict a history-lattice 1-chain to the Z decoding-graph edges."""
    out = np.zeros(geometry.num_edges, dtype=np.uint8)
    valid = geometry.z_to_history_edge >= 0
    out[valid] = history[geometry.z_to_history_edge[valid]]
    return out


# -----------------------------------------------------------------------------
# GF(2) linear algebra (dense, bit-packed rows)
# -----------------------------------------------------------------------------


def gf2_rref(matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon form over GF(2): (nonzero rows as 0/1 uint8, pivot columns)."""
    m, n = matrix.shape
    if m == 0 or n == 0:
        return np.zeros((0, n), dtype=np.uint8), []
    packed = np.packbits(matrix.astype(bool), axis=1)
    pivots: List[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        byte, shift = col >> 3, 7 - (col & 7)
        hits = np.flatnonzero((packed[row:, byte] >> shift) & 1)
        if hits.size == 0:
            continue
        pivot = row + int(hits[0])
        if pivot != row:
            packed[[row, pivot]] = packed[[pivot, row]]
        others = np.flatnonzero((packed[:, byte] >> shift) & 1)
        others = others[others != row]
        if others.size:
            packed[others] ^= packed[row]
        pivots.append(col)
        row += 1
    return np.unpackbits(packed[:row], axis=1, count=n), pivots


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Basis of {u : matrix u = 0} over GF(2), one vector per row (uint8)."""
    n = matrix.shape[1]
    reduced, pivots = gf2_rref(matrix)
    free = [col for col in range(n) if col not in set(pivots)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for index, col in enumerate(free):
        basis[index, col] = 1
        basis[index, pivots] = reduced[:, col]
    return basis


def gf2_solve(matrix: np.ndarray, rhs: np.ndarray) -> Optional[np.ndarray]:
    """One solution s of matrix s = rhs over GF(2) (zero off the pivots), or None."""
    m, n = matrix.shape
    augmented = np.concatenate([matrix.astype(np.uint8) & 1, (rhs.astype(np.uint8) & 1)[:, None]], axis=1)
    reduced, pivots = gf2_rref(augmented)
    if n in pivots:
        return None
    solution = np.zeros(n, dtype=np.uint8)
    solution[pivots] = reduced[:, n]
    return solution


# -----------------------------------------------------------------------------
# kappa^b and c_0
# -----------------------------------------------------------------------------


@dataclass
class LinkingInfo:
    """Per-shot bookkeeping of the linking-charge step (summed over shots by callers)."""

    undefined: int = 0            # no history (periodic-time winding) or odd charge parity
    components: int = 0           # coupled coin clusters with flux of >= 2 colours
    max_component: int = 0        # largest such cluster (coins)
    nonzero_kappa: int = 0        # detectors with kappa = 1
    linking_charges: int = 0      # weight of the c_0 syndrome
    asymmetric: int = 0           # entries with M != M^T (OBC last slice only)
    boundary_clusters: int = 0    # clusters on which M is not symmetric
    boundary_fallback: int = 0    # boundary clusters whose charges were dropped (odd total / ill-defined Q)
    kappa_mismatch: int = 0       # detectors whose kappa changed with the history (should be 0)

    def add(self, other: "LinkingInfo") -> None:
        for name in self.__dataclass_fields__:
            if name == "max_component":
                self.max_component = max(self.max_component, other.max_component)
            else:
                setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def linking_syndrome(
    geometry: TwistGeometry,
    x_loops: Dict[str, np.ndarray],
    history_rng: Optional[np.random.Generator] = None,
    check_history: bool = True,
    all_components: bool = False,
) -> Tuple[Optional[np.ndarray], LinkingInfo, Dict[str, object]]:
    """The fixed Z syndrome s (3 N) with <u, s> = kappa^b_u for every u in ker M^b.

    Works cluster by cluster: M^b is block diagonal over the connected clusters
    of coins it couples. kappa is evaluated with a uniformly random history
    (history_rng). A cluster is skipped -- kappa = 0 exactly -- when its twisted
    edges come from the flux of one colour (then Q = 0 on ker M) *and* the linear
    term vanishes there in the temporal gauge a_0, which is local (a_0 is nonzero
    only between an X error and its correction). A single-colour cluster is not
    enough on its own: a red loop threaded by a distant green loop is a red-only
    cluster with kappa = 1. all_components=True computes every cluster.

    check_history compares kappa from the random history with kappa from a_0 on
    every computed cluster and counts the detectors whose value moved (kappa must
    not depend on the history; a nonzero count flags a non-contractible flux).

    Returns (s or None if undefined, info, extras) where extras carries the
    matrices for diagnostics.
    """
    info = LinkingInfo()
    rng = history_rng if history_rng is not None else np.random.default_rng()
    flux = geometry.flux_on_faces(x_loops)
    twisted, flux_mask = geometry.twisted_matrix(flux)
    coins = len(ORDER) * geometry.num_nodes
    syndrome = np.zeros(coins, dtype=np.uint8)
    extras: Dict[str, object] = {"twisted": twisted, "flux": flux}
    if twisted.nnz == 0:
        return syndrome, info, extras

    coupling = (geometry.incidence3 @ twisted).tocsr()
    coupling.data %= 2
    coupling.eliminate_zeros()
    extras["coupling"] = coupling
    info.asymmetric = int((coupling != coupling.T).nnz)

    random_histories, gauge_histories = [], []
    for color in ORDER:
        reference = sample_history(geometry, flux[COLOR_INDEX[color]], None)
        if reference is None:
            info.undefined = 1
            return None, info, extras
        gauge_histories.append(history_on_z_edges(geometry, reference))
        random_histories.append(history_on_z_edges(geometry, sample_history(geometry, flux[COLOR_INDEX[color]], rng)))
    linear = (twisted.T @ np.concatenate(random_histories).astype(np.int64)) % 2
    linear_gauge = (twisted.T @ np.concatenate(gauge_histories).astype(np.int64)) % 2

    touched = np.flatnonzero(np.diff(coupling.indptr) | np.diff((coupling.T.tocsr()).indptr))
    if touched.size == 0:
        return syndrome, info, extras
    graph = coupling[touched][:, touched]
    count, labels = connected_components(graph + graph.T, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    details = []
    boundary_part = np.zeros(coins, dtype=np.uint8)
    for component in range(count):
        members = touched[order[bounds[component]:bounds[component + 1]]]
        colors = int(np.bitwise_or.reduce(flux_mask[members]))
        if not all_components and bin(colors).count("1") < 2 and not linear_gauge[members].any():
            continue
        info.components += 1
        info.max_component = max(info.max_component, members.size)
        block = coupling[members][:, members]
        dense = block.toarray() % 2
        symmetric = not (dense != dense.T).any()
        # A detector u needs M u = 0 (kappa independent of the history) and
        # u^T M = 0 (no twisted error flips it). In the bulk M is symmetric and
        # the two agree; the OBC truncation of the twisted vertical edges above
        # the last slice breaks the symmetry there, and only the intersection is
        # kept on such a cluster.
        kernel = gf2_nullspace(dense if symmetric else np.concatenate([dense, dense.T]))
        if not symmetric:
            info.boundary_clusters += 1
        if kernel.shape[0] == 0:
            details.append({"colors": colors, "size": int(members.size), "kernel": 0, "kappa": 0})
            continue
        basis = kernel.astype(np.int64)
        quadratic = np.asarray((triu(block, k=1).tocsr() @ basis.T).T * basis).sum(axis=1) % 2
        if not symmetric:
            lower = np.asarray((triu(block.T, k=1).tocsr() @ basis.T).T * basis).sum(axis=1) % 2
            if not np.array_equal(quadratic, lower):
                info.boundary_fallback += 1
                details.append({"colors": colors, "size": int(members.size), "kernel": int(kernel.shape[0]),
                                "kappa": 0, "fallback": True})
                continue
        kappa = (basis @ linear[members] + quadratic) % 2
        details.append({"colors": colors, "size": int(members.size), "kernel": int(kernel.shape[0]),
                        "kappa": int(kappa.sum())})
        if all_components:  # the checks re-verify kappa and c_0 against these
            details[-1].update({"members": members, "basis": kernel, "kappa_vector": kappa,
                                "gauge_linear_zero": not linear_gauge[members].any()})
        if check_history:
            info.kappa_mismatch += int(((basis @ (linear[members] ^ linear_gauge[members])) % 2).sum())
        if not kappa.any():
            continue
        local = gf2_solve(kernel, kappa)
        if local is None:  # kernel rows are independent, so this cannot happen
            info.undefined = 1
            return None, info, extras
        # A single cluster may carry an odd charge: two linked loops of colours
        # r and g are two clusters with one b charge each. Only the total must be
        # even per colour, and c_0 pairs the charges across clusters.
        info.nonzero_kappa += int(kappa.sum())
        syndrome[members] ^= local
        if not symmetric:
            boundary_part[members] ^= local

    per_color_parity = [int(syndrome[i * geometry.num_nodes:(i + 1) * geometry.num_nodes].sum() % 2)
                        for i in range(len(ORDER))]
    if any(per_color_parity) and boundary_part.any():
        # The truncated last slice can leave an unpaired charge; drop the
        # boundary clusters' charges rather than the whole shot.
        info.boundary_fallback += 1
        syndrome ^= boundary_part
        per_color_parity = [int(syndrome[i * geometry.num_nodes:(i + 1) * geometry.num_nodes].sum() % 2)
                            for i in range(len(ORDER))]
    if any(per_color_parity):
        info.undefined = 1
        return None, info, extras
    info.linking_charges = int(syndrome.sum())
    extras["components"] = details
    return syndrome, info, extras


def linking_charge_errors(
    geometry: TwistGeometry,
    x_loops: Dict[str, np.ndarray],
    z_matching,
    history_rng: Optional[np.random.Generator] = None,
    check_history: bool = True,
) -> Tuple[Optional[Dict[str, np.ndarray]], LinkingInfo]:
    """c_0 for a residual X configuration: short Z strings with K^b d^T c_0 = kappa^b.

    The syndrome s comes from linking_syndrome(); c_0 is the minimum-weight
    chain with d^T c_0 = s (z_matching: the unweighted Z-graph Matching), which
    keeps it local -- a chain that wrapped the torus would change the logical
    check. Any solution s works: two differ by twisted-error syndromes, which
    the random twisted errors absorb. Returns (None, info) when undefined.
    """
    syndrome, info = linking_syndrome(geometry, x_loops, history_rng, check_history)[:2]
    if syndrome is None:
        return None, info
    per_color = geometry.split(syndrome.astype(np.int64), geometry.num_nodes)
    chains = {}
    for color in ORDER:
        defects = per_color[color]
        if not defects.any():
            chains[color] = np.zeros(geometry.num_edges, dtype=np.uint8)
            continue
        if defects.sum() % 2:
            info.undefined = 1
            return None, info
        chains[color] = (np.asarray(z_matching.decode(defects)) % 2).astype(np.uint8)
    return chains, info
