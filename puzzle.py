"""Puzzle generator."""

import matplotlib.pyplot as plt
from feynman import Diagram, Vertex, RegularBubbleOperator
from typing import Self
import numpy as np
from PIL import Image

CM = 1 / 2.54  # centimeters in inches
FIGSIZE = np.array([29.7 * CM, 21 * CM])
SIZE = 5.0 * CM
TILESIZE = np.array([SIZE, SIZE])

FERMION = 0.1 * SIZE

ARROW_PARAM = {
    "t": (0.5 * SIZE - FERMION) / SIZE,
    "width": 0.1 * SIZE,
    "length": 0.3 * SIZE,
}


def corners(d: Diagram, xy: tuple[float, float]):
    """Corners of single tile."""
    v1 = d.vertex(xy=xy, marker="")
    return (
        v1,
        d.vertex(v1.xy, dx=SIZE, marker=""),
        d.vertex(v1.xy, dy=SIZE, marker=""),
        d.vertex(v1.xy, dx=SIZE, dy=SIZE, marker=""),
        d.vertex(v1.xy, dx=SIZE / 2.0, dy=SIZE / 2.0, marker=""),
    )


def fv(d: Diagram, x1: Vertex, x3: Vertex) -> Vertex:
    """Vertical fermion border."""
    sgn = 1.0 if x1.y < x3.y else -1.0
    w1, w2, w3 = (
        d.vertex(x1.xy, dy=sgn * (SIZE / 2.0 - FERMION), marker=""),
        d.vertex(x1.xy, dy=sgn * (SIZE / 2.0), dx=sgn * FERMION, marker=""),
        d.vertex(x1.xy, dy=sgn * (SIZE / 2.0 + FERMION), marker=""),
    )
    d.line(x1, w1, arrow=False)
    d.line(w1, w2, arrow=False)
    d.line(w2, w3, arrow=False)
    d.line(w3, x3, arrow=False)
    return w2


def fh(d: Diagram, x1: Vertex, x2: Vertex) -> Vertex:
    """Horizontal fermion border."""
    sgn = 1.0 if x1.x < x2.x else -1.0
    w1, w2, w3 = (
        d.vertex(x1.xy, dx=sgn * (SIZE / 2.0 - FERMION), marker=""),
        d.vertex(x1.xy, dx=sgn * (SIZE / 2.0), dy=sgn * FERMION, marker=""),
        d.vertex(x1.xy, dx=sgn * (SIZE / 2.0 + FERMION), marker=""),
    )
    d.line(x1, w1, arrow=False)
    d.line(w1, w2, arrow=False)
    d.line(w2, w3, arrow=False)
    d.line(w3, x2, arrow=False)
    return w2


def bv(d: Diagram, x1: Vertex, x3: Vertex) -> Vertex:
    """Vertical Boson border."""
    sgn = 1.0 if x1.y < x3.y else -1.0
    w2 = d.vertex(x1.xy, dy=sgn * SIZE / 2.0, marker="")
    d.line(x1, x3, arrow=False)
    return w2


def bh(d: Diagram, x1: Vertex, x2: Vertex) -> Vertex:
    """Horizontal boson border."""
    sgn = 1.0 if x1.x < x2.x else -1.0
    w2 = d.vertex(x1.xy, dx=sgn * SIZE / 2.0, marker="")
    d.line(x1, x2, arrow=False)
    return w2


gv = bv
"""Vertical gluon border."""

gh = bh
"""Horizontal gluon border."""


class TileConfig:
    """Single tile configuration."""

    w: str
    s: str
    e: str
    n: str

    def __init__(self, w: str = "", s: str = "", e: str = "", n: str = "") -> None:
        self.w = w
        self.s = s
        self.e = e
        self.n = n


class Tile:
    """Single tile."""

    d: Diagram
    # border points
    v1: Vertex
    v2: Vertex
    v3: Vertex
    v4: Vertex
    # center
    c: Vertex
    # border reference points
    fi: Vertex | None = None
    fo: Vertex | None = None
    bi: Vertex | None = None
    bo: Vertex | None = None
    gi: Vertex | None = None
    go: Vertex | None = None
    # mapped border reference points
    walls: dict[str, Vertex]

    def __init__(self: Self, d: Diagram, xy) -> None:
        self.d = d
        self.v1, self.v2, self.v3, self.v4, self.c = corners(d, xy)
        self.walls = {}

    def border(self: Self, z: str, x1: Vertex, x2: Vertex, fz, bz, gz) -> Vertex | None:
        """Draw suitable border."""
        v = None
        if z == "":
            self.d.line(x1, x2, arrow=False)
        elif z == "fi":
            v = self.fi = fz(self.d, x2, x1)
        elif z == "fo":
            v = self.fo = fz(self.d, x1, x2)
        elif z == "bi":
            v = self.bi = bz(self.d, x2, x1)
        elif z == "bo":
            v = self.bo = bz(self.d, x1, x2)
        elif z == "gi":
            v = self.gi = gz(self.d, x2, x1)
        elif z == "go":
            v = self.go = gz(self.d, x1, x2)
        return v

    def draw(self: Self, c: TileConfig) -> None:
        """Draw all elements."""
        # borders
        self.walls["w"] = self.border(c.w, self.v3, self.v1, fv, bv, gv)
        self.walls["s"] = self.border(c.s, self.v2, self.v1, fh, bh, gh)
        self.walls["e"] = self.border(c.e, self.v2, self.v4, fv, bv, gv)
        self.walls["n"] = self.border(c.n, self.v3, self.v4, fh, bh, gh)
        # content
        self.draw_particles()

    def draw_particles(self: Self) -> None:
        """Draw suitable line."""
        # fermions
        if self.fi is not None and self.fo is not None:
            if (
                self.bi is not None
                or self.bo is not None
                or self.gi is not None
                or self.go is not None
            ):
                self.d.line(self.fi, self.c, arrow_param=ARROW_PARAM)
                self.d.line(self.c, self.fo, arrow_param=ARROW_PARAM)
            else:
                if self.fi.x == self.fo.x or self.fi.y == self.fo.y:
                    self.d.line(self.fi, self.fo, arrow_param=ARROW_PARAM)
                else:
                    self.d.line(
                        self.fi,
                        self.fo,
                        arrow_param=ARROW_PARAM,
                        shape="elliptic",
                        ellipse_excentricity=1.0
                        if self.fi.x < self.fo.x
                        and (
                            (self.fi.x < self.c.x and self.fi.y < self.fo.y)
                            or (self.c.x < self.fo.x and self.fo.y < self.fi.y)
                        )
                        else -1.0,
                        ellipse_spread=0.25,
                    )
        # ew bosons
        if self.bi is not None and self.bo is not None:
            if self.bi.x == self.bo.x or self.bi.y == self.bo.y:
                self.d.line(self.bi, self.bo, flavour="wiggly")
            else:
                self.d.line(
                    self.bi,
                    self.bo,
                    flavour="wiggly",
                    shape="elliptic",
                    ellipse_excentricity=1.0 if self.bi.x < self.bo.x else -1.0,
                    ellipse_spread=0.25,
                )
        elif self.bi is not None:
            self.d.line(self.bi, self.c, flavour="wiggly", nwiggles=2.5, phase=0.5)
        elif self.bo is not None:
            self.d.line(self.c, self.bo, flavour="wiggly", nwiggles=2.5, phase=0.5)
        # gluon
        if self.gi is not None and self.go is not None:
            self.d.line(self.gi, self.go, flavour="loopy", nloops=8)
        elif self.gi is not None:
            self.d.line(self.gi, self.c, flavour="loopy", nloops=4)
        elif self.go is not None:
            self.d.line(self.c, self.go, flavour="loopy", nloops=4)


def draw_tiles(
    d: Diagram,
    xy: np.ndarray[tuple[int], np.dtype[np.float64]],
    tiles: list[list[TileConfig]],
) -> list[list[Tile]]:
    """Draw and collect all tiles."""
    out = []
    for j, row in enumerate(reversed(tiles)):
        out_row = []
        for k, col in enumerate(row):
            t = Tile(d, (xy[0] + k * SIZE, xy[1] + j * SIZE))
            t.draw(col)
            out_row.append(t)
        out.append(out_row)
    return out


def pdf(
    d: Diagram, xy: tuple[float, float], size: float, dx: float, dy: float
) -> RegularBubbleOperator:
    """PDF."""
    k = 24
    op = RegularBubbleOperator(k, xy, size, 0.25, facecolor="None")
    op.text("PDF", fontsize=20)
    d.add_operator(op)
    d.line(
        d.vertex(xy=(0.0, op.vertices[k // 4 + 1].y), marker=""),
        op.vertices[k // 4 + 1],
    )
    d.line(d.vertex(xy=(0.0, op.vertices[k // 4].y), marker=""), op.vertices[k // 4])
    d.line(
        d.vertex(xy=(0.0, op.vertices[k // 4 - 1].y), marker=""),
        op.vertices[k // 4 - 1],
    )
    d.line(
        op.vertices[k // 4 * 3 + 1],
        d.vertex(xy=(dx, op.vertices[k // 4 * 3 + 1].y + dy), marker=""),
    )
    v = d.vertex(xy=(dx, op.vertices[k // 4 * 3].y), marker="")
    d.line(op.vertices[k // 4 * 3], v)
    d.line(
        op.vertices[k // 4 * 3 - 1],
        d.vertex(xy=(dx, op.vertices[k // 4 * 3 - 1].y - dy), marker=""),
    )
    return op


def dy_tiles(ver: int = 0) -> list[list[TileConfig]]:
    """Drell-Yan variant tiles."""
    # ver = 0
    m = [
        [
            TileConfig(w="fo", e="fi"),
            TileConfig(w="fo", e="fi"),
            TileConfig(w="fo", s="fi"),
        ],
        [TileConfig(), TileConfig(), TileConfig(s="fi", n="fo", e="bo")],
        [
            TileConfig(w="fi", e="fo"),
            TileConfig(w="fi", e="fo"),
            TileConfig(w="fi", n="fo"),
        ],
    ]
    # ---------------
    # |   ||   ||   |
    # >--->>--->>-- |
    # |   ||   || | |
    # ------------v--
    # ------------v--
    # |   ||   || | |
    # |   ||   || |-b
    # |   ||   || | |
    # ------------v--
    # ------------v--
    # |   ||   || | |
    # <---<<---<<-- |
    # |   ||   ||   |
    # ---------------
    if ver == 1:
        m = [
            [
                TileConfig(w="fi", e="fo"),
                TileConfig(w="fi", e="fo"),
                TileConfig(w="fi", s="fo"),
            ],
            [TileConfig(), TileConfig(), TileConfig(s="fo", n="fi", e="bo")],
            [
                TileConfig(w="fo", e="fi"),
                TileConfig(w="fo", e="fi"),
                TileConfig(w="fo", n="fi"),
            ],
        ]
    elif ver == 2:
        m = [
            [TileConfig(w="fo", e="fi"), TileConfig(w="fo", s="fi"), TileConfig()],
            [
                TileConfig(),
                TileConfig(s="fi", n="fo", e="bo"),
                TileConfig(w="bi", e="bo"),
            ],
            [TileConfig(w="fi", e="fo"), TileConfig(w="fi", n="fo"), TileConfig()],
        ]
    elif ver == 4:
        m = [
            [
                TileConfig(w="fo", e="fi", s="gi"),
                TileConfig(w="fo", e="fi"),
                TileConfig(w="fo", s="fi"),
            ],
            [
                TileConfig(s="gi", n="go"),
                TileConfig(),
                TileConfig(s="fi", n="fo", e="bo"),
            ],
            [
                TileConfig(w="fi", e="fo", n="go"),
                TileConfig(w="fi", e="fo"),
                TileConfig(w="fi", n="fo"),
            ],
        ]
    return m


def dy_frame(d: Diagram, tiles: list[list[Tile]], ver1: bool) -> None:
    """Surrounding layout."""
    # final state
    phi = tiles[1][2].walls["e"]
    pho = d.vertex(phi.xy, dx=SIZE / 2.0, marker="")
    d.line(phi, pho, flavour="wiggly", nwiggles=2.5).text(
        r"$\gamma/Z/W$", y=0.11, t=0.2, fontsize=20
    )
    fo = d.vertex(phi.xy, dx=SIZE, dy=SIZE / 2.0, marker="")
    fo.text(r"$\bar{l}$", x=0.1, fontsize=20)
    afo = d.vertex(phi.xy, dx=SIZE, dy=-SIZE / 2.0, marker="")
    afo.text(r"$l'$", x=0.1, y=-0.025, fontsize=20)
    d.line(fo, pho, arrow_param=ARROW_PARAM)
    d.line(pho, afo, arrow_param=ARROW_PARAM)
    # initial state
    xy1 = tiles[0][0].v1.xy
    pdf1 = pdf(d, (xy1[0] / 2.0, xy1[1] / 2.0), SIZE * 0.25, xy1[0], 0.2)
    d.line(
        pdf1.vertices[-2] if ver1 else tiles[0][0].walls["w"],
        tiles[0][0].walls["w"] if ver1 else pdf1.vertices[-2],
        arrow_param=ARROW_PARAM,
        shape="elliptic",
        ellipse_excentricity=1.8 * (1 if ver1 else -1),
        ellipse_spread=0.25,
    )
    xy2 = tiles[2][0].v3.xy
    pdf2 = pdf(d, (xy2[0] / 2.0, (FIGSIZE[1] + xy2[1]) / 2.0), SIZE * 0.25, xy1[0], 0.2)
    d.line(
        tiles[2][0].walls["w"] if ver1 else pdf2.vertices[14],
        pdf2.vertices[14] if ver1 else tiles[2][0].walls["w"],
        arrow_param=ARROW_PARAM,
        shape="elliptic",
        ellipse_excentricity=1.8 * (1 if ver1 else -1),
        ellipse_spread=0.25,
    )
    # text
    d.text(FIGSIZE[0] * 0.5, FIGSIZE[1] * 0.92, "Drell-Yan process", fontsize=37)
    d.text(
        FIGSIZE[0] * 0.01,
        FIGSIZE[1] * 0.62,
        "Used for:",
        fontsize=25,
        va="top",
        ha="left",
    )
    # fmt: off
    txt = "- Search for new\n"\
          "  particles (Higgs)\n"\
          "- precision\n"\
          "  measurements\n"\
          "  (PDF, $m_W$)"
    # fmt: on
    d.text(FIGSIZE[0] * 0.01, FIGSIZE[1] * 0.57, txt, fontsize=20, va="top", ha="left")
    img1 = Image.open("DielectronsAll.png")
    ax1 = d.ax.inset_axes([0.75, 0.65, 0.25, 0.25])
    ax1.imshow(img1)
    ax1.set(xticks=[], yticks=[])
    img2 = Image.open("fig_11a.png")
    ax2 = d.ax.inset_axes([0.75, 0.05, 0.25, 0.25])
    ax2.imshow(img2)
    ax2.set(xticks=[], yticks=[])
    d.text(
        FIGSIZE[0] * 0.5,
        FIGSIZE[1] * 0.03,
        "Images: CMS collaboration; arXiv:2502.21088",
        fontsize=10,
    )


def dy(ver: int = 0):
    """Drell-Yan."""
    diagram = Diagram(figsize=FIGSIZE)
    tiles = draw_tiles(diagram, (FIGSIZE - 3 * TILESIZE) / 2.0, dy_tiles(ver))
    dy_frame(diagram, tiles, ver % 2 == 0)

    # adjust size
    diagram.ax.set_xlim(0.0, FIGSIZE[0])
    diagram.ax.set_ylim(0.0, FIGSIZE[1])
    diagram.ax.margins(0.0)
    diagram.fig.tight_layout(pad=0.0)

    # draw
    diagram.plot()
    diagram.savefig(f"dy-v{ver}.pdf")
    plt.close(diagram.fig)


dy(0)
dy(1)
# dy(2)
