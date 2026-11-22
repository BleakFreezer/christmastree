#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 3D ASCII CHRISTMAS TREE  -  Pygame + NumPy, smooth 60 FPS, 4 click-able themes
===============================================================================
 Install :  pip install pygame-ce numpy        (or: pip install pygame numpy)
 Run     :  python christmas_tree_pygame.py
 Music   :  put "last_christmas.mp3" in the same folder you run it from
            (optional - the animation runs silently if the file is missing)

 Controls:  LEFT CLICK = next theme      SPACE = pause / resume
            R = restart                  ESC / close = quit

 Themes (click anywhere to cycle, order below)
 ---------------------------------------------
   1. default          Classic Green     + red / gold / warm / ice baubles
   2. pink_sweets      Pink -> white     + cakes, ice cream, cupcakes, slices,
                                           donuts, candies and lollipops
   3. white_red        White tree        + subtle red accents, red baubles
   4. lightblue_sweets Blue -> white     + the same sweets in blue tones

 To edit a palette, change the hex codes in the THEMES dictionary below.

 How it stays fast
 -----------------
 * The tree is a static NumPy point cloud built once at startup (both
   decoration sets - baubles and sweets - are in the cloud; only the active
   set is projected each frame).
 * Every frame: rotate + project all visible points with vectorised NumPy,
   z-buffer them into the 88x74 character grid (sort by depth, last write
   wins), then run lighting ONLY for the <=6.5k winning points.
 * 60 FPS: the depth sort is a 16-bit radix sort, the cell maths runs only on
   on-screen points, glyph surfaces come from a cache via one list
   comprehension, and the snow is drawn from pre-built surfaces.
 * All motion (rotation, twinkle, pulses, snow) runs on real elapsed seconds,
   so 60 FPS is smoother, not faster.
 * Glyphs are pre-rendered lazily into a cache, so the main loop never calls
   font.render() per cell - it only blits cached little surfaces in one call.
 * Theme changes cross-fade through a few quantised palettes (each with its
   own glyph cache), so no per-frame colour maths on the cells is needed.

 Timeline (seconds)
 ------------------
   0.2 -  1.6   trunk grows
   1.0 -  8.5   needles / tiers grow bottom -> top
   4.0 - 11.0   garland spirals upward
   5.0 - 12.0   fairy lights switch on (then twinkle forever)
   6.0 - 12.0   ornaments / sweets pop in
   8.8          star pops onto the top
   9.6 - 11.4   gift boxes pop in around the base
  13.0 - 14.5   "MERRY CHRISTMAS 2026" fades in
===============================================================================
"""

import math
import os
import sys

import numpy as np
import pygame

# ------------------------------------------------------------------ settings
COLS, ROWS = 88, 74              # character grid
CW, CH = 9, 14                   # cell size in pixels
W, H = COLS * CW, ROWS * CH      # logical canvas: 792 x 1036
FPS = 60                         # frame-rate cap (clock.tick(FPS))
SNOW_COUNT = 520                 # background snow particles
FONT_SIZE = 15
FONT_NAMES = "consolas,menlo,dejavusansmono,couriernew,liberationmono,monospace"
MUSIC_FILE = "last_christmas.mp3"

BG = (6, 8, 14)                  # #06080E
HUD_COLOR = (112, 122, 146)
HINT_COLOR = (52, 60, 78)

# Brightest -> darkest.  Last char is a blank (nothing drawn).
RAMP = "@#W$9876543210?!abc;:+=-,._ "
EXTRA = "Oo*~"                   # extra glyphs: fairy lights + sweets cues
CHARS = RAMP + EXTRA
SPACE_IDX = RAMP.index(" ")
LV = 14                          # brightness levels per material (glyph cache)

# glyph "cues" used by the sweets (index into CHARS), blended with shading
CUE_O_SMALL, CUE_O_BIG, CUE_WAVE, CUE_PLUS = 1, 2, 3, 4
CUE_IDX = np.array([-1,
                    len(RAMP) + 1,          # 'o'  cherries, sprinkles
                    len(RAMP) + 0,          # 'O'  ice cream scoops
                    len(RAMP) + 3,          # '~'  frosting swirls
                    RAMP.index("+")],       # '+'  waffle / wrapper / stripes
                   dtype=np.int32)

ROT_SPEED = 36.0                 # degrees per second around the Y axis
PITCH = math.radians(10.0)       # camera looks slightly down on the tree
CP, SP = math.cos(PITCH), math.sin(PITCH)
CAM_D = 3000.0                   # perspective distance
GEO_SCALE = 0.93                 # shrink whole model to fit the canvas
CX = W / 2.0                     # screen x of the tree axis
SY0 = 492.0                      # screen y of the model's vertical centre
YC = 400.0 * GEO_SCALE           # model height used as rotation/pitch centre
INV_CW, INV_CH = 1.0 / CW, 1.0 / CH
ORN_SP = 3.8                     # bauble surface point spacing (smaller = denser)

# ---------------------------------------------------------------- timeline
T_TRUNK0, T_TRUNK_DUR = 0.2, 1.4
T_FOL0, T_FOL_DUR = 1.0, 7.5
T_GAR0, T_GAR_DUR = 4.0, 7.0
T_LIGHT0, T_LIGHT_DUR = 5.0, 7.0
T_ORN0, T_ORN_DUR = 6.0, 6.0
T_STAR = T_FOL0 + T_FOL_DUR + 0.3
T_GIFT0, T_GIFT_STEP = 9.6, 0.45
T_GREET, T_GREET_FADE = 13.0, 1.5
GREETING = "MERRY CHRISTMAS 2026"

# theme switching
THEME_FADE = 0.8                 # seconds for the colour cross-fade
FADE_STEPS = 4                   # quantised palettes used during the fade
DECO_IN, DECO_OUT = 0.55, 0.25   # decoration pop-in / shrink-out seconds


# =============================================================================
#  Colour helpers
# =============================================================================
def hex_to_rgb(hex_str):
    """'#FF8000' / 'FF8000' / '#F80'  ->  (255, 128, 0)."""
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"invalid hex colour: {hex_str!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def lerp_color(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def scale_color(c, k):
    return tuple(min(255, int(v * k)) for v in c)


def gradient(stops, x):
    """Sample a multi-stop colour gradient (list of RGB) at x in 0..1."""
    x = min(1.0, max(0.0, x))
    pos = x * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    return lerp_color(stops[i], stops[i + 1], pos - i)


def shade_color(base, x):
    """Flat-colour material: dark at x=0, base colour mid, whitish at x=1."""
    k = 0.16 + 0.84 * x
    hl = max(0.0, (x - 0.86) / 0.14) * 0.40          # only the brightest spots go whitish
    return tuple(int(min(255, c * k + (255 - c) * hl)) for c in base)


# =============================================================================
#  THEMES  -  edit the hex codes to customise
# =============================================================================
#  foliage    : gradient, dark shadow -> bright highlight (3-4 stops)
#  accent     : optional red-ish gradient sprinkled over the needles (or None)
#  garland    : (colour, highlight colour)
#  decor_style: "baubles" (spheres) or "sweets" (3D bakery mini-clusters)
#  decor      : baubles -> 3-4 colours;  sweets -> 5 roles:
#               [strong, mid, cream, base/dough, accent]
#  lights     : 4 fairy-light colours
#  star       : top star colour
#  gifts      : 5 x (box colour, ribbon colour)
#  ground     : snow disc under the tree
THEMES = {
    "default": dict(
        label="Classic Green",
        foliage=["#145A2A", "#2E9E44", "#7EE887"],
        accent=None,
        garland=("#FFD43B", "#FFE875"),
        decor_style="baubles",
        decor=["#FF2E4C", "#FFB703", "#FFF3B0", "#74C0FC"],
        lights=["#FF2E4C", "#FFB703", "#FFF3B0", "#74C0FC"],
        star="#FFEC78",
        gifts=[("#4678F0", "#FFD250"), ("#E03642", "#F0F4FF"),
               ("#A85AE2", "#FFD250"), ("#F8942C", "#F0F4FF"),
               ("#E03642", "#FFD250")],
        ground="#96ACCC",
    ),
    "pink_sweets": dict(
        label="Pink Sweets",
        foliage=["#C94A7C", "#FF85B3", "#FFD6E7", "#FFF5F8"],
        accent=None,
        garland=("#FFB3D1", "#FFF0F6"),
        decor_style="sweets",
        decor=["#FF4D79", "#FF8FAB", "#FFF8FA", "#D49A6A", "#A8E6CF"],
        lights=["#FF4D79", "#FF8FAB", "#FFF8FA", "#A8E6CF"],
        star="#FFE3EE",
        gifts=[("#FF4D79", "#FFF8FA"), ("#FFF8FA", "#FF4D79"),
               ("#A8E6CF", "#FF4D79"), ("#FF8FAB", "#FFF8FA"),
               ("#D49A6A", "#FFF8FA")],
        ground="#C7A9BC",
    ),
    "white_red": dict(
        label="White & Red",
        foliage=["#9AA5B4", "#DCE3EC", "#FFFFFF"],
        accent=["#E03131", "#FF6B6B"],
        garland=("#F8F9FA", "#E9ECEF"),
        decor_style="baubles",
        decor=["#C92A2A", "#FA5252", "#FF8787"],
        lights=["#FA5252", "#FF8787", "#FFFFFF", "#FFE3E3"],
        star="#FFE066",
        gifts=[("#C92A2A", "#FFFFFF"), ("#F1F3F5", "#C92A2A"),
               ("#FA5252", "#FFFFFF"), ("#E9ECEF", "#E03131"),
               ("#C92A2A", "#F8F9FA")],
        ground="#A9B4C4",
    ),
    "lightblue_sweets": dict(
        label="Light-Blue Sweets",
        foliage=["#2B6CB0", "#63B3ED", "#BEE3F8", "#F7FAFC"],
        accent=None,
        garland=("#90CDF4", "#EBF8FF"),
        decor_style="sweets",
        decor=["#3182CE", "#63B3ED", "#F8FAFC", "#BEE3F8", "#A3BFFA"],
        lights=["#3182CE", "#63B3ED", "#BEE3F8", "#F8FAFC"],
        star="#E6F4FF",
        gifts=[("#3182CE", "#F8FAFC"), ("#F8FAFC", "#3182CE"),
               ("#A3BFFA", "#F8FAFC"), ("#63B3ED", "#F8FAFC"),
               ("#BEE3F8", "#2B6CB0")],
        ground="#9CB4D0",
    ),
}
THEME_ORDER = list(THEMES)
DECO_SET = {"baubles": 0, "sweets": 1}
ACCENT_P = 0.07                  # share of needles that get the accent colour

# --------------------------------------------------------------- materials
(M_NEEDLE_A, M_NEEDLE_B, M_NEEDLE_C, M_NEEDLE_X, M_TRUNK, M_GARLAND,
 M_D0, M_D1, M_D2, M_D3, M_D4,                      # decoration colour slots
 M_L0, M_L1, M_L2, M_L3,                            # fairy-light colours
 M_STAR,
 M_BOX0, M_BOX1, M_BOX2, M_BOX3, M_BOX4,
 M_RIB0, M_RIB1, M_RIB2, M_RIB3, M_RIB4,
 M_GROUND) = range(27)
NMAT = 27

# sweets colour roles -> decoration slots
SW_STRONG, SW_MID, SW_CREAM, SW_BASE, SW_ACCENT = range(5)


def build_palette(theme):
    """palette[material][level] -> RGB for one theme (LV brightness levels)."""
    foliage = [hex_to_rgb(h) for h in theme["foliage"]]
    if theme.get("accent"):
        a = [hex_to_rgb(h) for h in theme["accent"]]
        accent = [scale_color(a[0], 0.55)] + a
    else:
        accent = foliage
    g1, g2 = (hex_to_rgb(h) for h in theme["garland"])
    garland = [scale_color(g1, 0.45), g1, g2]

    spec = [None] * NMAT                      # ("grad", stops, offset) | ("flat", rgb)
    spec[M_NEEDLE_A] = ("grad", foliage, 0.0)
    spec[M_NEEDLE_B] = ("grad", foliage, -0.10)   # slightly darker patches
    spec[M_NEEDLE_C] = ("grad", foliage, 0.10)    # slightly brighter patches
    spec[M_NEEDLE_X] = ("grad", accent, 0.0)
    spec[M_GARLAND] = ("grad", garland, 0.0)
    spec[M_TRUNK] = ("flat", hex_to_rgb(theme.get("trunk", "#9E6434")))
    decor = [hex_to_rgb(h) for h in theme["decor"]]
    for i in range(5):
        spec[M_D0 + i] = ("flat", decor[i % len(decor)])
    for i, h in enumerate(theme["lights"]):
        spec[M_L0 + i] = ("flat", hex_to_rgb(h))
    spec[M_STAR] = ("flat", hex_to_rgb(theme["star"]))
    for i, (box, rib) in enumerate(theme["gifts"]):
        spec[M_BOX0 + i] = ("flat", hex_to_rgb(box))
        spec[M_RIB0 + i] = ("flat", hex_to_rgb(rib))
    spec[M_GROUND] = ("flat", hex_to_rgb(theme["ground"]))

    pal = []
    for s in spec:
        row = []
        for l in range(LV):
            x = l / (LV - 1)
            if s[0] == "grad":
                row.append(gradient(s[1], x + s[2]))
            else:
                row.append(shade_color(s[1], x))
        pal.append(row)
    return pal


# point "kinds" (drive special lighting / animation)
K_FOL, K_LIGHT, K_ORN, K_GARLAND, K_STAR, K_BOX, K_GROUND, K_SWEET = range(8)

# light direction (camera space: x right, y up, z towards viewer)
def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)

L_MAIN = _norm((-0.45, 0.60, 0.66))
L_FILL = _norm((0.70, -0.10, 0.55))
L_HALF = _norm(L_MAIN + np.array([0.0, 0.0, 1.0]))
AMBIENT = 0.16

# fairy light glyph ladder, dim -> bright  (indices into CHARS)
LIGHT_SEQ = np.array([RAMP.index("."), RAMP.index("+"), len(RAMP) + 2,
                      len(RAMP) + 1, len(RAMP), 0], dtype=np.int32)


# -------------------------------------------------------------------- helpers
def ease_out_back(u):
    """Overshooting ease used for pop-in animations (u in 0..1)."""
    c1 = 1.70158
    c3 = c1 + 1.0
    u = np.asarray(u, dtype=np.float32) - 1.0
    return 1.0 + c3 * u ** 3 + c1 * u ** 2


def smooth01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def fib_sphere(n):
    """n roughly evenly spread unit vectors."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi),
                     np.sin(th) * np.sin(phi)], axis=1)


# ---------------------------------------------------------- tree dimensions
N_TIER = 8
STEP = 88.0                       # vertical distance between tiers
Y_BASE = 48.0                     # y where the lowest tier starts
DROP = 57.0                       # radius lost per STEP inside one tier (bigger = deeper skirts)
R_TIER = np.array([262, 230, 199, 170, 141, 113, 85, 56], dtype=np.float64)
Y0_TIER = Y_BASE + STEP * np.arange(N_TIER)
FOL_TOP = 800.0
H_TIER = R_TIER * STEP / DROP
H_TIER[-1] = FOL_TOP - Y0_TIER[-1]
STAR_Y = 820.0
GROUND_R = 352.0


def tree_profile(y):
    """Outer radius of the tiered tree at height y (max over all tiers)."""
    y = np.asarray(y, dtype=np.float64)
    prof = np.zeros_like(y)
    for R, y0, h in zip(R_TIER, Y0_TIER, H_TIER):
        r = R * (1.0 - (y - y0) / h)
        r = np.where((y >= y0) & (y <= y0 + h), r, 0.0)
        prof = np.maximum(prof, r)
    return prof


def foliage_birth(y):
    """Time at which foliage at height y has grown in (bottom -> top)."""
    return T_FOL0 + T_FOL_DUR * np.clip((y - Y_BASE) / (FOL_TOP - Y_BASE), 0, 1)


# =============================================================================
#  Sweets: tiny 3D bakery models built from surface samplers
#  (local coordinates, y up; each returns points + outward normals)
# =============================================================================
SP_S = 3.8                        # surface sampling spacing for sweets


def _rotx(A, a):
    c, s = math.cos(a), math.sin(a)
    return np.stack([A[:, 0], A[:, 1] * c - A[:, 2] * s, A[:, 1] * s + A[:, 2] * c], 1)


def _rotz(A, a):
    c, s = math.cos(a), math.sin(a)
    return np.stack([A[:, 0] * c - A[:, 1] * s, A[:, 0] * s + A[:, 1] * c, A[:, 2]], 1)


def s_frustum(r0, r1, y0, y1, sp=SP_S):
    """Side of a cylinder / cone (radius r0 at y0 -> r1 at y1)."""
    h = y1 - y0
    rows = max(2, int(round(math.hypot(h, r1 - r0) / sp)) + 1)
    slope = (r0 - r1) / h
    P, N = [], []
    for i, t in enumerate(np.linspace(0.0, 1.0, rows)):
        r = max(r0 + (r1 - r0) * t, 0.5)
        n = max(6, int(2 * math.pi * r / sp))
        th = (np.arange(n) + 0.5 * (i % 2)) * (2 * math.pi / n)
        c, s = np.cos(th), np.sin(th)
        P.append(np.stack([r * c, np.full(n, y0 + h * t), r * s], 1))
        N.append(np.stack([c, np.full(n, slope), s], 1))
    return np.concatenate(P), np.concatenate(N)


def s_disc(r, y, r_in=0.0, sp=SP_S, up=1.0):
    """Flat disc / ring facing up (or down with up=-1)."""
    P = [np.array([[0.0, y, 0.0]])] if r_in <= 0 else []
    radii = list(np.arange(r_in if r_in > 0 else sp, r + 1e-6, sp))
    for i, rr in enumerate(radii):
        n = max(6, int(2 * math.pi * rr / sp))
        th = (np.arange(n) + 0.5 * (i % 2)) * (2 * math.pi / n)
        P.append(np.stack([rr * np.cos(th), np.full(n, y), rr * np.sin(th)], 1))
    P = np.concatenate(P)
    return P, np.tile([0.0, up, 0.0], (len(P), 1))


def s_sphere(c, r, sp=SP_S, keep=None, sy=1.0):
    """Sphere / ellipsoid / dome (keep = mask function of the unit normals)."""
    u = fib_sphere(max(12, int(4 * math.pi * r * r / (sp * sp))))
    if keep is not None:
        u = u[keep(u)]
    return np.asarray(c, dtype=np.float64) + u * np.array([r, r * sy, r]), u


def s_torus(R, r, tilt, sp=SP_S):
    """Donut. Returns points, normals and the untilted normal-y (icing mask)."""
    nu, nv = max(10, int(2 * math.pi * R / sp)), max(8, int(2 * math.pi * r / sp))
    U, V = np.meshgrid(np.arange(nu) * (2 * math.pi / nu), np.arange(nv) * (2 * math.pi / nv))
    U, V = U.ravel(), V.ravel()
    P = np.stack([(R + r * np.cos(V)) * np.cos(U), r * np.sin(V),
                  (R + r * np.cos(V)) * np.sin(U)], 1)
    N = np.stack([np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U)], 1)
    return _rotx(P, tilt), _rotx(N, tilt), N[:, 1].copy()


class Shape:
    """Collects the coloured parts of one sweet (local coordinates)."""

    def __init__(self, rng):
        self.rng = rng
        self.P, self.N, self.slot, self.cue = [], [], [], []

    def add(self, pn, slot, cue=0, cue_p=1.0):
        P, N = pn
        n = len(P)
        slot = np.broadcast_to(np.asarray(slot), (n,)).astype(np.int16)
        if np.ndim(cue) == 0:
            cue = (np.where(self.rng.random(n) < cue_p, cue, 0) if cue
                   else np.zeros(n))
        self.P.append(np.asarray(P, dtype=np.float64))
        self.N.append(np.asarray(N, dtype=np.float64))
        self.slot.append(slot)
        self.cue.append(np.asarray(cue, dtype=np.int8))

    def finish(self):
        P = np.concatenate(self.P)
        P = P - (P.min(0) + P.max(0)) / 2.0                 # centre on its bounding box
        return (P, np.concatenate(self.N), np.concatenate(self.slot),
                np.concatenate(self.cue))


def sw_cake(S):
    S.add(s_frustum(19, 19, 0, 13), SW_BASE, CUE_PLUS, 0.20)          # sponge
    S.add(s_frustum(19.5, 19.5, 13, 16.5), SW_CREAM)                  # cream layer
    S.add(s_disc(19, 16.5, r_in=15), SW_CREAM)                        # ledge
    S.add(s_frustum(15, 15, 16.5, 28), SW_MID, CUE_WAVE, 0.25)        # pink tier
    S.add(s_disc(15, 28), SW_CREAM, CUE_WAVE, 0.50)                   # frosting top
    for a in np.arange(8) * (math.pi / 4):                            # cream drips
        S.add(s_sphere((13.5 * math.cos(a), 26.0, 13.5 * math.sin(a)), 3.2, 2.6),
              SW_CREAM, CUE_WAVE, 0.4)
    S.add(s_sphere((0, 32.5, 0), 4.2, 3.0), SW_STRONG, CUE_O_SMALL, 0.8)   # cherry


def sw_icecream(S):
    S.add(s_frustum(0.5, 11, 0, 21), SW_BASE, CUE_PLUS, 0.45)         # waffle cone
    S.add(s_sphere((0, 28, 0), 11.5), SW_MID, CUE_O_BIG, 0.30)        # scoop 1
    S.add(s_sphere((0, 39, 0), 9.5), SW_CREAM, CUE_O_BIG, 0.30)       # scoop 2
    S.add(s_disc(12.5, 22, r_in=9, up=-1.0), SW_CREAM, CUE_WAVE, 0.5)  # melting skirt
    S.add(s_sphere((0, 47.5, 0), 4.2, 3.0), SW_STRONG, CUE_O_SMALL, 0.8)   # cherry


def sw_cupcake(S):
    S.add(s_frustum(10, 14, 0, 13), SW_ACCENT, CUE_PLUS, 0.35)        # paper wrapper
    S.add(s_sphere((0, 16, 0), 13.5, keep=lambda u: u[:, 1] > -0.25), SW_MID, CUE_WAVE, 0.30)
    S.add(s_sphere((0, 25, 0), 10.0, keep=lambda u: u[:, 1] > -0.1), SW_CREAM, CUE_WAVE, 0.40)
    S.add(s_sphere((0, 32, 0), 6.0, 3.0, keep=lambda u: u[:, 1] > 0.0), SW_MID, CUE_WAVE, 0.40)
    S.add(s_sphere((0, 37.5, 0), 4.0, 3.0), SW_STRONG, CUE_O_SMALL, 0.8)   # cherry


def sw_slice(S):
    R, A, H = 27.0, math.radians(60.0), 24.0
    a0, a1 = -A / 2.0, A / 2.0
    band = lambda y: np.where(y < 8, SW_BASE, np.where(y < 12, SW_CREAM,
                              np.where(y < 20, SW_BASE, SW_MID)))
    ys = np.arange(1.0, H, 2.6)
    th = np.linspace(a0, a1, max(4, int(R * A / SP_S)))
    TH, Y = [m.ravel() for m in np.meshgrid(th, ys)]                  # curved crust
    S.add((np.stack([R * np.cos(TH), Y, R * np.sin(TH)], 1),
           np.stack([np.cos(TH), np.zeros_like(TH), np.sin(TH)], 1)), band(Y))
    rr = np.arange(SP_S, R, SP_S)
    for a, sg in ((a0, -1.0), (a1, 1.0)):                             # two cut faces
        RR, Y = [m.ravel() for m in np.meshgrid(rr, ys)]
        S.add((np.stack([RR * math.cos(a), Y, RR * math.sin(a)], 1),
               np.tile([-sg * math.sin(a), 0.0, sg * math.cos(a)], (len(RR), 1))), band(Y))
    top = []
    for r in rr:                                                      # frosted top
        t = np.linspace(a0, a1, max(2, int(r * A / SP_S) + 1))
        top.append(np.stack([r * np.cos(t), np.full(t.shape, H), r * np.sin(t)], 1))
    top = np.concatenate(top)
    S.add((top, np.tile([0.0, 1.0, 0.0], (len(top), 1))), SW_CREAM, CUE_WAVE, 0.5)
    S.add(s_sphere((17.0, H + 3.0, 0.0), 4.2, 3.0), SW_STRONG, CUE_O_SMALL, 0.8)


def sw_donut(S):
    P, N, ny = s_torus(13.0, 6.2, tilt=math.radians(48.0))
    icing = ny > -0.05
    spr = icing & (S.rng.random(len(P)) < 0.17)                       # sprinkles
    pick = S.rng.choice([SW_STRONG, SW_CREAM, SW_ACCENT], len(P))
    slot = np.where(spr, pick, np.where(icing, SW_MID, SW_BASE))
    S.add((P, N), slot, np.where(spr, CUE_O_SMALL, 0))


def sw_candy(S):
    a = math.radians(22.0)
    rz = lambda pn: (_rotz(pn[0], a), _rotz(pn[1], a))
    P, N = s_sphere((0, 0, 0), 8.5, sy=0.85)                          # striped body
    k = np.floor((np.arctan2(N[:, 2], N[:, 0]) + math.pi) / (math.pi / 4)).astype(int) % 2
    S.add(rz((P, N)), np.where(k == 0, SW_STRONG, SW_CREAM),
          np.where((k == 1) & (S.rng.random(len(P)) < 0.5), CUE_PLUS, 0))
    for sg in (1.0, -1.0):                                            # twisted wrapper ends
        P, N = s_frustum(3.0, 8.0, 9.0, 17.0)
        P2 = np.stack([sg * P[:, 1], P[:, 0], P[:, 2]], 1)
        N2 = np.stack([sg * N[:, 1], N[:, 0], N[:, 2]], 1)
        S.add(rz((P2, N2)), SW_MID, CUE_WAVE, 0.4)


def sw_lollipop(S):
    P, N = s_sphere((0, 26, 0), 11.0)
    ang = np.arctan2(N[:, 2], N[:, 0]) + 3.0 * N[:, 1]                # spiral swirl
    k = np.floor(ang / (math.pi / 3)).astype(int) % 2
    S.add((P, N), np.where(k == 0, SW_STRONG, SW_CREAM),
          np.where((k == 1) & (S.rng.random(len(P)) < 0.5), CUE_WAVE, 0))
    S.add(s_frustum(2.4, 2.4, 0, 16, sp=2.8), SW_CREAM)               # stick


SWEETS = [sw_cake, sw_icecream, sw_cupcake, sw_slice, sw_donut, sw_candy, sw_lollipop]


# =============================================================================
#  Scene: builds the static 3D point cloud, then renders it every frame
# =============================================================================
class TreeScene:
    def __init__(self, active_set=0, seed=2026):
        self.rng = np.random.default_rng(seed)        # original tree stream (unchanged)
        self.rng_x = np.random.default_rng(seed + 1)  # extras: accents + sweets
        self._parts = []
        self._build_ground()
        self._build_trunk()
        self._build_foliage()
        self._build_garland()
        self._build_ornaments()
        self._build_sweets()
        self._build_lights()
        self._build_star()
        self._build_gifts()
        self._finalize()

        # decoration-set switching state (times are on the "fx" clock)
        self.deco_active = active_set
        self.deco_since = [-1e9, -1e9]
        self.deco_off = [-1e9, -1e9]
        self.swap_end = -1.0

    # ---- point collection ---------------------------------------------------
    def _add(self, pos, nrm, mat, birth, kind, *, pop=0.0, piv=None,
             shin=0.0, ao=1.0, phase=0.0, tw=0.0, dset=-1, cue=0):
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        n = len(pos)
        if n == 0:
            return
        nrm = np.asarray(nrm, dtype=np.float32).reshape(-1, 3)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        if piv is None:
            piv = pos
        piv = np.broadcast_to(np.asarray(piv, dtype=np.float32), (n, 3))

        def bc(v, dt):
            return np.broadcast_to(np.asarray(v, dtype=dt), (n,)).copy()

        self._parts.append(dict(
            pos=pos, nrm=nrm, piv=piv.copy(),
            mat=bc(mat, np.int16), birth=bc(birth, np.float32),
            kind=bc(kind, np.int8), pop=bc(pop, np.float32),
            shin=bc(shin, np.float32), ao=bc(ao, np.float32),
            phase=bc(phase, np.float32), tw=bc(tw, np.float32),
            cue=bc(cue, np.int8),
            setcode=bc(dset if dset >= 0 else 2, np.int8)))

    def _finalize(self):
        cat = lambda k: np.concatenate([p[k] for p in self._parts])
        self.pos = cat("pos") * GEO_SCALE
        self.piv = cat("piv") * GEO_SCALE
        self.nrm = cat("nrm")
        self.mat = cat("mat")
        self.birth = cat("birth")
        self.kind = cat("kind")
        self.pop = cat("pop")
        self.shin = cat("shin")
        self.ao = cat("ao")
        self.phase = cat("phase")
        self.tw = cat("tw")
        self.cue = cat("cue")
        self.setcode = cat("setcode")                  # 0 baubles, 1 sweets, 2 always
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float32)
        self.count = len(self.pos)
        self._parts = []

    # ---- decoration-set switching ------------------------------------------
    def set_deco(self, dset, fx):
        """Swap to decoration set `dset` (old one shrinks away, new one pops in)."""
        if dset == self.deco_active:
            return
        self.deco_off[self.deco_active] = fx
        self.deco_since[dset] = fx
        self.deco_active = dset
        self.swap_end = fx + DECO_IN + 0.05

    def _deco_state(self, fx):
        vis = np.ones(3, dtype=bool)
        sf = np.ones(3, dtype=np.float32)
        for s in (0, 1):
            if s == self.deco_active:
                u = (fx - self.deco_since[s]) / DECO_IN
                f = 1.0 if u >= 1.0 else float(ease_out_back(max(u, 0.0)))
            else:
                f = max(0.0, 1.0 - (fx - self.deco_off[s]) / DECO_OUT)
            sf[s] = f
            vis[s] = f > 0.0
        return vis, sf

    # ---- ground (soft snow disc so the tree is not floating) ----------------
    def _build_ground(self):
        rng = self.rng
        pts, ao = [], []
        for r in np.arange(8.0, GROUND_R, 7.0):
            n = max(6, int(2 * math.pi * r / 7.0))
            th = (np.arange(n) + rng.random()) * (2 * math.pi / n)
            th += rng.uniform(-0.3, 0.3, n) * (2 * math.pi / n)
            pts.append(np.stack([r * np.cos(th), np.zeros(n), r * np.sin(th)], 1))
            ao.append(np.full(n, 0.10 + 0.34 * (1.0 - (r / GROUND_R) ** 1.4)))
        pts = np.concatenate(pts)
        ao = np.concatenate(ao)
        nrm = np.tile([0.0, 1.0, 0.0], (len(pts), 1))
        self._add(pts, nrm, M_GROUND, 0.0, K_GROUND, ao=ao)

    # ---- trunk ---------------------------------------------------------------
    def _build_trunk(self):
        rng = self.rng
        pos, nrm, birth = [], [], []
        for y in np.arange(0.0, Y_BASE + 70.0, 4.0):
            r = 24.0 + 12.0 * math.exp(-y / 14.0)          # root flare
            n = max(8, int(2 * math.pi * r / 4.5))
            th = np.arange(n) * (2 * math.pi / n) + rng.uniform(-0.08, 0.08)
            pos.append(np.stack([r * np.cos(th), np.full(n, y), r * np.sin(th)], 1))
            nn = np.stack([np.cos(th), np.zeros(n), np.sin(th)], 1)
            nn += rng.normal(0, 0.22, nn.shape)              # bark texture
            nrm.append(nn)
            birth.append(np.full(n, T_TRUNK0 + T_TRUNK_DUR * min(1.0, y / 70.0)))
        self._add(np.concatenate(pos), np.concatenate(nrm),
                  M_TRUNK, np.concatenate(birth), K_FOL)

    # ---- tiered needles ------------------------------------------------------
    def _build_foliage(self):
        rng = self.rng
        for i in range(N_TIER):
            R, y0, h = R_TIER[i], Y0_TIER[i], H_TIER[i]
            last = (i == N_TIER - 1)
            tmax = 1.0 if last else min(1.0, (STEP + 16.0) / h)  # visible band only
            slant = tmax * math.hypot(h, R)
            nrow = max(3, int(math.ceil(slant / 4.2)))
            ts = np.linspace(0.0, tmax, nrow)
            k = 13 - i                       # number of "branches" around the cone
            A = 0.07
            phi = rng.uniform(0, 2 * math.pi)

            TH, TT = [], []
            for j, t in enumerate(ts):
                n = max(8, int(2 * math.pi * R * (1 - t) / 4.0))
                dth = 2 * math.pi / n
                th = (np.arange(n) + (0.5 if j % 2 else 0.0)) * dth
                th += rng.uniform(-0.25, 0.25, n) * dth
                tt = t + rng.uniform(-0.3, 0.3, n) * (tmax / nrow)
                TH.append(th)
                TT.append(tt)
            th = np.concatenate(TH)
            tt = np.clip(np.concatenate(TT), 0.0, tmax)

            # branch ridges (modulate the radius around the circumference)
            s1 = np.sin(k * th + phi)
            m = 1.0 + A * s1 + 0.03 * np.sin((k + 4) * th + 2 * phi)
            dm = A * k * np.cos(k * th + phi) + 0.03 * (k + 4) * np.cos((k + 4) * th + 2 * phi)
            rn = R * (1.0 - tt)
            r = rn * m + rng.uniform(-2.0, 2.0, th.shape)
            y = y0 + tt * h - 16.0 * (1.0 - tt) * (0.5 + 0.5 * s1) ** 2  # drooping branch tips
            c, s = np.cos(th), np.sin(th)
            pos = np.stack([r * c, y, r * s], 1)

            # exact surface normal from the two tangent vectors
            Tt = np.stack([-R * m * c, np.full(th.shape, h), -R * m * s], 1)
            Tth = np.stack([rn * dm * c - rn * m * s, np.zeros(th.shape),
                            rn * dm * s + rn * m * c], 1)
            nrm = np.cross(Tt, Tth)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
            nrm += rng.normal(0, 0.16, nrm.shape)            # needle texture

            # darker just under the skirt of the tier above (cheap ambient occlusion)
            if last:
                ao = np.ones(th.shape)
            else:
                x = np.clip((tt / tmax - 0.35) / 0.65, 0, 1)
                ao = 1.0 - 0.62 * x * x * (3 - 2 * x)

            v = np.sin(k * th * 0.5 + phi * 2 + tt * 9.0)    # patchy colour variation
            mat = np.where(v > 0.65, M_NEEDLE_C, np.where(v < -0.6, M_NEEDLE_B, M_NEEDLE_A))
            # sprinkled accent needles (only coloured differently by themes with an accent)
            mat = np.where(self.rng_x.random(th.shape) < ACCENT_P, M_NEEDLE_X, mat)
            self._add(pos, nrm, mat, foliage_birth(y), K_FOL, ao=ao)

    # ---- golden spiral garland ----------------------------------------------
    def _build_garland(self):
        ys = np.arange(0.0, 830.0, 1.0)
        prof = tree_profile(ys)
        kern = np.exp(-0.5 * (np.arange(-40, 41) / 14.0) ** 2)
        kern /= kern.sum()
        prof_s = np.convolve(np.pad(prof, 40, mode="edge"), kern, mode="valid")
        prof_s = np.maximum(prof_s, prof)

        n = 1900
        u = np.linspace(0.0, 1.0, n)
        y = 72.0 + (772.0 - 72.0) * u
        turns = 7.5
        th = turns * 2 * math.pi * u + 0.7
        rc = np.interp(y, ys, prof_s) * 1.10 + 9.0
        radial = np.stack([np.cos(th), np.zeros(n), np.sin(th)], 1)
        center = radial * rc[:, None] + np.stack([np.zeros(n), y, np.zeros(n)], 1)
        up = np.tile([0.0, 1.0, 0.0], (n, 1))
        tube = 6.5
        for j in range(4):                                     # tube cross-section
            a = j * math.pi / 2 + 0.4
            nn = math.cos(a) * radial + math.sin(a) * up
            pts = center + tube * nn
            birth = np.maximum(T_GAR0 + T_GAR_DUR * u, foliage_birth(y) + 0.4)
            self._add(pts, nn, M_GARLAND, birth, K_GARLAND, shin=0.6,
                      phase=u * 34.0)

    # ---- baubles (decoration set 0) -------------------------------------------
    def _build_ornaments(self):
        rng = self.rng
        mats = [M_D0, M_D1, M_D2, M_D3]
        probs = [0.34, 0.28, 0.14, 0.24]
        for _ in range(72):
            y = 100.0 + 620.0 * rng.random() ** 1.15
            th = rng.uniform(0, 2 * math.pi)
            rad = 12.5 + 7.5 * (1.0 - y / 800.0)               # bigger baubles (was 8..13)
            rad_old = 8.0 + 5.0 * (1.0 - y / 800.0)
            # centre pushed out with the extra radius so they sit proud on the branches
            rc = float(tree_profile(y)) * 1.06 + 4.0 + 0.5 * (rad - rad_old)
            unit = fib_sphere(max(60, int(4 * math.pi * rad * rad / (ORN_SP * ORN_SP))))
            c = np.array([rc * math.cos(th), y, rc * math.sin(th)])
            birth = max(T_ORN0 + T_ORN_DUR * rng.random(), float(foliage_birth(y)) + 0.5)
            self._add(c + unit * rad, unit, int(rng.choice(mats, p=probs)), birth,
                      K_ORN, pop=0.55, piv=c, shin=0.95, phase=rng.uniform(0, 6.28),
                      dset=0)

    # ---- bakery sweets (decoration set 1) --------------------------------------
    def _build_sweets(self):
        rng = self.rng_x
        n = 64
        kinds = np.resize(np.arange(len(SWEETS)), n)
        rng.shuffle(kinds)
        placed = []
        for kind in kinds:
            for _ in range(60):                       # keep clusters apart
                y = 95.0 + 640.0 * rng.random() ** 1.1
                th = rng.uniform(0, 2 * math.pi)
                rc = float(tree_profile(y)) * 1.04 + 16.0
                c = np.array([rc * math.cos(th), y, rc * math.sin(th)])
                if all(np.linalg.norm(c - q) >= 52.0 for q in placed):
                    break
            placed.append(c)

            S = Shape(rng)
            SWEETS[int(kind)](S)
            P, N, slot, cue = S.finish()

            scale = (1.15 + 0.45 * (1.0 - y / 800.0)) * rng.uniform(0.92, 1.08)   # ~1.3x bigger
            yaw = rng.uniform(0, 2 * math.pi)
            ca, sa = math.cos(yaw), math.sin(yaw)
            rot = lambda A: np.stack([A[:, 0] * ca + A[:, 2] * sa, A[:, 1],
                                      -A[:, 0] * sa + A[:, 2] * ca], 1)
            P = rot(P * scale) + c
            N = rot(N) + rng.normal(0, 0.05, N.shape)

            birth = max(T_ORN0 + T_ORN_DUR * rng.random(), float(foliage_birth(y)) + 0.5)
            self._add(P, N, M_D0 + slot, birth, K_SWEET, pop=0.6, piv=c,
                      shin=0.25, cue=cue, dset=1)

    # ---- fairy lights --------------------------------------------------------
    def _build_lights(self):
        rng = self.rng
        n = 900
        y = rng.uniform(70.0, 780.0, n)
        th = rng.uniform(0, 2 * math.pi, n)
        r = tree_profile(y) * 1.09 + 6.0
        pos = np.stack([r * np.cos(th), y, r * np.sin(th)], 1)
        nrm = np.stack([np.cos(th), np.zeros(n), np.sin(th)], 1)
        mat = rng.choice([M_L0, M_L1, M_L2, M_L3], n, p=[0.28, 0.26, 0.24, 0.22])
        birth = np.maximum(T_LIGHT0 + T_LIGHT_DUR * rng.random(n), foliage_birth(y) + 0.3)
        self._add(pos, nrm, mat, birth, K_LIGHT,
                  phase=rng.uniform(0, 6.28, n), tw=rng.uniform(1.8, 4.6, n))

    # ---- star ----------------------------------------------------------------
    def _build_star(self):
        R, r_in, T = 64.0, 28.0, 18.0
        g = np.arange(-R, R + 0.1, 3.0)
        gx, gy = np.meshgrid(g, g)
        gx, gy = gx.ravel(), gy.ravel()
        rr = np.hypot(gx, gy)
        phi = np.arctan2(gx, gy)                    # 0 = straight up
        rb = r_in + (R - r_in) * (0.5 + 0.5 * np.cos(5 * phi)) ** 1.6
        keep = (rr <= rb) & (rr > 0)
        gx, gy, rr, rb, phi = gx[keep], gy[keep], rr[keep], rb[keep], phi[keep]
        rho = rr / rb
        z = T * (1.0 - rho) ** 0.8
        slope = T / rb * 0.9
        nx, ny = gx / rr * slope, gy / rr * slope
        centre = np.array([0.0, STAR_Y, 0.0])
        for sgn in (1.0, -1.0):
            pos = np.stack([gx, gy + STAR_Y, sgn * z], 1)
            nrm = np.stack([nx, ny, np.full(len(gx), sgn)], 1)
            self._add(pos, nrm, M_STAR, T_STAR, K_STAR, pop=1.1, piv=centre,
                      shin=0.8, phase=rho * 3.0)

    # ---- gift boxes ----------------------------------------------------------
    def _build_gifts(self):
        # (angle deg, radius, w, h, d, yaw deg)  - colours come from the theme
        gifts = [
            (25, 318, 84, 62, 84, 20),
            (100, 312, 66, 86, 66, -15),
            (170, 322, 96, 54, 72, 35),
            (238, 314, 62, 62, 62, 8),
            (305, 320, 78, 74, 66, -28),
        ]
        for idx, (ang, rad, w, h, d, yaw) in enumerate(gifts):
            self._make_box(math.radians(ang), rad, w, h, d, math.radians(yaw),
                           M_BOX0 + idx, M_RIB0 + idx, T_GIFT0 + idx * T_GIFT_STEP)

    def _make_box(self, ang, rad, w, h, d, yaw, mat_box, mat_rib, birth):
        sp = 4.5
        rw = 6.0                                          # ribbon half-width
        xs = np.arange(-w / 2 + sp / 2, w / 2, sp)
        ys = np.arange(sp / 2, h, sp)
        zs = np.arange(-d / 2 + sp / 2, d / 2, sp)
        P, N, RIB = [], [], []

        X, Z = np.meshgrid(xs, zs)                       # top face
        X, Z = X.ravel(), Z.ravel()
        P.append(np.stack([X, np.full(X.shape, h), Z], 1))
        N.append(np.tile([0, 1, 0], (len(X), 1)))
        RIB.append((np.abs(X) < rw) | (np.abs(Z) < rw))
        for sgn in (1, -1):
            X, Y = np.meshgrid(xs, ys)                   # front / back
            X, Y = X.ravel(), Y.ravel()
            P.append(np.stack([X, Y, np.full(X.shape, sgn * d / 2)], 1))
            N.append(np.tile([0, 0, sgn], (len(X), 1)))
            RIB.append(np.abs(X) < rw)
            Z, Y = np.meshgrid(zs, ys)                   # left / right
            Z, Y = Z.ravel(), Y.ravel()
            P.append(np.stack([np.full(Z.shape, sgn * w / 2), Y, Z], 1))
            N.append(np.tile([sgn, 0, 0], (len(Z), 1)))
            RIB.append(np.abs(Z) < rw)

        P = np.concatenate(P).astype(np.float64)
        N = np.concatenate(N).astype(np.float64)
        RIB = np.concatenate(RIB)
        P += N * (RIB[:, None] * 1.5)                    # ribbon sits slightly proud

        # bow = three small spheres on top of the box
        unit = fib_sphere(36)
        bow_c = [(-8.0, h + 7.0, 0.0, 8.0), (8.0, h + 7.0, 0.0, 8.0), (0.0, h + 4.0, 0.0, 6.0)]
        BP = np.concatenate([unit * r + np.array([cx, cy, cz]) for cx, cy, cz, r in bow_c])
        BN = np.tile(unit, (3, 1))
        P = np.concatenate([P, BP])
        N = np.concatenate([N, BN])
        RIB = np.concatenate([RIB, np.ones(len(BP), dtype=bool)])

        ca, sa = math.cos(yaw), math.sin(yaw)            # yaw the box, then place it
        rot = lambda A: np.stack([A[:, 0] * ca + A[:, 2] * sa, A[:, 1],
                                  -A[:, 0] * sa + A[:, 2] * ca], 1)
        P, N = rot(P), rot(N)
        cx, cz = rad * math.cos(ang), rad * math.sin(ang)
        P += np.array([cx, 0.0, cz])
        N += self.rng.normal(0, 0.07, N.shape)           # slight texture
        mat = np.where(RIB, mat_rib, mat_box)
        self._add(P, N, mat, birth, K_BOX, pop=0.7, piv=np.array([cx, 0.0, cz]),
                  shin=0.15)

    # =========================================================================
    #  Per-frame rendering  (all NumPy, no Python loops over points)
    # =========================================================================
    def render(self, t, ang, fx):
        """Return arrays describing which glyph goes in which grid cell.

        t  = animation time (pauses),  fx = UI clock (theme swaps, never pauses)
        """
        ca, sa = math.cos(ang), math.sin(ang)

        vis, sf = self._deco_state(fx)
        alive = np.flatnonzero((self.birth <= t) & vis[self.setcode])
        occupied = np.zeros(COLS * ROWS, dtype=bool)
        if alive.size == 0:
            return None, occupied, 0, 0
        p = self.pos[alive]

        # pop-in scaling (ornaments / sweets / star / gifts grow from their pivot)
        age = t - self.birth[alive]
        pd = self.pop[alive]
        k = np.flatnonzero((pd > 0) & (age < pd))
        if k.size:
            sc = ease_out_back(age[k] / pd[k]).astype(np.float32)
            pv = self.piv[alive[k]]
            p[k] = pv + (p[k] - pv) * sc[:, None]

        # theme swap: decoration set shrinks away / pops in around its pivot
        if fx < self.swap_end:
            sets = self.setcode[alive]
            ks = np.flatnonzero(sets < 2)
            if ks.size:
                pv = self.piv[alive[ks]]
                p[ks] = pv + (p[ks] - pv) * sf[sets[ks]][:, None]

        # rotate around Y, pitch around X, perspective-project
        x = p[:, 0]
        y = p[:, 1] - YC
        z = p[:, 2]
        xr = x * ca + z * sa
        zr = z * ca - x * sa
        y2 = y * CP - zr * SP
        z2 = y * SP + zr * CP
        f = CAM_D / (CAM_D - z2)
        sx = CX + xr * f
        sy = SY0 - y2 * f
        sel = np.flatnonzero((sx >= 0) & (sx < W) & (sy >= 0) & (sy < H))
        if sel.size == 0:
            return None, occupied, 0, 0
        cell = (sy[sel] * INV_CH).astype(np.int32) * COLS + (sx[sel] * INV_CW).astype(np.int32)
        # far -> near; 16-bit keys let numpy use a radix sort (much faster than float argsort)
        zq = ((z2[sel] + 1500.0) * 20.0).astype(np.uint16)
        order = np.argsort(zq, kind="stable")
        win = np.full(COLS * ROWS, -1, dtype=np.int32)
        win[cell[order]] = sel[order]                   # nearest write wins (z-buffer)
        cells = np.flatnonzero(win >= 0)
        occupied[cells] = True
        gid = alive[win[cells]]                         # global ids of the winners

        # ---- lighting for the winners only -----------------------------------
        n = self.nrm[gid]
        nxr = n[:, 0] * ca + n[:, 2] * sa
        nzr = n[:, 2] * ca - n[:, 0] * sa
        ny2 = n[:, 1] * CP - nzr * SP
        nz2 = n[:, 1] * SP + nzr * CP
        diff = np.clip(nxr * L_MAIN[0] + ny2 * L_MAIN[1] + nz2 * L_MAIN[2], 0, 1)
        fill = np.clip(nxr * L_FILL[0] + ny2 * L_FILL[1] + nz2 * L_FILL[2], 0, 1)
        hv = np.clip(nxr * L_HALF[0] + ny2 * L_HALF[1] + nz2 * L_HALF[2], 0, 1) ** 28
        lum = AMBIENT + 0.62 * diff + 0.16 * fill + hv * self.shin[gid]
        lum = lum * self.ao[gid]

        kind = self.kind[gid]
        ph = self.phase[gid]
        agew = t - self.birth[gid]
        fade = np.clip(agew / 0.6, 0.0, 1.0)

        m = kind == K_GARLAND                           # travelling shimmer
        if m.any():
            lum[m] = np.maximum(lum[m], 0.45) * 0.85 + 0.30 * (0.5 + 0.5 * np.sin(t * 5.0 - ph[m]))
        m = kind == K_ORN                               # glint every now and then
        if m.any():
            glint = np.clip(np.sin(t * 2.4 + ph[m]), 0, 1) ** 8
            lum[m] = np.maximum(lum[m], 0.30) + 0.40 * glint
        m = kind == K_SWEET                             # sweets: soft, never too dark
        if m.any():
            lum[m] = np.maximum(lum[m], 0.32)
        m = kind == K_STAR                              # pulsing star
        if m.any():
            lum[m] = 0.72 + 0.28 * np.sin(t * 4.0 + ph[m]) + 0.5 * hv[m]

        lum = np.where(kind == K_FOL, np.maximum(lum, 0.11), lum)   # keep needles from going blank
        flash = ((kind == K_FOL) | (kind == K_GARLAND)) * (0.45 * (1.0 - fade))
        lum = lum * (0.6 + 0.4 * fade) + flash          # growth front sparkles

        lit = 0
        ml = kind == K_LIGHT                            # twinkling fairy lights
        if ml.any():
            b = 0.5 + 0.5 * np.sin(t * self.tw[gid][ml] + ph[ml])
            lum[ml] = (0.10 + 0.90 * b ** 0.8) * fade[ml]
            lit = int((b[fade[ml] > 0.5] > 0.45).sum())

        lum = np.clip(lum, 0.0, 1.0)
        ci = ((1.0 - lum) * SPACE_IDX).astype(np.int32)
        if ml.any():
            ci[ml] = LIGHT_SEQ[np.minimum((lum[ml] * 5.99).astype(np.int32), 5)]

        # sweets: blend glyph cues (o O ~ +) into the shading characters
        cu = self.cue[gid]
        cm = (cu > 0) & (lum > 0.20)
        if cm.any():
            ci[cm] = CUE_IDX[cu[cm]]

        lv = np.minimum((lum * (LV - 1) + 0.5).astype(np.int32), LV - 1)

        draw = (ci < SPACE_IDX) | cm
        letters = int(draw.sum())
        out = (cells[draw], ci[draw], self.mat[gid][draw], lv[draw])
        return out, occupied, lit, letters


# =============================================================================
#  Background snow
# =============================================================================
class Snow:
    COLORS = [(46, 54, 72), (70, 82, 106), (104, 118, 146)]
    CHOICES = [".·", "·+.", "*+·"]

    def __init__(self, rng, n=SNOW_COUNT):
        self.x = rng.uniform(0, COLS, n)
        self.y = rng.uniform(0, ROWS, n)
        self.layer = rng.choice(3, n, p=[0.45, 0.35, 0.20])
        self.vy = 1.5 + self.layer * 2.2 + rng.uniform(0, 1.2, n)    # rows / second
        self.sway = 0.6 + self.layer * 0.5
        self.freq = rng.uniform(0.6, 1.6, n)
        self.ph = rng.uniform(0, 2 * math.pi, n)
        self.chars = [self.CHOICES[l][rng.integers(len(self.CHOICES[l]))]
                      for l in self.layer]

    def update(self, dt, t):
        self.y = (self.y + self.vy * dt) % ROWS
        self.x = (self.x + np.sin(t * self.freq + self.ph) * self.sway * dt * 2.0) % COLS


# =============================================================================
#  Application
# =============================================================================
class App:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        s0 = min(1.0, (info.current_h - 110) / H, (info.current_w - 40) / W)
        s0 = max(s0, 0.3)
        self.screen = pygame.display.set_mode((int(W * s0), int(H * s0)), pygame.RESIZABLE)
        pygame.display.set_caption(
            "3D ASCII Christmas Tree   [Click theme | SPACE pause | R restart | ESC quit]")
        self.canvas = pygame.Surface((W, H)).convert()
        self.font = pygame.font.SysFont(FONT_NAMES, FONT_SIZE)
        self.clock = pygame.time.Clock()

        # themes: one palette per theme, glyph caches per (quantised) palette
        self.palettes = [build_palette(THEMES[k]) for k in THEME_ORDER]
        self.pal_mix = {}                # (from, to, step) -> mixed palette
        self.glyph_caches = {}           # palette key -> {glyph key -> surface}
        self.text_glyphs = {}            # (char, rgb) -> surface
        self.theme_i = 0
        self.prev_i = 0
        self.blend = 1.0                 # 0..1 cross-fade progress (1 = finished)

        print("Building 3D tree ...", end=" ", flush=True)
        self.scene = TreeScene(DECO_SET[THEMES[THEME_ORDER[0]]["decor_style"]])
        print(f"{self.scene.count:,} points")
        self.snow = Snow(np.random.default_rng(7))
        self.snow_surfs = [self._text_glyph(ch, Snow.COLORS[l])      # fixed per particle
                           for ch, l in zip(self.snow.chars, self.snow.layer.tolist())]
        self._scaled = None              # reusable buffer for window scaling

        self.music_ok = self._init_music()
        self.elapsed = 0.0
        self.snow_t = 0.0
        self.fx_t = 0.0                  # UI clock: keeps running while paused
        self.paused = False

    # ---- themes ----------------------------------------------------------------
    def next_theme(self):
        """Cycle to the next theme (colours cross-fade, decoration set swaps)."""
        self.prev_i = self.theme_i
        self.theme_i = (self.theme_i + 1) % len(THEME_ORDER)
        self.blend = 0.0
        theme = THEMES[THEME_ORDER[self.theme_i]]
        self.scene.set_deco(DECO_SET[theme["decor_style"]], self.fx_t)

    def _palette(self):
        """Current palette + its cache key (quantised while cross-fading)."""
        cur, prv = self.theme_i, self.prev_i
        if self.blend < 1.0:
            q = int(self.blend * FADE_STEPS + 0.5)
            if q <= 0:
                return (prv, prv, FADE_STEPS), self.palettes[prv]
            if q < FADE_STEPS:
                key = (prv, cur, q)
                pal = self.pal_mix.get(key)
                if pal is None:
                    a, b, k = self.palettes[prv], self.palettes[cur], q / FADE_STEPS
                    pal = self.pal_mix[key] = [
                        [lerp_color(a[m][l], b[m][l], k) for l in range(LV)]
                        for m in range(NMAT)]
                return key, pal
        return (cur, cur, FADE_STEPS), self.palettes[cur]

    def _tick_fx(self, dt):
        self.fx_t += dt
        if self.blend < 1.0:
            self.blend = min(1.0, self.blend + dt / THEME_FADE)
            if self.blend >= 1.0:        # drop the temporary in-between caches
                for k in [k for k in self.glyph_caches if k[0] != k[1]]:
                    del self.glyph_caches[k]
                self.pal_mix.clear()

    # ---- glyphs ----------------------------------------------------------------
    def _make_glyph(self, ch, rgb):
        g = self.font.render(ch, True, rgb)
        surf = pygame.Surface((CW, CH), pygame.SRCALPHA)
        surf.blit(g, ((CW - g.get_width()) // 2, (CH - g.get_height()) // 2))
        return surf.convert_alpha()

    def _text_glyph(self, ch, rgb):
        key = (ch, rgb)
        s = self.text_glyphs.get(key)
        if s is None:
            s = self.text_glyphs[key] = self._make_glyph(ch, rgb)
        return s

    # ---- music ----------------------------------------------------------------
    def _init_music(self):
        folders = [os.getcwd()]
        try:
            folders.append(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            pass
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            for folder in folders:
                path = os.path.join(folder, MUSIC_FILE)
                if os.path.isfile(path):
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.set_volume(0.6)
                    pygame.mixer.music.play(-1)
                    print(f"Playing {path}")
                    return True
            print(f"{MUSIC_FILE} not found - running without music.")
        except pygame.error as e:
            print(f"Audio unavailable ({e}) - running without music.")
        return False

    def _music(self, action):
        if not self.music_ok:
            return
        try:
            if action == "pause":
                pygame.mixer.music.pause()
            elif action == "resume":
                pygame.mixer.music.unpause()
            elif action == "restart":
                pygame.mixer.music.play(-1)
        except pygame.error:
            pass

    # ---- drawing ----------------------------------------------------------------
    def put_text(self, blits, s, row, col, color):
        for i, ch in enumerate(s):
            if ch != " ":
                blits.append((self._text_glyph(ch, color), ((col + i) * CW, row * CH)))

    def draw_frame(self, t, snow_t, paused=False):
        canvas = self.canvas
        canvas.fill(BG)
        deg = (t * ROT_SPEED) % 360.0
        out, occupied, lit, letters = self.scene.render(t, math.radians(deg), self.fx_t)

        blits = []
        # snow (skipped wherever the tree already occupies the cell)
        sn = self.snow
        sc = np.minimum(sn.x.astype(np.int32), COLS - 1)
        sr = np.minimum(sn.y.astype(np.int32), ROWS - 1)
        free = np.flatnonzero(~occupied[sr * COLS + sc])
        surfs = self.snow_surfs
        blits.extend([(surfs[i], (x, y)) for i, x, y in
                      zip(free.tolist(), (sc[free] * CW).tolist(), (sr[free] * CH).tolist())])

        # tree glyphs (cached surfaces, one blits() call)
        if out is not None:
            cells, ci, mat, lv = out
            key, pal = self._palette()
            gl = self.glyph_caches.setdefault(key, {})
            xs = ((cells % COLS) * CW).tolist()
            ys = ((cells // COLS) * CH).tolist()
            keys = ((ci * NMAT + mat) * LV + lv).tolist()
            try:                                           # fast path: everything cached
                blits.extend([(gl[k], (x, y)) for k, x, y in zip(keys, xs, ys)])
            except KeyError:                               # first sight of some glyphs
                for k, x, y in zip(keys, xs, ys):
                    s = gl.get(k)
                    if s is None:
                        cm_, l = divmod(k, LV)
                        c, m = divmod(cm_, NMAT)
                        s = gl[k] = self._make_glyph(CHARS[c], pal[m][l])
                    blits.append((s, (x, y)))

        # greeting (fades in once the tree is assembled)
        if t > T_GREET:
            a = round(smooth01((t - T_GREET) / T_GREET_FADE) * 10) / 10.0
            col0 = (COLS - len(GREETING)) // 2
            for i, ch in enumerate(GREETING):
                if ch == " ":
                    continue
                sh = round((0.5 + 0.5 * math.sin(t * 2.0 - i * 0.55)) * 7) / 7.0
                base = lerp_color((255, 190, 84), (255, 246, 224), sh * 0.65)
                rgb = lerp_color(BG, base, a * (0.72 + 0.28 * sh))
                blits.append((self._text_glyph(ch, rgb), ((col0 + i) * CW, 71 * CH)))

        # HUD
        top = f"t {t:05.2f}s"
        self.put_text(blits, top, 1, COLS - len(top) - 2, HUD_COLOR)
        status = f"rotation {deg:.2f}\u00b0   lights - {lit}   letters {letters:,}"
        self.put_text(blits, status, ROWS - 1, (COLS - len(status)) // 2, HUD_COLOR)
        self.put_text(blits, "SPACE pause   R restart   ESC quit", 1, 2, HINT_COLOR)

        theme = THEMES[THEME_ORDER[self.theme_i]]          # theme indicator (top right)
        name = f"theme: {theme['label']} ({self.theme_i + 1}/{len(THEME_ORDER)})"
        tint = hex_to_rgb(theme["foliage"][1])
        self.put_text(blits, name, 3, COLS - len(name) - 2,
                      lerp_color(HUD_COLOR, tint, 0.45))
        hint = "[Click] Change Theme"
        self.put_text(blits, hint, 4, COLS - len(hint) - 2, HINT_COLOR)
        if paused:
            self.put_text(blits, "PAUSED", 3, 2, (200, 170, 90))

        canvas.blits(blits, False)

    def present(self):
        screen = pygame.display.get_surface()
        sw, sh = screen.get_size()
        sc = min(sw / W, sh / H)
        if abs(sc - 1.0) < 0.005:
            screen.blit(self.canvas, (0, 0))
        else:
            tw, th = max(1, int(W * sc)), max(1, int(H * sc))
            if self._scaled is None or self._scaled.get_size() != (tw, th):
                self._scaled = pygame.Surface((tw, th)).convert()
            scaled = pygame.transform.smoothscale(self.canvas, (tw, th), self._scaled)
            screen.fill(BG)
            screen.blit(scaled, ((sw - tw) // 2, (sh - th) // 2))
        pygame.display.flip()

    # ---- main loop ----------------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.1)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.next_theme()
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        self._music("pause" if self.paused else "resume")
                    elif ev.key == pygame.K_r:
                        self.elapsed = 0.0
                        self.paused = False
                        self._music("restart")
            self._tick_fx(dt)
            if not self.paused:
                self.elapsed += dt
                self.snow_t += dt
                self.snow.update(dt, self.snow_t)
            self.draw_frame(self.elapsed, self.snow_t, self.paused)
            self.present()
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
    sys.exit(0)