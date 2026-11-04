"""Three-page A4 CLMSynth control-flow figure, the PDF the manual embeds.

    python tools/CLMSynth_engine_decision_map.py [pdf_folder] [png_folder]

The PDF goes to docs/sources/manual/ by default; PNG previews need a png_folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon, Rectangle

import clmsynth.main as clm_main
from clmsynth.clm_errors import CODES

# The release the map documents; it may run ahead of the installed __version__.
VERSION = "0.7.6"
STEM = "engine_decision_map_v" + VERSION.replace(".", "")

# ---------------------------------------------------------------- model ----

PAGES: list[dict[str, Any]] = [
    {
        "id": "p1",
        "title": "CLMSynth · run pipeline",
        "subtitle": "clmsynth <config>  →  YYYYMMDD_HHMMSS_Source/{csv, png, info} · one process per config",
        "note": "CLMSynth v" + VERSION,
        "caption": "Figure A.1 · outer pipeline (1 of 3)",
        "nodes": [
            {
                "id": "load",
                "kind": "stage",
                "title": "load_config()",
                "sub": "JSON, or YAML with PyYAML · data source resolved",
            },
            {
                "id": "rundir",
                "kind": "stage",
                "title": "build_run_dir()",
                "sub": "run folder · csv/ png/ info/ with the run log",
            },
            {
                "id": "select",
                "kind": "stage",
                "title": "resolve_selection()",
                "sub": "batteries × datasets → jobs · resolved config to info/",
            },
            {
                "id": "precheck",
                "kind": "choice",
                "title": "precheck_byoc_matching_ids()",
                "sub": "byoc only · id guards over every CSV",
                "codes": [104, 105, 141],
                "branches": [
                    {
                        "id": "byoc_abort",
                        "side": "term",
                        "label": "if byoc · id clash",
                        "text": "abort before\nany output",
                    }
                ],
            },
            {
                "id": "loop",
                "kind": "panel",
                "title": "for battery, dataset in jobs",
                "repeat": "next dataset",
                "children": [
                    {"id": "fetch", "title": "fetcher(battery, dataset)"},
                    {"id": "ctx", "title": "build_context()"},
                    {
                        "id": "gen",
                        "title": "generate_additional_labels()",
                        "codes": [
                            102,
                            119,
                            120,
                            125,
                            127,
                            133,
                            150,
                            151,
                            152,
                            153,
                            201,
                            202,
                            203,
                            204,
                            205,
                            206,
                            207,
                            208,
                            209,
                        ],
                        "branches": [
                            {
                                "id": "gen_abort",
                                "side": "term",
                                "label": "1xx ValueError",
                                "text": "skip dataset\nor exit 2",
                            },
                            {
                                "id": "gen_bare",
                                "side": "join",
                                "label": "120 · 15x · 2xx",
                                "text": "CSV written\nno Label_n",
                            },
                        ],
                    },
                    {"id": "csv", "title": "to_dataframe().to_csv()"},
                    {"id": "txt", "title": "_write_summary_txt()"},
                    {"id": "png", "title": "_render_dataset_plots()"},
                ],
            },
            {
                "id": "finalize",
                "kind": "choice",
                "title": "n_ok == 0 ?",
                "sub": "a failed run's folder goes, unless --keep-failed",
                "branches": [
                    {
                        "id": "exit1",
                        "side": "term",
                        "label": "yes · nothing written",
                        "text": "exit 1\nfolder removed",
                    },
                ],
            },
            {"id": "exit0", "kind": "stage", "title": "exit 0"},
        ],
    },
    {
        "id": "p2",
        "title": "CLMSynth · clm_label_engine",
        "subtitle": "c(x), X, cfg, seed  →  L(x) ∈ {0 … M−1}, deterministic under seed",
        "note": "CLMSynth v" + VERSION,
        "caption": "Figure A.2 · engine internals (2 of 3)",
        "nodes": [
            {
                "id": "enter",
                "kind": "stage",
                "title": "generate_clm_labels()",
                "sub": "mode checked first · M, K, sizes · input guards",
                "codes": [101, 102, 125, 126, 127, 133, 141, 201, 202],
            },
            {
                "id": "counts",
                "kind": "stage",
                "title": "resolve_label_counts()",
                "sub": "balance, skew → m_counts, Σ = N · not under perfect",
                "codes": [106, 107, 121, 131, 132, 203, 301],
            },
            {
                "id": "mode",
                "kind": "choice",
                "title": "matching_mode",
                "sub": "perfect · single · random · custom",
                "codes": [114, 115, 311],
                "branches": [
                    {
                        "id": "rand",
                        "side": "term",
                        "label": "random",
                        "text": "shuffle counts\nthen return",
                    },
                    {
                        "id": "perf",
                        "side": "join",
                        "label": "perfect",
                        "text": "m_counts ←\ncluster sizes",
                    },
                ],
            },
            {
                "id": "validate",
                "kind": "choice",
                "title": "_validate_*_cfg()",
                "sub": "spillover · centroid · target_metric · split · noise",
                "codes": [
                    108,
                    110,
                    111,
                    112,
                    113,
                    116,
                    117,
                    118,
                    122,
                    123,
                    124,
                    128,
                    129,
                    130,
                    144,
                    209,
                    303,
                    307,
                    308,
                ],
                "branches": [
                    {
                        "id": "pair",
                        "side": "join",
                        "label": "scope: pair",
                        "text": "pair inversion\nα = 1",
                    }
                ],
            },
            {
                "id": "solve",
                "kind": "panel",
                "title": "solve_alpha_for_target_metric()",
                "note": "only when target_metric · scope global",
                "codes": [120, 306],
                "repeat": "scan, then brentq",
                "children": [
                    {"id": "rules", "title": "build_rules(α)"},
                    {"id": "probe", "title": "allocate + assign (probe)"},
                    {"id": "score", "title": "mcc / ari (α)"},
                ],
            },
            {
                "id": "rules",
                "kind": "stage",
                "title": "build_rules(cfg)",
                "sub": "always runs · solved α overrides recall_target",
                "codes": [103, 104, 105, 142, 204, 205, 206, 207, 208, 302],
            },
            {
                "id": "alloc",
                "kind": "stage",
                "title": "_run_allocation_pipeline()",
                "sub": "own RNG stream · expanded on page 3",
            },
            {
                "id": "ret",
                "kind": "stage",
                "title": "verify → return L(x)",
                "sub": "delivered counts · metric vs target · report",
                "codes": [154, 304, 309, 310],
            },
        ],
    },
    {
        "id": "p3",
        "title": "CLMSynth · _run_allocation_pipeline",
        "subtitle": "rules, m_counts, cluster_sizes  →  out(x), one label per row",
        "note": "CLMSynth v" + VERSION,
        "caption": "Figure A.3 · allocation and placement (3 of 3)",
        "nodes": [
            {
                "id": "stream",
                "kind": "stage",
                "title": "_run_allocation_pipeline()",
                "sub": "own RNG stream, restarted from the run seed",
            },
            {
                "id": "allocate",
                "kind": "stage",
                "title": "allocate()",
                "sub": "rules → per-(cluster, label) demand + capacity left",
                "codes": [150, 151, 153],
            },
            {
                "id": "competing",
                "kind": "choice",
                "title": "_competing_demand()",
                "sub": "a share of one cluster's UNCLAIMED points",
                "codes": [119, 152, 305],
                "branches": [
                    {
                        "id": "nonoise",
                        "side": "join",
                        "label": "no competing_noise",
                        "text": "skipped\nentirely",
                    },
                ],
            },
            {
                "id": "spill",
                "kind": "stage",
                "title": "_spillover_draws()",
                "sub": "leftover budget → label draws, per spillover_rule",
                "codes": [109],
            },
            {
                "id": "place",
                "kind": "stage",
                "title": "assign_points_in_cluster()",
                "sub": "targets, then competing noise, then spillover",
            },
            {
                "id": "out",
                "kind": "stage",
                "title": "→ out(x)",
            },
        ],
    },
]

LEGEND = [
    [("arrow", "continues"), ("tbar", "stops here"), ("rule", "branches")],
    [
        ("grey", "run or dataset stops"),
        ("white", "returns to the spine"),
        ("dash", "one step, expanded"),
    ],
]

BAND_KEY = "1xx invalid config · 15x infeasible allocation · 2xx missing key · 3xx warning"
GUTTER_HEAD = "diagnostics [CLM-###]"

# ------------------------------------------------------------- geometry ----

PAGE_W, PAGE_H = 595.276, 841.890
L, R = 57.638, 537.638
COL = 40.0
X_SPINE = 237.638

TRK_TERM = (57.638, 117.638)
TRK_SPINE = (137.638, 337.638)
TRK_JOIN = (357.638, 417.638)
GUT_X, GUT_CODE_X = 437.638, 455.638

Y_TITLE = 765.890
Y_SUBTITLE = 751.890
Y_RULE_TOP = 741.890
Y_GUTTER_HEAD = 731.890
Y_FLOW_TOP = 723.890
Y_FLOW_BOT = 145.890
Y_RULE_BOT = 133.890
Y_LEG_A = 119.890
Y_LEG_B = 95.890
Y_CAPTION = 71.890
FLOW_H = Y_FLOW_TOP - Y_FLOW_BOT

GAP_MIN = 16.0
H_NODE_1, H_NODE_2 = 36.0, 48.0
NODE_W = TRK_SPINE[1] - TRK_SPINE[0]
NODE_PAD = 8.0
LINE_GAP, CAP_MID = 12.0, 2.6
CHOICE_RULE_W, CHOICE_RULE_INSET = 2.4, 1.2

PANEL_X = (L, TRK_JOIN[1])
PANEL_PAD_TOP, PANEL_PAD_BOT, PANEL_PAD_SIDE = 18.0, 10.0, 10.0
PANEL_TITLE_DROP = 12.0
MINI_W, MINI_H, MINI_GAP = 160.0, 20.0, 10.0
MINI_TRK_TERM = (67.638, 117.638)
MINI_TRK_JOIN = (357.638, 407.638)
REPEAT_INSET = 6.0

SIDE_W, SIDE_H = 60.0, 28.0
MINI_SIDE_W, MINI_SIDE_H = 50.0, 20.0
BR_DROP, BR_ELBOW, SLOT_PITCH = 12.0, 8.0, 48.0
MINI_BR_DROP, MINI_BR_ELBOW = 7.0, 5.0
LABEL_LIFT, LABEL_CLEAR = 3.5, 4.0
TBAR_CLEAR, TBAR_LEN, TBAR_W = 5.0, 12.0, 1.2
ARROW_LEN, ARROW_W = 5.0, 3.6

GUT_LEAD, GUT_FIRST = 8.5, 11.0
T3_LEAD, SIDE_PAD = 8.0, 3.0
LEG_GLYPH, LEG_TEXT_X = 18.0, 28.0
MAX_EDGE = 220.0

LW_SPINE, LW_NODE, LW_BRANCH, LW_MINI, LW_PANEL, LW_HAIR = 1.0, 0.8, 0.7, 0.7, 0.6, 0.5

# -------------------------------------------------------------- palette ----

PAPER = "#FFFFFF"
INK = "#1A1A1A"
MID = "#595959"
HAIR = "#BFBFBF"
TERM_FILL = "#F2F2F2"
BANDS = {"1xx": "#994455", "15x": "#004488", "2xx": "#4D4D4D", "3xx": "#8A6D00"}
BAND_ORDER = ("1xx", "15x", "2xx", "3xx")

SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
TOKENS: dict[str, dict[str, Any]] = {
    "T1": {"size": 9.0, "family": SANS, "weight": "bold", "color": INK, "adv": 0.52},
    "T2": {"size": 7.5, "family": MONO, "weight": "normal", "color": INK, "adv": 0.6023},
    "T2m": {"size": 7.5, "family": MONO, "weight": "normal", "color": MID, "adv": 0.6023},
    "T3": {"size": 6.5, "family": SANS, "weight": "normal", "color": MID, "adv": 0.52},
    "T3m": {"size": 6.5, "family": MONO, "weight": "normal", "color": MID, "adv": 0.6023},
}


def text_w(s, token):
    """Advance width of a string."""
    t = TOKENS[token]
    return len(s) * t["size"] * t["adv"]


def band_of(code):
    """Band tag for a code number."""
    if code < 150:
        return "1xx"
    if code < 200:
        return "15x"
    if code < 300:
        return "2xx"
    return "3xx"


def skip_codes():
    """Live skip set read from main.py."""
    return set(clm_main.PER_DATASET_CODES)


# --------------------------------------------------------------- layout ----


def _box(scene, bid, x, y, w, h, kind, parent=None, fill=PAPER, edge=INK, lw=LW_NODE):
    scene["boxes"][bid] = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "kind": kind,
        "parent": parent,
        "fill": fill,
        "edge": edge,
        "lw": lw,
    }
    return scene["boxes"][bid]


def _text(scene, x, y, s, token, ha="left", owner=None, color=None):
    w = text_w(s, token)
    t = TOKENS[token]
    x0 = x if ha == "left" else (x - w if ha == "right" else x - w / 2.0)
    scene["texts"].append(
        {
            "x": x,
            "y": y,
            "s": s,
            "token": token,
            "ha": ha,
            "owner": owner,
            "color": color or t["color"],
            "x0": x0,
            "x1": x0 + w,
            "y0": y - 0.22 * t["size"],
            "y1": y + 0.75 * t["size"],
        }
    )


def _line(scene, pts, lw, color, dash=None):
    scene["lines"].append({"pts": pts, "lw": lw, "color": color, "dash": dash})


def _arrow(scene, tip, dirn, color):
    scene["arrows"].append({"tip": tip, "dirn": dirn, "color": color})


def _edge(scene, pts, lw, color, tip_dir, label=None, label_at=None, owner=None):
    _line(scene, pts, lw, color)
    _arrow(scene, pts[-1], tip_dir, color)
    length = sum(
        abs(pts[i + 1][0] - pts[i][0]) + abs(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)
    )
    scene["edges"].append({"pts": pts, "len": length, "owner": owner})
    if label:
        x, y, ha = label_at
        _text(scene, x, y, label, "T3", ha=ha)


def node_h(node):
    """Height of one spine element."""
    if node["kind"] == "panel":
        n = len(node["children"])
        return PANEL_PAD_TOP + n * MINI_H + (n - 1) * MINI_GAP + PANEL_PAD_BOT
    return H_NODE_2 if node.get("sub") else H_NODE_1


def _branches(scene, node, box, bid, level, gap):
    """Place one node's side branches."""
    if level == "outer":
        drop, elbow, sw, sh = BR_DROP, BR_ELBOW, SIDE_W, SIDE_H
        term_trk, join_trk = TRK_TERM, TRK_JOIN
    else:
        drop, elbow, sw, sh = MINI_BR_DROP, MINI_BR_ELBOW, MINI_SIDE_W, MINI_SIDE_H
        term_trk, join_trk = MINI_TRK_TERM, MINI_TRK_JOIN
    top = box["y"] + box["h"]
    used = {"term": 0, "join": 0}
    for br in node.get("branches", []):
        term = br["side"] == "term"
        y_h = top - drop - used[br["side"]] * SLOT_PITCH
        used[br["side"]] += 1
        if y_h < box["y"]:
            raise SystemExit(f"build failed: branch {br['id']} leaves outside {bid}")
        trk = term_trk if term else join_trk
        sx = trk[1] - sw if term else trk[0]
        scx = sx + sw / 2.0
        sy = y_h - elbow - sh
        sbid = f"{bid}.{br['id']}"
        _box(
            scene,
            sbid,
            sx,
            sy,
            sw,
            sh,
            "side",
            parent=box["parent"] or bid,
            fill=TERM_FILL if term else PAPER,
            edge=MID,
            lw=LW_BRANCH,
        )
        edge_x = box["x"] if term else box["x"] + box["w"]
        _edge(
            scene,
            [(edge_x, y_h), (scx, y_h), (scx, sy + sh)],
            LW_BRANCH,
            MID,
            "down",
            label=br["label"],
            label_at=(
                edge_x - LABEL_CLEAR if term else edge_x + LABEL_CLEAR,
                y_h + LABEL_LIFT,
                "right" if term else "left",
            ),
        )
        lines = br["text"].split("\n")
        base = sy + sh / 2.0 + (len(lines) - 1) * T3_LEAD / 2.0 - CAP_MID
        for i, ln in enumerate(lines):
            _text(scene, scx, base - i * T3_LEAD, ln, "T3", ha="center", owner=sbid)
        if term:
            bx = sx + sw + TBAR_CLEAR
            _line(
                scene,
                [(bx, sy + sh / 2.0 - TBAR_LEN / 2.0), (bx, sy + sh / 2.0 + TBAR_LEN / 2.0)],
                TBAR_W,
                MID,
            )
        else:
            if level == "outer":
                y_land = min(box["y"] - gap / 2.0, sy - BR_ELBOW)
                _edge(scene, [(scx, sy), (scx, y_land), (X_SPINE, y_land)], LW_BRANCH, MID, "left")
            else:
                y_land = sy + sh / 2.0
                _edge(scene, [(sx, y_land), (X_SPINE, y_land)], LW_BRANCH, MID, "left")


def _gutter(scene, codes, y_top):
    """Emit one node's diagnostic rows."""
    if not codes:
        return
    per_line = int((R - GUT_CODE_X + text_w(" ", "T3m")) // text_w("000 ", "T3m"))
    y = y_top - GUT_FIRST
    for tag in BAND_ORDER:
        got = sorted(c for c in codes if band_of(c) == tag)
        if not got:
            continue
        _text(scene, GUT_X, y, tag, "T3m", owner="gutter", color=BANDS[tag])
        for i in range(0, len(got), per_line):
            chunk = " ".join(str(c) for c in got[i : i + per_line])
            _text(scene, GUT_CODE_X, y, chunk, "T3m", owner="gutter", color=BANDS[tag])
            y -= GUT_LEAD


def _panel(scene, node, box, gap):
    """Lay out one expansion panel."""
    pid = node["id"]
    _text(
        scene,
        box["x"] + PANEL_PAD_SIDE,
        box["y"] + box["h"] - PANEL_TITLE_DROP,
        node["title"],
        "T2m",
        owner=pid,
    )
    if node.get("note"):
        _text(
            scene,
            box["x"] + box["w"] - PANEL_PAD_SIDE,
            box["y"] + box["h"] - PANEL_TITLE_DROP,
            node["note"],
            "T3",
            ha="right",
            owner=pid,
        )
    _gutter(scene, node.get("codes", []), box["y"] + box["h"])
    tops = []
    kids = node["children"]
    y = box["y"] + box["h"] - PANEL_PAD_TOP
    for idx, child in enumerate(kids):
        cid = f"{pid}.{child['id']}"
        cbox = _box(
            scene,
            cid,
            X_SPINE - MINI_W / 2.0,
            y - MINI_H,
            MINI_W,
            MINI_H,
            "mini",
            parent=pid,
            lw=LW_MINI,
        )
        _text(
            scene, X_SPINE, y - MINI_H / 2.0 - CAP_MID, child["title"], "T2", ha="center", owner=cid
        )
        _gutter(scene, child.get("codes", []), y)
        _branches(scene, child, cbox, cid, "mini", MINI_GAP)
        tops.append(y)
        if idx + 1 < len(kids):
            nxt = y - MINI_H - MINI_GAP
            _edge(scene, [(X_SPINE, y - MINI_H), (X_SPINE, nxt)], LW_MINI, INK, "down")
            if kids[idx + 1].get("label"):
                _text(
                    scene,
                    X_SPINE + LABEL_CLEAR,
                    nxt + (MINI_GAP - LABEL_LIFT) / 2.0,
                    kids[idx + 1]["label"],
                    "T3",
                )
        y -= MINI_H + MINI_GAP
    last_bot = tops[-1] - MINI_H
    if node.get("repeat"):
        xr = box["x"] + box["w"] - REPEAT_INSET
        y_bot = last_bot - BR_ELBOW
        y_first = tops[0] - MINI_H / 2.0
        _edge(
            scene,
            [
                (X_SPINE, last_bot),
                (X_SPINE, y_bot),
                (xr, y_bot),
                (xr, y_first),
                (X_SPINE + MINI_W / 2.0, y_first),
            ],
            LW_BRANCH,
            MID,
            "left",
            owner="repeat",
        )
        _text(scene, X_SPINE + MINI_W / 2.0 + LABEL_CLEAR, y_bot + LABEL_LIFT, node["repeat"], "T3")


def layout(page):
    """Coordinates for one page."""
    scene = {"boxes": {}, "texts": [], "lines": [], "arrows": [], "edges": [], "page": page}
    nodes = page["nodes"]
    hs = [node_h(n) for n in nodes]
    gap = (FLOW_H - sum(hs)) / (len(nodes) - 1)
    scene["gap"] = gap

    _text(scene, L, Y_TITLE, page["title"], "T1")
    _text(scene, L, Y_SUBTITLE, page["subtitle"], "T3")
    _text(scene, R, Y_SUBTITLE, page["note"], "T3", ha="right")
    _line(scene, [(L, Y_RULE_TOP), (R, Y_RULE_TOP)], LW_HAIR, HAIR)
    _line(scene, [(L, Y_RULE_BOT), (R, Y_RULE_BOT)], LW_HAIR, HAIR)
    _text(scene, GUT_X, Y_GUTTER_HEAD, GUTTER_HEAD, "T3m", owner="gutter")
    _text(scene, L, Y_CAPTION, page["caption"], "T3")
    _text(scene, R, Y_CAPTION, BAND_KEY, "T3", ha="right")

    y = Y_FLOW_TOP
    for node, h in zip(nodes, hs, strict=True):
        nid = node["id"]
        if node["kind"] == "panel":
            box = _box(
                scene,
                nid,
                PANEL_X[0],
                y - h,
                PANEL_X[1] - PANEL_X[0],
                h,
                "panel",
                fill="none",
                edge=MID,
                lw=LW_PANEL,
            )
            _panel(scene, node, box, gap)
        else:
            box = _box(scene, nid, TRK_SPINE[0], y - h, NODE_W, h, node["kind"])
            rows = [(node["title"], "T2")]
            if node.get("sub"):
                rows.append((node["sub"], "T3"))
            base = y - (h - (len(rows) - 1) * LINE_GAP) / 2.0 - CAP_MID
            for i, (line, tok) in enumerate(rows):
                _text(scene, X_SPINE, base - i * LINE_GAP, line, tok, ha="center", owner=nid)
            _gutter(scene, node.get("codes", []), y)
            _branches(scene, node, box, nid, "outer", gap)
        if node is not nodes[-1]:
            top_next = y - h - gap
            inset = PANEL_PAD_TOP if nodes[nodes.index(node) + 1]["kind"] == "panel" else 0.0
            start = y - h + (PANEL_PAD_BOT if node["kind"] == "panel" else 0.0)
            _edge(scene, [(X_SPINE, start), (X_SPINE, top_next - inset)], LW_SPINE, INK, "down")
        y -= h + gap
    return scene


def legend_scene(scene):
    """Legend and its glyph specimens."""
    for row, y in zip(LEGEND, (Y_LEG_A, Y_LEG_B), strict=True):
        for i, (glyph, label) in enumerate(row):
            x = L + i * 4 * COL
            scene["legend"].append({"glyph": glyph, "x": x, "y": y})
            _text(scene, x + LEG_TEXT_X, y, label, "T3")


# --------------------------------------------------------------- verify ----


def _overlap(a, b):
    return (
        a["x"] < b["x"] + b["w"]
        and b["x"] < a["x"] + a["w"]
        and a["y"] < b["y"] + b["h"]
        and b["y"] < a["y"] + a["h"]
    )


def _ancestors(scene, bid):
    out = set()
    cur = scene["boxes"].get(bid, {}).get("parent")
    while cur:
        out.add(cur)
        cur = scene["boxes"].get(cur, {}).get("parent")
    return out


def _segments(scene):
    for ln in scene["lines"]:
        pts = ln["pts"]
        for i in range(len(pts) - 1):
            yield pts[i], pts[i + 1]


def _crossings(scene):
    """Strictly interior connector intersections."""
    segs = list(_segments(scene))
    hor = [(min(a[0], b[0]), max(a[0], b[0]), a[1]) for a, b in segs if a[1] == b[1]]
    ver = [(min(a[1], b[1]), max(a[1], b[1]), a[0]) for a, b in segs if a[0] == b[0]]
    eps = 0.75
    for x0, x1, y in hor:
        for y0, y1, x in ver:
            if x0 + eps < x < x1 - eps and y0 + eps < y < y1 - eps:
                yield (x0, y), (x, y0)


def verify(scene, page):
    """Assert the page is drawable and complete."""
    bad = []
    boxes = scene["boxes"]
    if scene["gap"] < GAP_MIN - 1e-6:
        bad.append(f"flow overfull: gap {scene['gap']:.2f} < {GAP_MIN}")
    ids = sorted(boxes)
    for i, a in enumerate(ids):
        ba = boxes[a]
        if ba["x"] < L - 1e-6 or ba["x"] + ba["w"] > R + 1e-6:
            bad.append(f"{a} leaves the margin box")
        if ba["y"] + ba["h"] > Y_RULE_TOP + 1e-6 or ba["y"] < Y_RULE_BOT - 1e-6:
            bad.append(f"{a} crosses a page rule")
        for b in ids[i + 1 :]:
            if b in _ancestors(scene, a) or a in _ancestors(scene, b):
                continue
            if _overlap(ba, boxes[b]):
                bad.append(f"boxes overlap: {a} / {b}")
    for t in scene["texts"]:
        if t["owner"] == "gutter":
            continue
        skip = {t["owner"]} | _ancestors(scene, t["owner"])
        for bid, bx in boxes.items():
            if bid in skip or bx["kind"] == "panel":
                continue
            if (
                t["x0"] < bx["x"] + bx["w"]
                and bx["x"] < t["x1"]
                and t["y0"] < bx["y"] + bx["h"]
                and bx["y"] < t["y1"]
            ):
                bad.append(f"text {t['s']!r} overlaps box {bid}")
    txt = scene["texts"]
    for i, a in enumerate(txt):
        for b in txt[i + 1 :]:
            if a["x0"] < b["x1"] and b["x0"] < a["x1"] and a["y0"] < b["y1"] and b["y0"] < a["y1"]:
                bad.append(f"text collides: {a['s']!r} / {b['s']!r}")
    for t in scene["texts"]:
        if t["x0"] < L - 1e-6 or t["x1"] > R + 1e-6:
            bad.append(f"text {t['s']!r} leaves the margin box")
        own = boxes.get(t["owner"])
        pad = SIDE_PAD if own and own["kind"] == "side" else NODE_PAD
        if own and t["x1"] - t["x0"] > own["w"] - 2 * pad + 1e-6:
            bad.append(f"text {t['s']!r} wider than {t['owner']}")
    for t in scene["texts"]:
        if t["owner"] == "gutter" or t["y1"] > Y_RULE_TOP or t["y0"] < Y_RULE_BOT:
            continue
        if t["x1"] > TRK_JOIN[1] + 1e-6:
            bad.append(f"text {t['s']!r} enters the quiet zone")
    for bx in boxes.values():
        if bx["x"] + bx["w"] > TRK_JOIN[1] + 1e-6:
            bad.append("a box enters the quiet zone")
    for h, v in _crossings(scene):
        bad.append(f"connectors cross near ({v[0]:.0f}, {h[1]:.0f})")
    for e in scene["edges"]:
        if e["owner"] != "repeat" and e["len"] > MAX_EDGE:
            bad.append(f"edge of {e['len']:.0f} pt exceeds {MAX_EDGE}")
    seen = []
    for node in page["nodes"]:
        seen += node.get("codes", [])
        for child in node.get("children", []):
            seen += child.get("codes", [])
    dupes = {c for c in seen if seen.count(c) > 1}
    if dupes:
        bad.append(f"codes repeated on {page['id']}: {sorted(dupes)}")
    unknown = [c for c in seen if c not in CODES]
    if unknown:
        bad.append(f"codes absent from clm_errors.CODES: {unknown}")
    if bad:
        raise SystemExit("build failed:\n  " + "\n  ".join(bad))
    return set(seen)


def verify_registry(drawn):
    """Whole-registry and dispatch cross-checks."""
    bad = []
    missing = sorted(set(CODES) - drawn)
    if missing:
        bad.append(f"codes in clm_errors.CODES drawn nowhere: {missing}")
    skips = skip_codes()
    p1 = {c for n in PAGES[0]["nodes"] for c in n.get("codes", [])}
    p1 |= {
        c for n in PAGES[0]["nodes"] for ch in n.get("children", []) for c in ch.get("codes", [])
    }
    if not skips <= p1:
        bad.append(f"main.py skips {sorted(skips - p1)}, not shown on page 1")
    if bad:
        raise SystemExit("build failed:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------- render ----


def new_page():
    """Blank A4 portrait axes in points."""
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    return fig, ax


def draw_arrow(ax, tip, dirn, color):
    """Filled triangle at an edge end."""
    x, y = tip
    if dirn == "down":
        pts = [(x, y), (x - ARROW_W / 2.0, y + ARROW_LEN), (x + ARROW_W / 2.0, y + ARROW_LEN)]
    elif dirn == "left":
        pts = [(x, y), (x + ARROW_LEN, y - ARROW_W / 2.0), (x + ARROW_LEN, y + ARROW_W / 2.0)]
    else:
        pts = [(x, y), (x - ARROW_LEN, y - ARROW_W / 2.0), (x - ARROW_LEN, y + ARROW_W / 2.0)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=4))


def draw_box(ax, box):
    """One node rectangle."""
    ax.add_patch(
        Rectangle(
            (box["x"], box["y"]),
            box["w"],
            box["h"],
            facecolor=box["fill"],
            edgecolor=box["edge"],
            linewidth=box["lw"],
            zorder=2,
            linestyle=(0, (2.5, 2.5)) if box["kind"] == "panel" else "solid",
        )
    )
    if box["kind"] == "choice":
        ax.add_patch(
            Rectangle(
                (box["x"] + CHOICE_RULE_INSET, box["y"]),
                CHOICE_RULE_W,
                box["h"],
                facecolor=INK,
                edgecolor="none",
                zorder=3,
            )
        )


def draw_legend(ax, item):
    """One legend glyph specimen."""
    g, x, y = item["glyph"], item["x"], item["y"]
    mid_y = y + 2.0
    if g == "arrow":
        ax.plot([x, x + LEG_GLYPH - ARROW_LEN], [mid_y, mid_y], color=INK, lw=LW_SPINE)
        draw_arrow(ax, (x + LEG_GLYPH, mid_y), "right", INK)
    elif g == "tbar":
        ax.plot([x, x + LEG_GLYPH - TBAR_CLEAR], [mid_y, mid_y], color=MID, lw=LW_BRANCH)
        ax.plot(
            [x + LEG_GLYPH, x + LEG_GLYPH],
            [mid_y - TBAR_LEN / 2.0, mid_y + TBAR_LEN / 2.0],
            color=MID,
            lw=TBAR_W,
        )
    elif g in ("rule", "grey", "white", "dash"):
        h = 12.0
        ax.add_patch(
            Rectangle(
                (x, mid_y - h / 2.0),
                LEG_GLYPH,
                h,
                facecolor=TERM_FILL if g == "grey" else PAPER,
                edgecolor=MID if g in ("grey", "white", "dash") else INK,
                linewidth=LW_BRANCH,
                linestyle=(0, (2.5, 2.5)) if g == "dash" else "solid",
            )
        )
        if g == "rule":
            ax.add_patch(
                Rectangle(
                    (x + CHOICE_RULE_INSET, mid_y - h / 2.0),
                    CHOICE_RULE_W,
                    h,
                    facecolor=INK,
                    edgecolor="none",
                )
            )


def render(ax, scene):
    """Draw a laid-out page."""
    for ln in scene["lines"]:
        xs = [p[0] for p in ln["pts"]]
        ys = [p[1] for p in ln["pts"]]
        ax.plot(
            xs,
            ys,
            color=ln["color"],
            lw=ln["lw"],
            solid_capstyle="butt",
            linestyle=ln["dash"] or "solid",
            zorder=1,
        )
    for box in scene["boxes"].values():
        draw_box(ax, box)
    for a in scene["arrows"]:
        draw_arrow(ax, a["tip"], a["dirn"], a["color"])
    for item in scene["legend"]:
        draw_legend(ax, item)
    for t in scene["texts"]:
        tok = TOKENS[t["token"]]
        ax.text(
            t["x"],
            t["y"],
            t["s"],
            color=t["color"],
            fontsize=tok["size"],
            family=tok["family"],
            fontweight=tok["weight"],
            ha=t["ha"],
            va="baseline",
            zorder=5,
            # matplotlib 3.11 shapes "fi" into a ligature with no code point and
            # writes it as CID 0 (.notdef), which fails PDF/A-2b 6.2.11.8.
            fontfeatures=("-liga",),
        )


# ------------------------------------------------------------------ cli ----


def build(out_dir, png_dir=None):
    """Render every page to PDF, and to PNG previews when png_dir is given."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.compression"] = 6
    matplotlib.rcParams["font.family"] = SANS
    out_dir.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{STEM}.pdf"
    drawn = set()
    figs = []
    for page in PAGES:
        scene = layout(page)
        scene["legend"] = []
        legend_scene(scene)
        drawn |= verify(scene, page)
        fig, ax = new_page()
        render(ax, scene)
        figs.append((page["id"], fig))
    verify_registry(drawn)
    with PdfPages(pdf_path) as pdf:
        for _, fig in figs:
            pdf.savefig(fig, pad_inches=0)
    for pid, fig in figs:
        if png_dir is not None:
            fig.savefig(png_dir / f"{STEM}_{pid}.png", dpi=200, pad_inches=0)
        plt.close(fig)
    return pdf_path


def main(argv):
    """Entry point."""
    out_dir = Path(argv[1]) if len(argv) > 1 else REPO / "docs" / "sources" / "manual"
    png_dir = Path(argv[2]) if len(argv) > 2 else None
    pdf_path = build(out_dir, png_dir)
    for code in sorted(CODES):
        print(f"  CLM-{code} [{band_of(code)}] {CODES[code].splitlines()[0][:58]}")
    previews = f" + {len(PAGES)} PNG previews" if png_dir else ""
    print(f"{len(CODES)} codes placed. wrote {pdf_path.name}{previews}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
