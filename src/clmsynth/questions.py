# questions.py
"""The wizard's questions, as data.

This module holds the question set the CLI wizard asks, one :class:`Question`
per prompt, separated from the wizard that reads them. Two consumers:

* ``config_wizard.py`` pulls each question by key and drives it through the
  terminal prompt helpers. The wizard is a *deletable* convenience, a config
  creator and nothing more; removing it leaves CLMSynth fully functional from a
  config file. The schema, being the reusable half, stays.
* a future in-package ``help`` command, which iterates the schema to surface the
  same ``explain`` text without asking anything.

**Plain data only.** No pandas, matplotlib or seaborn is imported here, and none
must be: a front end or help command that imports this module should pay for
nothing but the dataclass. The dataset-registry lookups the wizard needs stay in
``config_wizard.py`` against ``dataset_sources``; they are dynamic and belong with
the control flow, not here.

**Ranges are carried, not hard-coded at a call site.** Two kinds of bound:

* ``lo`` / ``hi`` (and ``lo_strict``) are the *UI* bounds the prompt helper
  enforces while re-asking.
* ``engine_min`` / ``engine_max`` (and ``engine_min_strict``) name the constraint
  the engine's own validator owns for that value. Where both are present they
  must agree; a UI bound *wider* than the engine's is a config the wizard builds
  happily and the engine refuses, which is the outcome the wizard exists to
  prevent. ``tests/test_07_text_wizard.py`` asserts the agreement by probing just
  outside every ``engine_*`` bound and requiring the engine to reject it.

The two target-metric ``value`` bounds are deliberately UI-only (no ``engine_*``):
MCC and ARI are mathematically defined on ``[-1, 1]``, but a negative *target* is
something the program does not promise to solve for, so the wizard floors it at
``0.0`` while the engine is left exactly as it is.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["str", "int", "float", "bool", "choice",
               "int_list", "float_list", "str_list", "selection", "id", "ids"]


@dataclass(frozen=True)
class Question:
    """One wizard question.

    ``key`` is the dotted path of the value in the assembled config (e.g.
    ``"clm_label.skew_params.alpha"``); control-flow questions whose answer is
    not itself written to the config use a ``_``-prefixed leaf (e.g.
    ``"clm_label.target_metric._enabled"``).

    ``visible_when`` receives the ``clm_label`` sub-config as built so far and
    returns whether the value is *live*. It is the same predicate three places
    share: whether the wizard asks whether ``resolve_label_counts`` consumes the
    value, and whether ``_validate_skew_cfg`` checks it.
    """

    key: str
    kind: Kind
    prompt: str
    explain: str | None = None
    default: Any = None
    lo: float | None = None            # UI lower bound (for ints: the minimum)
    hi: float | None = None            # UI upper bound (for ints: the maximum)
    lo_strict: bool = False            # UI rejects a value equal to lo
    choices: tuple[str, ...] | None = None
    # The engine's own constraint on this value, where it has one. The agreement
    # property probes just outside whichever side is not None.
    engine_min: float | None = None
    engine_max: float | None = None
    engine_min_strict: bool = False    # engine rejects a value equal to engine_min
    visible_when: Callable[[dict], bool] = field(default=lambda clm: True)


# Skew parameters are only consumed when the split is unbalanced, no explicit
# proportions were given, and the owning skew_rule is selected. This mirrors the
# early return in _validate_skew_cfg exactly.
def _skew_live(rule: str) -> Callable[[dict], bool]:
    def pred(clm: dict) -> bool:
        return (clm.get("balance") == "unbalanced"
                and not clm.get("proportions")
                and clm.get("skew_rule") == rule)
    return pred


_QUESTIONS: list[Question] = [
    # -- 1. Data source -----------------------------------------------------
    Question(
        "global_settings.data_source", "choice", "Data source",
        explain="Choose where the base clusters come from:\n"
                "  clustbench      : real benchmark datasets downloaded online (Gagolewski suite)\n"
                "  mdcgen          : synthetic clusters generated on the fly (needs the mdcgenpy package)\n"
                "  fabricated_data : tiny offline example data (no internet, no extra packages)\n"
                "  byoc            : your OWN CSV file (features + one cluster-id column)",
        default="clustbench",
        choices=("clustbench", "mdcgen", "fabricated_data", "byoc")),
    Question(
        "global_settings.output_dir", "str", "Where to save results (folder)",
        explain="Where results go. Usually just press Enter to keep 'OUTPUT';\n"
                "each run still gets its own timestamped subfolder inside it.\n"
                "(Pick a plain folder name, not the same name as a file.)",
        default="OUTPUT"),

    # -- 1b. byoc suite -----------------------------------------------------
    Question(
        "byoc_suite.input_dir", "str", "Folder that holds your CSV file(s)",
        explain="Bring-your-own-clusters: point at your CSV(s).", default="INPUT"),
    Question(
        "byoc_suite.datasets", "str_list", "CSV file name(s) WITHOUT '.csv'",
        explain="The file stem only, no folder and no '.csv'. A stem must not\n"
                "contain '/', '\\' or ':', since the pipeline builds both the input\n"
                "and the output paths from it. List several to run them in one go."),
    Question(
        "byoc_suite.cluster_column", "str", "Name of the single cluster-id column",
        explain="Exactly ONE column holds the cluster id of each row.\n"
                "Every other numeric column is treated as a feature.",
        default="cluster"),
    Question(
        "byoc_suite.standardize", "bool", "Rescale features to 0..1 on import?",
        explain="Min-max standardization puts every feature on the same 0..1 scale.\n"
                "Use it when your features have very different units/ranges so no\n"
                "single one dominates the geometry. It rescales the saved CSV too.",
        default=False),
    Question(
        "byoc_suite.seed", "int", "Random seed",
        explain="Same seed -> same result every run.", default=42),

    # -- 1c. registry suite (the battery picker stays procedural in the wizard) -
    Question(
        "registry_suite.datasets", "selection",
        "Pick dataset(s) by number/name, comma-separated, or 'all'",
        explain="The numbered list above is this group's datasets. Give numbers,\n"
                "names, or a mix of both; 'all' takes the whole group, which can be\n"
                "hundreds of datasets in the larger ones.",
        default="all"),
    Question(
        "registry_suite.seed", "int", "Random seed",
        explain="Used by mdcgen/fabricated_data; ignored by clustbench.", default=42),

    # -- 2. Label generation ------------------------------------------------
    Question(
        "label_generation.n_labels", "int", "How many synthetic labels to generate",
        explain="Each makes one Label_0, Label_1, ... column (a different random\n"
                "seed each), so you can compare several labellings of the same data.",
        default=1, lo=1),
    Question(
        "label_generation.source_labeling", "str", "Which ground-truth labeling to match",
        explain="Most datasets have 'labels0'. Leave as-is unless you "
                "know there are more (e.g. 'labels1').",
        default="labels0"),
    Question(
        "label_generation.seed", "int", "Random seed for label generation", default=42),

    # -- 3. Cluster-label matching ------------------------------------------
    Question(
        "clm_label.num_classes", "int", "Number of labels (M)",
        explain="How many DIFFERENT label values to create. This is NOT the total number\n"
                "of labels, every datapoint always gets a label, it's how many classes\n"
                "they split into (e.g. 3 -> values 0, 1, 2), sized to your whole dataset.\n"
                "At least 2: one label for everything has no matching to measure, and\n"
                "some modes/skews are undefined at M=1.",
        default=3, lo=2),
    Question(
        "clm_label.matching_mode", "choice", "Matching mode",
        explain="How should the new label relate to your clusters?\n"
                "  perfect : an exact copy, one label per cluster (score MCC = 1.0).\n"
                "            Needs M = your cluster count.\n"
                "  single  : ONE cluster becomes one label; all other points are unrelated.\n"
                "  random  : the label ignores the clusters completely (MCC ~ 0), a baseline.\n"
                "  custom  : you write rules (send a share of a label into chosen clusters).\n"
                "            Use for partial/realistic agreement, or to aim at a target score.",
        default="custom", choices=("perfect", "single", "random", "custom")),
    Question(
        "clm_label.target_metric._enabled", "bool",
        "Aim for a specific agreement score instead of setting it by hand?",
        explain="Normally YOU set how much the label matches (recall, below).\n"
                "Say yes to instead name a target MCC/ARI and let the tool solve\n"
                "for it. Caveat: a target can be impossible for your data (e.g.\n"
                "MCC=1 with fewer labels than clusters), then it gets as close\n"
                "as it can and tells you it fell short.\n"
                "Note: the spillover and competing-noise choices you make later\n"
                "are held fixed and still shape the recall the solver lands on.",
        default=False),
    Question(
        "clm_label.target_metric.type", "choice", "Target score",
        explain="Both measure agreement (1 = identical, 0 = unrelated);\n"
                "either is fine, mcc is the more common choice here.",
        default="mcc", choices=("mcc", "ari")),
    # UI-only floor at 0.0: a negative MCC/ARI target is meaningful but not
    # something the program promises to solve for, so it is guarded here and the
    # engine is left untouched (hence no engine_min/engine_max).
    Question(
        "clm_label.target_metric.value", "float", "Target value (0..1)",
        default=0.6, lo=0.0, hi=1.0),
    Question(
        "clm_label.target_metric.scope", "choice", "Measure that MCC on",
        explain="How much of the picture should the score judge?\n"
                "  pair   : only your one chosen cluster and its label, each against\n"
                "           everything else. The tool hits this exactly and instantly.\n"
                "  global : the whole set of labels against all the clusters at once\n"
                "           (the standard multiclass MCC). Found by search; it can\n"
                "           fall short if the value is impossible for your data.",
        default="global", choices=("pair", "global")),
    Question(
        "clm_label.target_metric.tolerance", "float", "How close counts as 'reached'",
        explain="If the finished labelling lands further than this "
                "from your target,\nthe run says so rather than "
                "reporting the number you asked for.",
        default=0.01, lo=0.0, hi=1.0),

    # single_match
    Question(
        "clm_label.single_match.cluster", "id",
        "Which cluster id becomes the aligned label? (an id from your cluster column, e.g. 1)"),
    Question(
        "clm_label.single_match.label", "int", "Which label value (0..M-1) it becomes",
        default=0, lo=0),

    # custom rules (the per-rule loop stays procedural; these are its fields)
    Question(
        "clm_label.assignment_matrix._count", "int", "How many rules", default=1, lo=1),
    Question(
        "clm_label.assignment_matrix.label", "int", "    which label value (0..M-1)",
        default=0, lo=0),
    Question(
        "clm_label.assignment_matrix.clusters", "ids",
        "  into which cluster id(s), comma-separated"),
    Question(
        "clm_label.assignment_matrix.recall_target", "float",
        "    recall_target, share of THAT label's points to put here "
        "(1.0 = all/strong, lower = partial)",
        default=0.8, lo=0.0, hi=1.0),
    Question(
        "clm_label.split_rule", "choice",
        "If a rule targets several clusters, how to split the label between them",
        explain="proportional_to_size : bigger clusters get more of the label (usual choice).\n"
                "equal                : each targeted cluster gets the same amount.",
        default="proportional_to_size", choices=("proportional_to_size", "equal")),

    # spillover
    Question(
        "clm_label.spillover_rule", "choice", "After the rules, how to fill the leftover points",
        explain="Rules rarely use every point; the rest still need a label:\n"
                "  proportional_to_marginal : fill them so final sizes EXACTLY match your\n"
                "                             proportions (recommended, keeps your split).\n"
                "  uniform                  : spread evenly, this CHANGES your proportions.\n"
                "  concentrated             : dump all leftovers into one label.",
        default="proportional_to_marginal",
        choices=("proportional_to_marginal", "uniform", "concentrated")),
    Question(
        "clm_label.concentrated_labels", "int_list",
        "Which label value(s) should absorb the leftovers",
        explain="All leftover points go to these labels. Leave one value for the\n"
                "usual case; the default without this is the single largest label."),

    # competing noise
    Question(
        "clm_label.competing_noise._enabled", "bool",
        "Give one cluster's leftover points a specific competing label (structured noise)?",
        explain="Optional 'structured noise': points NOT claimed by your rules normally get\n"
                "filler labels from the spillover rule. This instead forces a share of ONE\n"
                "cluster's leftover points to a single competing label, placed at that\n"
                "cluster's edge (or center). Consequences to be aware of:\n"
                "  - it BYPASSES your proportions: final label sizes will no longer match\n"
                "    them (same caveat as the uniform/concentrated spillover rules);\n"
                "  - it CHANGES the agreement score: structured noise scores differently\n"
                "    from random noise, comparing the two is exactly what it is for;\n"
                "  - it only shapes points not already claimed by the rules above.",
        default=False),
    Question(
        "clm_label.competing_noise.cluster", "id",
        "  which cluster id gets the competing label"),
    Question(
        "clm_label.competing_noise.label", "int",
        "    which label value (0..M-1) competes there", default=0, lo=0),
    Question(
        "clm_label.competing_noise.share", "float",
        "    share of that cluster's LEFTOVER points to convert (0..1)",
        default=1.0, lo=0.0, hi=1.0),
    Question(
        "clm_label.competing_noise.favors", "choice",
        "    where the competing label sits in the cluster",
        explain="boundary : the cluster's rim (the usual 'hard case').\n"
                "core     : the cluster's center.\n"
                "random   : anywhere in the cluster.",
        default="boundary", choices=("boundary", "core", "random")),
    Question(
        "clm_label.competing_noise._add_another", "bool",
        "  Add another competing-noise entry?", default=False),

    # centroid dependence
    Question(
        "clm_label.centroid_dependence._enabled", "bool",
        "Control WHERE inside each cluster the labelled points sit (center vs edge)?",
        explain="Optional, spatial only: choose whether the labelled points sit near the\n"
                "cluster CENTER (core) or its EDGE (boundary). This does NOT change the score,\n"
                "only which points carry the label. Useful for testing how a metric reacts\n"
                "to geometry.",
        default=False),
    Question(
        "clm_label.centroid_dependence.profile", "choice", "How strongly to prefer them",
        explain="linear      : gentle, even preference from center to edge.\n"
                "exponential : strong, points very near the target dominate\n"
                "              (tune with 'steepness').\n"
                "step        : a hard cut, take the N nearest (or farthest),\n"
                "              nothing in between.",
        default="linear", choices=("linear", "exponential", "step")),
    Question(
        "clm_label.centroid_dependence.favors", "choice", "Put the labelled points at the",
        explain="core     : the center of each cluster.\n"
                "boundary : the edge/rim of each cluster.",
        default="core", choices=("core", "boundary")),
    Question(
        "clm_label.centroid_dependence.steepness", "float",
        "Steepness (higher = more extreme; typical 2-8)", default=3.0, lo=0.0),

    # balance / skew
    Question(
        "clm_label.balance", "choice", "Label balance",
        explain="balanced   : every label is roughly the same size.\n"
                "unbalanced : some labels are bigger than others (like real data,\n"
                "             where one class is often much more common).",
        default="balanced", choices=("balanced", "unbalanced")),
    Question(
        "clm_label._explicit_proportions", "bool", "Type the exact sizes yourself?",
        explain="Yes: you give the fraction for each label (they must add up to 1).\n"
                "No: a 'skew rule' invents an uneven split for you.",
        default=True),
    Question(
        "clm_label.proportions", "float_list", "Fraction for each of the {M} labels",
        explain="One number per label, adding up to 1.0. They become the label\n"
                "sizes, scaled to your data (e.g. 0.5 of 6500 points = 3250)."),
    Question(
        "clm_label.skew_rule", "choice", "Skew rule",
        explain="How to make the label sizes uneven:\n"
                "  geometric         : a smooth shrinking series (e.g. 50, 25, 12, ...).\n"
                "                      One 'ratio' knob, smaller ratio = steeper drop.\n"
                "  dominant_minority : ONE label takes a big share you choose; the rest\n"
                "                      split what's left equally (one common, several rare).\n"
                "  dirichlet         : sizes drawn at RANDOM but repeatable from the seed.\n"
                "                      'alpha' sets how uneven: <1 = very lopsided,\n"
                "                      >1 = nearly equal. Good for sampling many imbalances.",
        default="geometric", choices=("geometric", "dominant_minority", "dirichlet")),
    Question(
        "clm_label.skew_params.ratio", "float", "ratio (0..1; smaller = steeper drop-off)",
        default=0.5, lo=0.0, hi=1.0, engine_min=0.0,
        visible_when=_skew_live("geometric")),
    # engine_max (dominant_index < M) is dynamic in M, so the wizard applies it at
    # the call site; only the static lower bound is declared here for the property.
    Question(
        "clm_label.skew_params.dominant_index", "int", "which label is the big one (0..M-1)",
        default=0, lo=0, engine_min=0,
        visible_when=_skew_live("dominant_minority")),
    Question(
        "clm_label.skew_params.dominant_share", "float", "its share of all points (0..1, e.g. 0.6)",
        default=0.6, lo=0.0, hi=1.0, engine_min=0.0, engine_max=1.0,
        visible_when=_skew_live("dominant_minority")),
    # alpha > 0 strictly: the engine divides by the normalized draw, so exactly 0
    # is a divide-by-zero. lo_strict makes the wizard refuse 0.0 as the engine does.
    Question(
        "clm_label.skew_params.alpha", "float", "alpha (smaller = more lopsided; try 0.3 or 1.0)",
        default=1.0, lo=0.0, lo_strict=True, engine_min=0.0, engine_min_strict=True,
        visible_when=_skew_live("dirichlet")),

    # -- 4. Save & run ------------------------------------------------------
    Question(
        "output.path", "str", "Save this config to (a .yaml file)",
        default="test_data_config.yaml"),
    Question("run.now", "bool", "Run the pipeline now?", default=True),
]

SCHEMA: dict[str, Question] = {q.key: q for q in _QUESTIONS}

if len(SCHEMA) != len(_QUESTIONS):                       # pragma: no cover - authoring guard
    counts: dict[str, int] = {}
    for _q in _QUESTIONS:
        counts[_q.key] = counts.get(_q.key, 0) + 1
    dupes = sorted(k for k, n in counts.items() if n > 1)
    raise RuntimeError(f"duplicate question keys in SCHEMA: {dupes}")
