# SPDX-License-Identifier: MIT
# clm_errors.py
"""
Central registry of CLM engine diagnostics: reference Manual/troubleshooting.tex.

Bands:
    1xx  ValueError:                invalid configuration / incompatible options
    15x  InfeasibleAllocationError: a ValueError subclass; valid config, counts don't fit
                                    or a strict target is missed
    2xx  MissingConfigKey:          a KeyError subclass; a required key is absent
    3xx  Warning:                   non-fatal; logged, execution continues
"""

import logging
from collections.abc import Mapping
from typing import Final

__all__ = [
    "CODES",
    "InfeasibleAllocationError",
    "MissingConfigKey",
    "clm_error",
    "clm_infeasible",
    "clm_missing",
    "clm_warn",
]


class InfeasibleAllocationError(ValueError):
    """
    Raised when a specific recall_target/rule (or competing_noise) demand
    cannot fit within cluster capacities, or when target_metric.strict refuses
    a delivered labeling outside the tolerance ([CLM-154]).
    A ValueError subclass, but caught *separately* by target_metric's search
    loop ("try a different alpha") and by main.py (skip the label).
    """


class MissingConfigKey(KeyError):
    """A required config key is absent: a key with no default, so no dataset can
    be labelled and main.py stops the run. A KeyError subclass; the message
    carries the code.
    """

    def __str__(self) -> str:
        # KeyError.__str__ would repr-quote the message.
        return self.args[0] if self.args else ""


CODES: Final[Mapping[int, str]] = {
    # --- 1xx : ValueError (invalid configuration) ---------------------------
    101: "Unknown matching_mode {mode!r}. Supported: perfect, single, random, custom.",
    102: "matching_mode 'perfect' requires M == K (got M={M}, K={K}).",
    103: "matching_mode 'single' requires M >= 2 and K >= 2 (got M={M}, K={K}).",
    104: "{where}: label {label!r} {problem} 0..{hi}.",
    105: "{where}: cluster id(s) {unknown} not found (available: {available}).",
    106: "proportions must sum to 1, got {total}.",
    107: "Unknown skew_rule {skew_rule!r}. Supported: geometric, dominant_minority, dirichlet.",
    108: "Unknown split_rule {split_rule!r}. Supported: proportional_to_size, equal.",
    109: "Unknown spillover_rule {rule_name!r}. Supported: "
         "proportional_to_marginal, uniform, concentrated.",
    110: "Unknown profile {profile!r}. Supported: linear, exponential, step.",
    111: "target_metric requires matching_mode in {{'single', 'custom'}} and M >= 2, K >= 2 "
         "(got {mode!r}, M={M}, K={K}): 'perfect' has no free recall_target to search over, "
         "'random' has no cluster-label structure to dial at all, and one class or one "
         "cluster leaves a single labeling, so every metric is fixed.",
    112: "target_metric.type must be 'mcc' or 'ari' (got {type!r}).",
    113: "target_metric.value must be in [-1, 1] (got {value}).",
    114: "target_metric is incompatible with matching_mode='random': "
         "there is no cluster-label structure to dial.",
    115: "competing_noise is incompatible with matching_mode='random': there is no "
         "aligned label for a competing label to compete with (use 'single' or 'custom').",
    116: "competing_noise: favors must be 'core', 'boundary' or 'random', got {favors!r}.",
    117: "competing_noise: share must be in [0, 1], got {share}.",
    118: "competing_noise: label {label!r} {problem} 0..{hi}.",
    119: "competing_noise: cluster {cluster} does not exist (available: {available}).",
    120: "target_metric: no feasible alpha in [0, 1], every candidate recall_target "
         "produced an infeasible allocation. Check that your rule clusters have enough "
         "capacity for the configured num_classes/proportions.",
    121: "proportions has {n} entries but num_classes is {M}; it must have exactly one "
         "entry per label. {detail}",
    122: "target_metric.scope must be 'pair' or 'global' (got {scope!r}).",
    123: "target_metric.scope 'pair' requires target_metric.type 'mcc': the single-pair "
         "MCC has an exact closed-form inverse, ARI does not.",
    124: "target_metric.scope 'pair' requires matching_mode 'single': the target pair is "
         "the single_match (cluster, label). Use scope 'global' for 'custom'.",
    125: "centroid placement requires per-point feature vectors: {placement} is set but "
         "coords is {got}. Provide an (N, d) feature array with d >= 2, or disable "
         "centroid_dependence and any competing_noise favors 'core'/'boundary'.",
    126: "num_classes must be an integer between 1 and {max_val} (got {M}).",
    127: "the dataset's cluster count K={K} exceeds the supported maximum of {max_val}.",
    128: "spillover_rule 'concentrated': concentrated_labels must be a LIST of integer "
         "label ids, each in 0..{hi}; got {given!r}. Values reaching the output "
         "unchecked would put labels in the dataset that num_classes never declared. "
         "Note a bare number is not a shorthand for a list: numpy reads it as a RANGE, "
         "scattering the remainder over that many labels, the opposite of "
         "'concentrated'. A non-integer (e.g. 1.5) is silently truncated on write, so "
         "it is rejected rather than guessed.",

    129: "centroid_dependence: favors must be 'core' or 'boundary', got {favors!r}. "
         "Matching is case-sensitive and exact; any other value would silently be "
         "treated as 'boundary', placing labels at the cluster rim when the core was "
         "intended.",

    130: "target_metric.scope 'pair' is incompatible with {what}. The closed-form "
         "construction sizes label {label} so that ALL of it sits inside cluster "
         "{cluster}; inverting the 2x2 phi depends on that. {what} can place that label "
         "outside the cluster, which breaks the identity and silently delivers a "
         "different value. Use spillover_rule 'proportional_to_marginal' (its leftover "
         "pool for the target label is empty by construction), keep competing_noise off "
         "label {label}, or switch to scope 'global'.",

    131: "skew_params: {problem}. These are read only when 'balance' is not 'balanced' "
         "and no explicit 'proportions' are given. Out-of-range values mostly do NOT "
         "crash there: they produce NEGATIVE label counts that still sum to N, so "
         "largest-remainder rounding is satisfied, nothing downstream objects, and the "
         "run delivers a labeling nobody asked for. Valid ranges: geometric 'ratio' "
         ">= 0; dominant_minority 'dominant_share' in [0, 1] and 'dominant_index' an "
         "integer in 0..{hi}, with num_classes >= 2; dirichlet 'alpha' > 0 and below "
         "about 1.8e308 / num_classes. Omit "
         "skew_params entirely to take the documented defaults.",

    132: "balance must be 'balanced' or 'unbalanced', got {balance!r}. Matching is "
         "case-sensitive and exact; any other value would silently be read as "
         "'unbalanced', so a typo of 'balanced' would deliver the proportions it was "
         "meant to ignore. Omit the key for the default, 'balanced'; a valueless "
         "`balance:` is not the default.",

    133: "the dataset cannot be labelled: {problem}. Every point needs one cluster id, "
         "the ids must be integers, and coords, when given, needs one row per point.",

    141:"{key} must be {expected}, got {value!r}. Values are not converted: YAML reads "
         "true/yes/no as booleans and .nan/.inf as numbers, and each was accepted as "
         "something else (true as 1, a quoted \"false\" as on, NaN past every range "
         "check, a mapping's keys as its values).",

    # One code, two blocks; {where} names which. Precedent: 104, 207.
    142: "{where}: cluster {cluster} is named more than once for label {label}. Each "
         "cluster is named once per label, so that one entry states how much of the "
         "label it holds, rather than a sum to work out: {fix}",

    144: "competing_noise entry {i}: label {label} is a target of cluster {cluster} "
         "({source}), so it cannot also be competing noise there. Competing noise "
         "places another label in a cluster: use a different label, or remove the "
         "entry.",

    # --- 15x : InfeasibleAllocationError (valid config, counts don't fit) ---
    150: "Infeasible rule: label {label} needs {tp} of its {m} points in clusters "
         "{clusters}, which hold only {capacity} points total. {remedy}",
    151: "Infeasible configuration: cluster {k} has {size} points but rules jointly claim "
         "{claimed} (per label: {per_label}), and no rounding of their counts to whole "
         "points fits. Lower the competing recall_targets or spread them across more "
         "clusters.",
    152: "competing_noise: entries for cluster {k} jointly claim {claimed} points but only "
         "{remaining} are unclaimed there. Lower the 'share' values.",
    153: "Infeasible configuration: {n_rules} assignment_matrix rules name label {label} "
         "and jointly claim {claimed} points, but that label's budget is only {budget}. "
         "Each rule's recall_target is a fraction of the label's WHOLE budget, not a share "
         "of it, so recall targets on the same label add up: {breakdown}. Either lower them "
         "so they sum to at most 1.0, or write one rule listing all the clusters and let "
         "split_rule divide the budget between them.",
    154: "target_metric strict: the delivered labeling scores {kind}={achieved:.6f}, "
         "{off:.3g} from the requested {target} (tolerance {tol}), so the label is not "
         "written.{detail}",

    # --- 2xx : MissingConfigKey (a required config key is absent) -----------
    201: "num_classes is required.",
    202: "matching_mode is required.",
    203: "skew_rule is required when balance is not 'balanced' and no proportions "
         "are given.",
    204: "single_match is required by matching_mode 'single'.",
    205: "single_match.{key} is required.",
    206: "assignment_matrix is required by matching_mode 'custom'.",
    207: "{where}: '{key}' is required.",
    208: "{where}: 'recall_target' is required without target_metric.",
    209: "target_metric.value is required.",

    # --- 3xx : Warnings (non-fatal) -----------------------------------------
    301: "balance='balanced': explicit 'proportions' are ignored (uniform 1/M split "
         "enforced). Set balance to 'unbalanced' to have your proportions used.",
    302: "matching_mode='perfect': proportions/balance/skew_rule/skew_params ignored "
         "and not validated, label counts are forced to match their paired cluster's "
         "size.",
    303: "target_metric present: per-rule recall_target values in assignment_matrix are "
         "ignored; recall_target is solved for globally.",
    304: "spillover_rule {rule!r}: the delivered label counts {achieved} differ from their "
         "targets {target}. 'proportional_to_marginal' refills every label to its target; "
         "'uniform' and 'concentrated' do not.",
    # One code, three causes; {cause} names which. Precedent: 104, 131.
    305: "competing_noise: entry {entry} {effect}: {cause}.",
    306: "target_metric: did not converge within tolerance after {n_iter} of {max_iter} "
         "iterations. "
         "Best found: alpha={best_alpha:.4f}, achieved={best_metric:.4f} "
         "(target={target:.3f}, tol={tol}).{ceiling}",
    307: "target_metric scope 'pair': target {target} is outside the reachable range "
         "[{phi_min:.3f}, {phi_max:.3f}] for this cluster/label pair; clamped to the "
         "nearest reachable value.",
    308: "target_metric scope 'pair': label {label} is sized to {m} points (a subset of "
         "its cluster) to meet the target {metric}; any explicit proportion for it is "
         "overridden.",
    309: "target_metric: the DELIVERED labeling achieves {type}={achieved:.4f}, outside the "
         "requested {target} +/- {tol}, but the search scored {type}={searched:.4f} at the "
         "same alpha={alpha:.4f}. The labeling the search scored is not the one written: the "
         "search ran on target_metric.probe_seed and the final generation on the run seed, "
         "and the two streams place the points differently. Omit probe_seed to have the "
         "search score the labeling it delivers. Treat the achieved value above as "
         "authoritative, not the requested one.",
    310: "target_metric scope 'pair': the DELIVERED labeling achieves pair {type}="
         "{achieved:.4f}, outside the requested {target} +/- {tol}: {cause}. Treat the "
         "achieved value above as authoritative, not the requested one.",
    311: "matching_mode='random': centroid_dependence ignored and not validated, labels "
         "are drawn with no regard to the clusters, so no point is placed by its distance "
         "to a centroid.",
}


def _format(code: int, **kw) -> str:
    return f"[CLM-{code}] " + CODES[code].format(**kw)


# `exc.code = code` attaches an attribute Python allows on any exception instance
# but that no exception type declares, so a type checker objects.

def clm_error(code: int, **kw: object) -> ValueError:
    """Build a coded ValueError (1xx). Use: ``raise clm_error(102, M=M, K=K)``."""
    exc = ValueError(_format(code, **kw))
    exc.code = code  # type: ignore[attr-defined]
    return exc


def clm_infeasible(code: int, **kw: object) -> InfeasibleAllocationError:
    """Build a coded InfeasibleAllocationError (15x)."""
    exc = InfeasibleAllocationError(_format(code, **kw))
    exc.code = code  # type: ignore[attr-defined]
    return exc


def clm_missing(code: int, **kw: object) -> MissingConfigKey:
    """Build a coded MissingConfigKey (2xx)."""
    exc = MissingConfigKey(_format(code, **kw))
    exc.code = code  # type: ignore[attr-defined]
    return exc


def clm_warn(log: logging.Logger, code: int, **kw: object) -> None:
    """Emit a coded warning (3xx) through the caller's logger."""
    log.warning(_format(code, **kw), extra={"clm_code": code})
