# clm_errors.py
"""
Central registry of CLM engine diagnostics: reference Manual/troubleshooting.tex.

Bands:
    1xx  ValueError:                invalid configuration / incompatible options
    15x  InfeasibleAllocationError: a ValueError subclass; valid config, counts don't fit
    2xx  raw Python KeyError, a required config key is missing (NOT represented here)
    3xx  Warning:                   non-fatal; logged, execution continues

Codes are a public contract: never renumber or reuse an assigned code; new
diagnostics get new numbers.
"""


class InfeasibleAllocationError(ValueError):
    """
    Raised when a specific recall_target/rule (or competing_noise) demand
    cannot fit within cluster capacities.
    A ValueError subclass, but caught *separately* by target_metric's search
    loop ("try a different alpha") and by main.py (skip the dataset). It
    must remain a distinct type: a config typo (plain ValueError) has to
    stay uncaught by those handlers and fail.
    """
    pass


CODES = {
    # --- 1xx : ValueError (invalid configuration) ---------------------------
    101: "Unknown matching_mode {mode!r}. Supported: perfect, single, random, custom.",
    102: "matching_mode 'perfect' requires M == K (got M={M}, K={K}).",
    103: "matching_mode 'single' requires M >= 2 and K >= 2 (got M={M}, K={K}).",
    104: "{where}: label {label!r} out of range 0..{hi}.",
    105: "{where}: cluster id(s) {unknown} not found (available: {available}).",
    106: "proportions must sum to 1, got {total}.",
    107: "Unknown skew_rule {skew_rule!r}. Supported: geometric, dominant_minority, dirichlet.",
    108: "Unknown split_rule {split_rule!r}. Supported: proportional_to_size, equal.",
    109: "Unknown spillover_rule {rule_name!r}. Supported: "
         "proportional_to_marginal, uniform, concentrated.",
    110: "Unknown profile {profile!r}. Supported: linear, exponential, step.",
    111: "target_metric requires matching_mode in {{'single', 'custom'}} (got {mode!r}): "
         "'perfect' has no free recall_target to search over, and 'random' has no "
         "cluster-label structure to dial at all.",
    112: "target_metric.type must be 'mcc' or 'ari' (got {type!r}).",
    113: "target_metric.value must be in [-1, 1] (got {value}).",
    114: "target_metric is incompatible with matching_mode='random': "
         "there is no cluster-label structure to dial.",
    115: "competing_noise is incompatible with matching_mode='random': there is no "
         "aligned label for a competing label to compete with (use 'single' or 'custom').",
    116: "competing_noise: favors must be 'core', 'boundary' or 'random', got {favors!r}.",
    117: "competing_noise: share must be in [0, 1], got {share}.",
    118: "competing_noise: label {label} out of range 0..{hi}.",
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
         "coords is {got}. Provide an (N, d) feature array, or disable "
         "centroid_dependence and any competing_noise favors 'core'/'boundary'.",
    126: "num_classes must be between 1 and {max_val} (got {M}).",
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

    # --- 15x : InfeasibleAllocationError (valid config, counts don't fit) ---
    150: "Infeasible rule: label {label} needs recall_target={rt} ({tp} of its {m} points) "
         "but clusters {clusters} hold only {capacity} points total. "
         "Max feasible recall_target here: {max_recall:.3f}.",
    151: "Infeasible configuration: cluster {k} has {size} points but rules jointly claim "
         "{claimed} (per label: {per_label}). Lower the competing recall_targets or spread "
         "them across more clusters.",
    152: "competing_noise: entries for cluster {k} jointly claim {claimed} points but only "
         "{remaining} are unclaimed there. Lower the 'share' values.",
    153: "Infeasible configuration: {n_rules} assignment_matrix rules name label {label} "
         "and jointly claim {claimed} points, but that label's budget is only {budget}. "
         "Each rule's recall_target is a fraction of the label's WHOLE budget, not a share "
         "of it, so recall targets on the same label add up: {breakdown}. Either lower them "
         "so they sum to at most 1.0, or write one rule listing all the clusters and let "
         "split_rule divide the budget between them.",

    # --- 3xx : Warnings (non-fatal) -----------------------------------------
    301: "balance='balanced': explicit 'proportions' are ignored (uniform 1/M split "
         "enforced). Set balance to 'unbalanced' to have your proportions used.",
    302: "matching_mode='perfect': proportions/balance/skew_rule ignored, label counts "
         "are forced to match their paired cluster's size.",
    303: "target_metric present: per-rule recall_target values in assignment_matrix are "
         "ignored; recall_target is solved for globally.",
    304: "competing_noise active: achieved label counts will deviate from the target "
         "proportions (structured noise bypasses the marginal, like uniform/concentrated "
         "spillover).",
    305: "competing_noise: cluster {k} has no unclaimed points (or share rounds to 0); "
         "entry {entry} has no effect.",
    306: "target_metric: did not converge within tolerance after {max_iter} iterations. "
         "Best found: alpha={best_alpha:.4f}, achieved={best_metric:.4f} "
         "(target={target:.3f}, tol={tol}).",
    307: "target_metric scope 'pair': target {target} is outside the reachable range "
         "[{phi_min:.3f}, {phi_max:.3f}] for this cluster/label pair; clamped to the "
         "nearest reachable value.",
    308: "target_metric scope 'pair': label {label} is sized to {m} points (a subset of "
         "its cluster) to meet the target MCC; any explicit proportion for it is overridden.",
    309: "target_metric: the DELIVERED labeling achieves {type}={achieved:.4f}, outside the "
         "requested {target} +/- {tol} (solved alpha={alpha:.4f}). The solver scores candidates "
         "on a fixed probe stream while this output is generated on the run's own stream, so a "
         "solve that converged internally can still land outside tolerance -- most often at "
         "small N, where spillover placement dominates. Treat the achieved value above as "
         "authoritative, not the requested one.",
    310: "target_metric scope 'pair': the DELIVERED labeling achieves pair mcc="
         "{achieved:.4f}, outside the requested {target} +/- {tol}. The closed-form "
         "construction assumes every point of label {label} lies inside cluster {cluster}, "
         "but {outside} of its {total} points lie outside. Treat the achieved value above "
         "as authoritative, not the requested one.",
}


def _format(code: int, **kw) -> str:
    return f"[CLM-{code}] " + CODES[code].format(**kw)


def clm_error(code: int, **kw) -> ValueError:
    """Build a coded ValueError (1xx). Use: ``raise clm_error(102, M=M, K=K)``."""
    exc = ValueError(_format(code, **kw))
    exc.code = code
    return exc


def clm_infeasible(code: int, **kw) -> InfeasibleAllocationError:
    """Build a coded InfeasibleAllocationError (15x)."""
    exc = InfeasibleAllocationError(_format(code, **kw))
    exc.code = code
    return exc


def clm_warn(log, code: int, **kw) -> None:
    """Emit a coded warning (3xx) through the caller's logger."""
    log.warning(_format(code, **kw))
