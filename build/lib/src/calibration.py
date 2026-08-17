"""Calibration, uncertainty, and three-tier routing for SafeDerm.

Implements the pieces ADR-002 commits to but that didn't exist in code
yet: temperature scaling, MC Dropout uncertainty, calibration-quality
metrics (ECE/MCE/Brier), and the normal/concerning/uncertain router.
Lives in src/ (not inline in 08_calibration_conformal.ipynb) because
api/main.py needs the exact same logic at inference time -- the same
reason engine.py's malignant_recall() was pulled out of a notebook cell.

See docs/decisions/ADR-002-safederm-technical-architecture.md for why
temperature scaling + MC Dropout were chosen over raw softmax, and
ADR-006 for the specific threshold-fitting procedure below.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

TIER_NORMAL = "normal"
TIER_CONCERNING = "concerning"
TIER_UNCERTAIN = "uncertain"


# --------------------------------------------------------------------------
# Temperature scaling (Guo et al., 2017)
# --------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Learns a single scalar T that divides logits before softmax.

    T > 1 softens an overconfident model's probabilities (spreads mass
    away from the top class) without changing its argmax predictions --
    accuracy is unaffected, only how trustworthy the reported confidence
    number is. Parameterized as exp(log_T) so T can't go to/below zero
    during optimization, which the classic raw-scalar implementation
    doesn't guard against.
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # T starts at 1.0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.1) -> float:
        """Fits T by minimizing NLL on held-out logits/labels.

        Must be called with VALIDATION logits, never train or test --
        fitting on train would calibrate against the same overconfidence
        the model learned to produce on train; fitting on test would
        leak test data into something that affects the API's output.

        line_search_fn="strong_wolfe" matters, not just a tuning nicety:
        PyTorch's LBFGS with no line search can converge to a poor step
        size and stop early. Verified against a manual grid search over T
        on synthetic overconfident logits -- without strong_wolfe this
        landed on T=1.58 against a true NLL-minimum near T=3.0; with it,
        it finds the actual minimum.
        """
        logits = logits.detach()
        labels = labels.detach()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


# --------------------------------------------------------------------------
# Calibration-quality metrics
# --------------------------------------------------------------------------


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted average |accuracy - confidence| across confidence
    bins (Guo et al., 2017). 0 = perfectly calibrated. This is the
    single number to report before/after temperature scaling to show it
    actually helped.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop_in_bin
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """MCE: worst-case |accuracy - confidence| across bins. ECE averages
    this away; a model can have low ECE and still be badly miscalibrated
    in one confidence range (e.g. exactly the range malignant calls tend
    to fall in), which is what MCE is for catching.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            gap = abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
            max_gap = max(max_gap, gap)
    return float(max_gap)


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Mean squared error between predicted probabilities and the one-hot
    true label, averaged over samples. 0 = perfect; a model predicting
    uniform 1/7 for everything scores worse than one that's right but
    underconfident, and much better than one that's confidently wrong --
    rewards calibrated uncertainty, not just correctness.
    """
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# --------------------------------------------------------------------------
# MC Dropout
# --------------------------------------------------------------------------


def enable_mc_dropout(model: nn.Module) -> None:
    """Puts the model in eval mode (frozen BatchNorm running stats) but
    forces Dropout layers back into train mode so they keep sampling
    randomly. Standard MC Dropout trick (Gal & Ghahramani, 2016): dropout
    has to stay "on" at inference for repeated stochastic passes to
    disagree with each other; BatchNorm must NOT switch to per-batch
    statistics for a single inference image, so it stays in eval mode.
    Only works if the model actually contains an nn.Dropout layer --
    see src/model.py's _TransformerHead, which exists for this reason.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 20, temperature: float = 1.0):
    """Runs n_passes stochastic forward passes with dropout active.

    Returns (mean_probs, predictive_entropy):
    - mean_probs [B, num_classes]: averaged, temperature-scaled softmax --
      this is the calibrated confidence to show the user, not any single
      pass's output.
    - predictive_entropy [B]: how much the passes disagreed with each
      other. High entropy means dropout produced genuinely different
      predictions across passes -- the model itself doesn't have a
      confident opinion, independent of what its single-pass softmax
      score would have claimed.
    """
    enable_mc_dropout(model)
    all_probs = []
    for _ in range(n_passes):
        logits = model(x) / temperature
        all_probs.append(torch.softmax(logits, dim=1))
    stacked = torch.stack(all_probs, dim=0)      # [n_passes, B, num_classes]
    mean_probs = stacked.mean(dim=0)              # [B, num_classes]

    eps = 1e-12
    entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)
    return mean_probs, entropy


# --------------------------------------------------------------------------
# Three-tier routing
# --------------------------------------------------------------------------


def assign_tier(
    predicted_risk_group: str,
    predictive_entropy: float,
    calibrated_confidence: float,
    entropy_threshold: float,
    confidence_threshold: float,
) -> str:
    """Routes a single prediction into normal / concerning / uncertain.

    Order matters and is safety-motivated, not incidental:

    1. Predicted malignant -> always "concerning", regardless of
       confidence or entropy. A low-confidence malignant guess is still
       exactly the situation where "go get this checked" is the right
       message -- confidence should never be able to suppress it.
    2. Otherwise (predicted benign): only trust that call as "normal" if
       both the model's uncertainty is low (MC Dropout passes agreed)
       AND its calibrated confidence clears the bar. Fails either check
       -> "uncertain", which still tells the person to get checked,
       rather than confidently saying "you're fine" on a call the model
       itself isn't sure about.

    Thresholds must come from fit_tier_thresholds() on real validation
    data -- this function never invents a "reasonable-looking" default,
    since a wrong default here is a safety mistake, not a UX nitpick.
    """
    if predicted_risk_group == "malignant":
        return TIER_CONCERNING
    if predictive_entropy > entropy_threshold:
        return TIER_UNCERTAIN
    if calibrated_confidence < confidence_threshold:
        return TIER_UNCERTAIN
    return TIER_NORMAL


def fit_tier_thresholds(
    calibrated_confidences,
    predictive_entropies,
    predicted_risk_groups,
    true_risk_groups,
    max_uncertain_rate_on_correct_benign: float = 0.15,
) -> dict:
    """Fits entropy_threshold and confidence_threshold on validation data.

    assign_tier() already routes every malignant-*predicted* case to
    "concerning" unconditionally -- these two thresholds only affect
    cases the model predicted benign. Those split into two groups: the
    model was right (true benign), or wrong (true malignant -- a missed
    malignancy). The thresholds' entire job is to "rescue" the second
    group into "uncertain" before they reach "normal".

    For each candidate threshold (swept from percentiles of the actual
    validation entropy/confidence distribution) this computes:
    - rescue_rate: fraction of missed-malignant-predicted-benign cases
      correctly pushed to "uncertain"
    - false_alarm_rate: fraction of correctly-predicted-benign cases
      needlessly pushed to "uncertain" too

    ...and picks the highest rescue_rate whose false_alarm_rate stays
    under `max_uncertain_rate_on_correct_benign` (default: at most 15%
    of correct benign calls get second-guessed). If nothing satisfies
    that budget, returns the lowest-false-alarm option with a `warning`
    key instead of silently picking something nobody reviewed.

    This is a starting point that makes the trade-off space explicit and
    auditable from real validation data -- not a claim that its output
    is the final number to ship. That call belongs to whoever owns the
    clinical/product risk tolerance, not a script.
    """
    conf = np.asarray(calibrated_confidences, dtype=float)
    entropy = np.asarray(predictive_entropies, dtype=float)
    pred_group = np.asarray(predicted_risk_groups)
    true_group = np.asarray(true_risk_groups)

    predicted_benign = pred_group == "benign"
    missed_malignant = predicted_benign & (true_group == "malignant")
    correct_benign = predicted_benign & (true_group == "benign")

    if missed_malignant.sum() == 0:
        raise ValueError(
            "No true-malignant/predicted-benign cases in this validation "
            "set -- either the classifier has 100% malignant recall "
            "already (verify this isn't a data bug) or the val set is "
            "too small to fit thresholds reliably on."
        )

    def sweep(values, mask_missed, mask_correct, higher_is_more_uncertain):
        candidates = np.unique(np.percentile(values[predicted_benign], np.arange(5, 100, 5)))
        results = []
        for t in candidates:
            flagged = (values > t) if higher_is_more_uncertain else (values < t)
            rescued = mask_missed & flagged
            false_alarmed = mask_correct & flagged
            rescue_rate = rescued.sum() / max(mask_missed.sum(), 1)
            false_alarm_rate = false_alarmed.sum() / max(mask_correct.sum(), 1)
            results.append((float(t), float(rescue_rate), float(false_alarm_rate)))
        return results

    entropy_results = sweep(entropy, missed_malignant, correct_benign, higher_is_more_uncertain=True)
    within_budget = [r for r in entropy_results if r[2] <= max_uncertain_rate_on_correct_benign]

    warning = None
    if within_budget:
        entropy_threshold, rescue_rate, false_alarm_rate = max(within_budget, key=lambda r: r[1])
    else:
        entropy_threshold, rescue_rate, false_alarm_rate = min(entropy_results, key=lambda r: r[2])
        warning = (
            f"No entropy threshold kept the false-alarm rate under "
            f"{max_uncertain_rate_on_correct_benign:.0%}; reporting the "
            "lowest false-alarm option instead ({:.0%}). Review manually "
            "before shipping.".format(false_alarm_rate)
        )

    # Second pass: among predicted-benign cases that already clear the
    # entropy bar, see if a confidence floor rescues any further misses.
    passes_entropy = entropy <= entropy_threshold
    remaining_missed = missed_malignant & passes_entropy
    remaining_correct = correct_benign & passes_entropy

    confidence_threshold = 0.0
    conf_rescue_rate = 0.0
    if remaining_missed.sum() > 0:
        conf_results = sweep(conf, remaining_missed, remaining_correct, higher_is_more_uncertain=False)
        conf_within_budget = [r for r in conf_results if r[2] <= max_uncertain_rate_on_correct_benign]
        if conf_within_budget:
            confidence_threshold, conf_rescue_rate, _ = max(conf_within_budget, key=lambda r: r[1])

    result = {
        "entropy_threshold": entropy_threshold,
        "confidence_threshold": confidence_threshold,
        "entropy_rescue_rate": rescue_rate,
        "entropy_false_alarm_rate": false_alarm_rate,
        "additional_rescue_rate_from_confidence": conf_rescue_rate,
        "n_missed_malignant_in_val": int(missed_malignant.sum()),
        "n_correct_benign_in_val": int(correct_benign.sum()),
    }
    if warning:
        result["warning"] = warning
    return result


# --------------------------------------------------------------------------
# Persistence: fit once in a notebook, load once at API startup
# --------------------------------------------------------------------------


@dataclass
class CalibrationArtifact:
    """Everything api/main.py needs to serve calibrated, tiered
    predictions instead of raw softmax. Fit once in
    08_calibration_conformal.ipynb, saved to disk, loaded once at API
    startup -- the API doesn't need to know HOW temperature scaling or
    threshold selection work, only how to apply already-fitted numbers.
    """

    temperature: float
    entropy_threshold: float
    confidence_threshold: float
    mc_dropout_passes: int = 20

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CalibrationArtifact":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def load_if_exists(cls, path: Path) -> Optional["CalibrationArtifact"]:
        """Returns None instead of raising if `path` doesn't exist yet --
        lets api/main.py start up and serve raw predictions (clearly
        flagged as uncalibrated) before 08 has produced this file,
        instead of the API being blocked on notebook work finishing.
        """
        path = Path(path)
        return cls.load(path) if path.exists() else None
