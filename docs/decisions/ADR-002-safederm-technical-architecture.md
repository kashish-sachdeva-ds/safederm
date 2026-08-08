# ADR-002: Technical Architecture — Calibrated Confidence + Uncertainty-Based Routing, Not a Plain Threshold Classifier

**Date:** 2026-08-08  
**Status:** Accepted

## Context

ADR-001 establishes SafeDerm as an AI-assisted triage system rather than an autonomous diagnostic system. The system must support a three-way operational routing decision:

- **Normal / low-risk:** sufficiently strong evidence to support local handling.
- **Concerning / high-risk:** refer to specialist care promptly.
- **Uncertain:** the system does not have enough reliable evidence to support a local clearance or a confident high-risk decision, so the patient remains referred.

That requirement creates a technical problem that is more specific than ordinary image classification.

A standard CNN can produce a class prediction and a softmax score, but a raw softmax score should not automatically be treated as a trustworthy probability or confidence measure. A model can be highly confident and still be wrong. That is particularly problematic here because the system's most important safety property is its ability to avoid turning an uncertain case into a false reassurance.

The technical question is therefore:

> **How should SafeDerm convert a model's raw prediction into a calibrated confidence and uncertainty signal that can safely drive three-way triage routing?**

The deployment constraint also matters. The intended deployment is CPU-only and uses a free-tier hosting plan, so the architecture cannot assume that serving or repeatedly evaluating a large collection of independent models is practical.

## Options Considered

1. **Single CNN, fixed threshold, raw softmax as "confidence"**

   Train a ResNet-50/EfficientNet-B0-style classifier, select a decision threshold, and treat the raw softmax output as the confidence shown to clinic staff.

2. **Deep ensemble for uncertainty, without a separate calibration step**

   Train multiple independently initialized models and use the spread across their predictions as the uncertainty signal, without adding a calibration stage.

3. **CNN + Transformer hybrid with temperature-scaled calibration and MC Dropout uncertainty, feeding an explicit three-tier decision rule**

   Use one trained model, calibrate its logits post-hoc with temperature scaling, estimate predictive uncertainty using stochastic MC Dropout passes, and combine those signals in an explicit normal/concerning/uncertain routing rule.

## Decision

**Option 3. SafeDerm will use calibrated confidence and uncertainty-based routing rather than a plain threshold classifier.**

Calibration and uncertainty are treated as distinct but complementary parts of the architecture.

**Temperature scaling** will be used after model training to calibrate the model's confidence scores on a validation set. This changes the confidence values without changing the underlying class prediction.

**MC Dropout** will be used at inference time to obtain an uncertainty signal from multiple stochastic forward passes through the same trained model.

These signals will feed an explicit three-tier routing rule:

1. **High-confidence normal → local handling / no specialist referral.**
2. **High-confidence concerning → specialist referral with priority.**
3. **Insufficient confidence / elevated uncertainty → specialist referral as uncertain.**

The system therefore does not force every image into a binary answer. The "I'm not sure" state is an intentional technical output required by the business decision established in ADR-001.

## Reasoning

### Why not a raw softmax threshold?

Option 1 was rejected because the raw softmax score does not provide a sufficient guarantee that a high numerical score represents reliable confidence. A fixed threshold would therefore risk converting model overconfidence into an operational routing decision.

That is incompatible with SafeDerm's core safety requirement: a model should not confidently clear a case merely because its numerical output is high.

### Why not a deep ensemble?

Option 2 was seriously considered because ensembles are a well-established approach for obtaining uncertainty information.

However, the current deployment constraint is CPU-only, with free-tier hosting. Training, storing, and serving 5–10 independent models creates a materially different computational and deployment cost profile.

The selected architecture instead obtains an uncertainty estimate from stochastic passes through one trained model, making the approach more compatible with the stated deployment constraint.

### Why temperature scaling?

Temperature scaling provides a relatively inexpensive post-hoc calibration step. A single learned temperature parameter is fitted on the validation set and applied to the model's logits before the softmax calculation.

The important architectural property is that calibration does not require retraining the full model and does not change which class is predicted. It is therefore a practical way to make the confidence signal more meaningful for downstream routing.

### Why MC Dropout?

MC Dropout reuses dropout already present in the network and performs multiple stochastic forward passes at inference time. The variation across those predictions provides an uncertainty signal without requiring several separately trained models.

This makes it a better fit for the current deployment constraint than a large deep ensemble, although it introduces additional inference latency.

### Why a CNN + Transformer hybrid?

The CNN + Transformer hybrid adds architectural complexity, but its attention mechanism provides attention maps that can support Grad-CAM-style visual explanation.

This is valuable in the intended clinical workflow because the doctor should have something more interpretable to inspect than a single numerical confidence score. The explanation mechanism is therefore primarily a human-review benefit rather than a claim of guaranteed accuracy improvement.

## Three-Tier Routing Logic

The technical architecture exists to implement the operational decision from ADR-001.

Conceptually:

**Image → model prediction → calibrated confidence + uncertainty → routing tier**

The routing system should not interpret confidence in isolation. A prediction can be classically "high confidence" while still being unsuitable for local clearance if the uncertainty signal indicates that the model is outside the conditions under which it can be trusted.

The uncertainty tier is therefore not a fallback error message. It is a first-class system outcome.

## Trade-offs / What This Costs

- **MC Dropout latency:** multiple forward passes are required for each prediction, increasing inference time compared with a single-pass classifier.
- **Calibration scope:** temperature scaling improves aggregate calibration on the validation set but does not guarantee equally good calibration across all subgroups.
- **Architecture complexity:** a CNN + Transformer hybrid is more complex than a plain CNN, and the main justification for that additional complexity is explainability rather than guaranteed benchmark accuracy.
- **Uncertainty coverage:** the uncertainty thresholds depend on the OOD stress-test set that was constructed. Real-world clinic inputs may be more varied than that test set.
- **Deployment ceiling:** the CPU-only/free-tier constraint limits the ability to adopt computationally heavier uncertainty methods such as a deep ensemble.
- **Human interpretation:** explanation maps can help a clinician inspect the model's suggestion, but they should not be treated as proof that the model's prediction is clinically correct.

## What Would Change My Mind

- If GPU inference becomes available or affordable, directly compare MC Dropout with a small deep ensemble and evaluate whether the additional computational cost produces meaningfully better uncertainty estimates.
- If subgroup calibration analysis shows that a single global temperature parameter is inadequate across skin tones, image quality, lesion types, or other relevant groups, evaluate a more granular calibration strategy.
- If MC Dropout's inference latency becomes a genuine bottleneck in clinic workflow rather than merely a benchmark cost, evaluate a faster uncertainty method such as an evidential deep-learning approach.
- If the OOD stress-test results show that the uncertainty mechanism does not reliably identify unfamiliar or low-quality inputs, the routing policy and uncertainty methodology should be reconsidered before deployment.

## Future Scope

The immediate technical scope is a single-model architecture with calibrated confidence, MC Dropout uncertainty, and explicit three-tier routing.

Future technical work can evaluate:

- deep ensembles when GPU resources permit;
- more granular subgroup calibration;
- improved OOD detection and uncertainty evaluation;
- faster uncertainty estimation if inference latency becomes operationally significant; and
- stronger clinician-facing explanations and audit logging.

The hospital-facing version described in ADR-001 may also require different system integrations and operational metrics, but those changes belong to a future product/workflow decision rather than this architecture decision.

## Relationship to ADR-001

ADR-001 answers:

> **What operational problem are we solving, and what decision should SafeDerm support?**

ADR-002 answers:

> **What technical architecture is required to make that decision using confidence and uncertainty rather than an unsafe forced binary prediction?**

The dependency is intentional: the three-tier technical design exists because the business problem requires an explicit and safe uncertain state.
