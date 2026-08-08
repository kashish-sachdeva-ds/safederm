# ADR-001: Business Problem Framing — Reduce Unnecessary Specialist Referrals Without Missing High-Risk Lesions

**Date:** 2026-08-08  
**Status:** Accepted

## Context

SafeDerm is an AI-based triage support system designed for government clinics and small hospitals, where a patient may present with a spot or mole but the clinician may not be a dermatologist. In that situation, the clinician may refer the patient to a specialist center such as AIIMS or PGI Chandigarh because the lesion cannot be confidently assessed locally.

The operational problem is not simply that skin lesions are difficult to classify. The problem is that uncertainty at the first point of care can create two competing consequences.

First, benign or low-risk cases may be referred unnecessarily. Those referrals consume specialist capacity and can require patients and their families to spend additional time and money travelling to a specialist center. Specialist clinicians then spend time evaluating cases that may ultimately prove to be normal.

Second, a genuinely concerning lesion must not be missed simply because the first-line system or clinician was given an apparently confident but incorrect answer. The cost of a missed malignant lesion is substantially more serious than the cost of referring a benign case.

The project therefore needs to be framed as a **triage-support problem rather than an autonomous diagnosis problem**. The system should help the clinic doctor make a safer routing decision from an image while preserving referral as the fallback when the case cannot be handled confidently.

The intended operational outcomes are:

- reduce unnecessary specialist referrals for high-confidence normal cases;
- identify concerning cases for prompt specialist attention;
- preserve referral for cases where the system does not have enough confidence to support a local decision;
- reduce patient waiting time and unnecessary travel where it is safe to do so; and
- avoid treating model accuracy as the only success criterion when the consequences of false negatives and false positives are asymmetric.

The underlying business/operational problem can therefore be stated as:

> **How can SafeDerm reduce avoidable specialist referrals and improve patient flow at first-line clinics while ensuring that uncertainty or model error does not result in a genuinely concerning lesion being incorrectly cleared?**

This framing deliberately comes before model selection. The project should first define what decision the prediction is intended to support and what failure is unacceptable, rather than allowing the dataset or a model's performance to define the business story after the fact.

## Options Considered

1. **Generic binary classification — predict benign vs. malignant and use the prediction as the routing decision.**

   The system could be framed primarily as a classifier whose output is converted directly into a referral/no-referral decision.

2. **Referral-first workflow — continue referring uncertain or potentially concerning cases without introducing an AI-supported local clearance decision.**

   This minimizes the risk of locally clearing a concerning case, but does not address the unnecessary-referral and specialist-capacity problem that motivates the project.

3. **AI-assisted triage — use the image-based model as a second opinion and route cases into normal, concerning, or uncertain outcomes.**

   High-confidence normal cases can avoid unnecessary referral; concerning cases can be sent to specialist care with priority; and cases where the system is not sufficiently reliable remain referred rather than being forced into a binary answer.

## Decision

**Option 3. SafeDerm will be framed as an AI-assisted triage system, not an autonomous diagnostic system.**

The business decision is that the system should support the clinic doctor with a **three-way operational routing outcome**:

- **Normal / low-risk:** the model has sufficient evidence to support local handling without specialist referral.
- **Concerning / high-risk:** the case should be referred to specialist care promptly.
- **Uncertain:** the model does not have enough reliable evidence to support either a local clearance or a confident high-risk classification, so the patient remains referred.

The key decision is not simply to predict whether a lesion is malignant. It is to determine **what action the clinic should take given the model's evidence and uncertainty**.

This also means that the project's primary objective is not "maximize classification accuracy." The system must be evaluated according to whether it supports the intended triage decision safely and reduces avoidable referral burden without creating unacceptable missed-case risk.

## Reasoning

A binary prediction forces the system to answer even when the available evidence is weak. That is unsuitable for this operational problem because the two mistakes are not equivalent.

If a benign lesion is referred unnecessarily, the cost is additional patient time, travel, expense, and specialist workload. That is undesirable but recoverable.

If a malignant or otherwise genuinely concerning lesion is incorrectly cleared, the consequence can be substantially more serious. Therefore, the system must have an explicit mechanism for refusing to make a confident local-clearance decision when its evidence is insufficient.

The three-way routing model directly represents that operational reality. It also preserves the human clinician as part of the decision loop: SafeDerm is a second opinion intended to support triage, not replace specialist diagnosis.

The framing also gives the technical architecture a clear purpose. Once the business requirement says that "uncertain" must be a real and safe outcome, the model cannot simply expose an uncalibrated class score and call it confidence. The technical system must produce evidence that is reliable enough to distinguish confident decisions from cases that should remain in the referral pathway.

## Trade-offs / What This Costs

- The system does not eliminate specialist referrals; it is designed to make them more targeted.
- The uncertain tier deliberately sends some cases to specialist care even when some of those cases may ultimately be benign.
- A human clinician remains involved in the workflow rather than allowing fully automated diagnosis or routing.
- Reducing referrals cannot be pursued at the expense of missed high-risk cases, so the system may accept additional false positives in exchange for a safer false-negative profile.
- The business value depends on the quality of the uncertainty/routing mechanism, not merely on the classifier's headline accuracy.
- The current business case is oriented around reducing unnecessary specialist burden, patient time/cost, and improving triage. A separate hospital-facing patient-flow business case may require different integration points and metrics.

## What Would Change My Mind

- If real-world deployment evidence showed that the three-way routing workflow did not materially reduce unnecessary referrals or patient waiting time, the business framing should be revisited.
- If clinical workflow analysis showed that a different intervention point was more valuable than image-based first-line triage, the system's role should be reconsidered.
- If evidence demonstrated that the model could not reliably identify cases appropriate for local handling without increasing unacceptable missed-case risk, the project should remain referral-first rather than forcing an AI clearance pathway.
- If a reliable non-AI workflow could achieve the same reduction in specialist burden with lower operational risk and cost, the value proposition for the AI system should be reassessed.

## Future Scope

A hospital-facing version could position SafeDerm primarily as a patient-flow and queue-reduction tool rather than an insurance-oriented triage system. That version could report metrics such as time-to-referral saved and integrate with hospital scheduling systems.

The current scope remains focused on the first-line clinic triage problem and the safe routing decision that follows from it.

## Relationship to ADR-002

ADR-001 establishes **what operational decision SafeDerm must support**: normal, concerning, or uncertain triage while protecting against unsafe clearance.

ADR-002 establishes **how the technical architecture should produce a sufficiently trustworthy confidence and uncertainty signal to support that decision**.
