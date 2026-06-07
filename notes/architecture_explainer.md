# Aperture — what each piece I added does

A one-page guide to every module added under [feat/trust-report-pillar-wiring](https://github.com/StanfordCS194/spr26-Team-21/tree/feat/trust-report-pillar-wiring) and [feat/privacy-anonymeter](https://github.com/StanfordCS194/spr26-Team-21/tree/feat/privacy-anonymeter), with the why behind each choice.

## The mental model

Every `/api/generate` call runs the same evaluation pipeline against the synthetic output. The pipeline is six pillars that each answer a different question:

| Pillar | Question it answers | Module |
|---|---|---|
| Utility (TSTR / TR+STR) | Can a model trained on synthetic data perform on real data? | `services/utility.py` |
| Rule packs | Does the synthetic data respect domain constraints we wrote down? | `services/rule_packs.py` |
| LLM auditor | Does each row look plausible to a model that has seen a lot of insurance? | `services/llm_auditor.py` |
| Detection | Can a discriminator tell synthetic rows from real ones? | `services/detection.py` |
| Privacy (distance) | On average, how close is synthetic data to specific real records? | `services/privacy.py` |
| Privacy (attacks) | Can an attacker actually do harmful things with this synthetic data? | `services/privacy_attacks.py` (this file) |

The trust report HTML at `services/trust_report.py` renders each pillar as one section. The final verdict tier (`high_confidence` / `moderate_confidence` / `prototype_only` / `review_required`) is computed by `classify_suitability` looking at the union of these signals.

## Detection pillar (already existed; my contribution: rendering it)

[backend/services/detection.py](../backend/services/detection.py) trains two discriminators — XGBoost (catches non-linear patterns) and Logistic Regression (catches linear-separability) — on rows labeled real vs synthetic. Reports:

- **ROC-AUC** per discriminator. 0.5 = synthetic indistinguishable from real (best). 1.0 = trivially separable (worst).
- **Expected Calibration Error (ECE)**. How well-calibrated the discriminator's probabilities are. Low ECE means a "0.7 probability" really does correspond to a 70% chance of being synthetic.
- **Agreement label.** Compares XGBoost vs LogReg: "both near chance", "non-linear separability only" (means there's a non-linear pattern XGBoost finds that LogReg can't), etc.

**What I added on [feat/trust-report-pillar-wiring](../backend/services/trust_report.py):**
`_render_detection_section()` reads the result dict from the session and renders it as a status pill + table. Without this function, the detection results were computed but never displayed to users.

## Privacy pillar — distance-based (already existed; my contribution: rendering + tier hook)

[backend/services/privacy.py](../backend/services/privacy.py) computes four numbers using the **Gower mixed-type distance** (handles numeric + categorical in one number):

| Metric | What it measures |
|---|---|
| **DCR** (Distance to Closest Record) | Median distance from a synthetic row to its nearest real row. Low = synth is "close" to real. Includes exact-match and near-duplicate counts (the actual leakage signal). |
| **NNDR** (Nearest-Neighbor Distance Ratio) | Ratio of nearest / second-nearest. If a synthetic row sits much closer to one real row than to any other, it's "tracking" that row. |
| **Baseline protection** | DCR normalized against a column-permuted random reference. 1.0 = synth is as far from real as random noise. < 0.3 = synth is much closer than random. |
| **Membership-inference attack (MIA) AUC** | A simple distance-based MIA: classify each candidate as a training-set member if its nearest neighbour in synth is close. AUC = 0.5 means attacker is guessing. |

**What I added on [feat/trust-report-pillar-wiring](../backend/services/trust_report.py):**
- `_render_privacy_section()` displays all four numbers with one-line explanations of what each one means.
- `classify_suitability` now takes an optional `privacy` argument. If MIA AUC ≥ 0.7 or exact-match duplicates exist, the verdict is downgraded to `review_required`. If MIA AUC ≥ 0.6 or near-duplicates ≥ 5%, downgraded to `moderate_confidence`. Without this, a privacy-leaky output could still ship as `high_confidence`.

## Privacy attacks — Anonymeter (my contribution, new)

[backend/services/privacy_attacks.py](../backend/services/privacy_attacks.py), `compute_anonymeter_risks()`. Wraps the [Anonymeter library](https://github.com/statice/anonymeter) (Giomi et al., PoPETS 2023; CNIL-validated). Runs three attacks, one for each GDPR Article 29 anonymization criterion:

| Attack | Story | Implementation |
|---|---|---|
| **Singling-out** | "I find a tiny attribute combination — e.g. `(Age=37, ZIP=94305, Job='dentist')` — that exists in the real data and uniquely matches one person." | Multivariate test with `n_cols=3`: tries 3-attribute combinations from synth, checks if they uniquely identify a real record. |
| **Linkability** | "I have two halves of an original record. Can the synthetic data help me re-link them?" | Splits feature columns in half; for each half-A record finds its nearest neighbour in synth; checks if the same neighbour also matches a half-B record. |
| **Inference** | "I know all but one attribute of a real record. Can I predict the missing (sensitive) attribute using the synthetic data?" | Trains a predictor on synth, evaluates on real records minus the secret column. The secret is the request's `label_col` or auto-detected. |

Each attack returns a `PrivacyRisk(value, ci)` where `value ∈ [0, 1]` (higher = worse). The summary `verdict` is the worst of the three.

**Why this is novel:** these are the three attacks that EU regulators care about. Commercial vendors (Mostly AI, Gretel, Syntho) often show "privacy ✓" with no numbers. Our trust report shows the value, confidence interval, and tier for each attack.

## Privacy attacks — DOMIAS density-based MIA (my contribution, new)

[backend/services/privacy_attacks.py](../backend/services/privacy_attacks.py), `compute_density_mia()`. Implements van Breugel et al.'s AISTATS 2023 attack:

> If a generator overfits to specific training records, those records will have HIGHER density under the synthetic distribution than non-training records, relative to a reference.

**Algorithm:**
1. Fit a kernel-density estimator on the synthetic data → `p_synth`.
2. Fit another on a held-out reference (non-training real data) → `p_ref`.
3. For each candidate record `x`, score: `log p_synth(x) − log p_ref(x)`. High score → likely a training-set member.
4. ROC-AUC of that score discriminating actual members vs non-members.

**Why this matters relative to the distance-based MIA:** the distance MIA only looks at the single nearest synth row to each candidate. DOMIAS sees the WHOLE synthetic distribution, so it catches local overfitting that "is the nearest neighbour close?" misses.

**Concrete example from the smoke test:** A "safe" synthetic dataset (column-shuffled real data) gets:
- Anonymeter: ALL three attacks at "low" risk
- DOMIAS: AUC = 0.86, "strong density-based leakage"

Reason: column-shuffling preserves the marginal distributions, so the density of training rows under synth is much higher than the density of non-training (holdout) rows under synth. DOMIAS catches it; Anonymeter and distance-MIA both miss it.

## Privacy ensemble verdict (my contribution, new)

[backend/services/privacy_attacks.py](../backend/services/privacy_attacks.py), `compose_privacy_ensemble()`. Combines the three sources above into one tier:

| Tier | Trigger |
|---|---|
| **severe** | Distance-MIA AUC ≥ 0.7 OR any Anonymeter risk ≥ 0.50 OR exact-match duplicates > 0 |
| **elevated** | Distance-MIA AUC ≥ 0.6 OR DOMIAS AUC ≥ 0.65 OR any Anonymeter risk ≥ 0.30 |
| **clean** | otherwise |

Conservative: one strong signal from any attack family flags the dataset. This is the right default — privacy claims should require all attacks to fail, not just one.

## Rule packs — fraud_oracle pack (my contribution, new)

[backend/rule_packs/fraud_oracle.yaml](../backend/rule_packs/fraud_oracle.yaml) added on [feat/rule-pack-expansion](https://github.com/StanfordCS194/spr26-Team-21/tree/feat/rule-pack-expansion). Nine hard rules covering the Kaggle Oracle Insurance Fraud Detection schema:

| Rule | What it checks |
|---|---|
| F1_age_range | Age ∈ [16, 100] |
| F2_year_range | Year ∈ {1994, 1995, 1996} |
| F3 / F4 week ranges | WeekOfMonth / WeekOfMonthClaimed ∈ [1, 5] |
| F5_driver_rating | DriverRating ∈ [1, 4] |
| F6_deductible_vocab | Deductible ∈ {300, 400, 500, 700} |
| F7_fraud_binary | FraudFound_P ∈ {0, 1} |
| F8_policy_type_vocab | PolicyType ∈ 9-value source vocabulary |
| F9_rep_number_range | RepNumber ∈ [1, 16] |

`detect_pack()` automatically recognizes fraud_oracle-shaped data when 3+ of `{FraudFound_P, BasePolicy, PolicyType, VehicleCategory}` are present, so the pack runs without explicit user selection.

This is what the published multi-generator showcase uses to compute the "rule violations per generator" table that demonstrated:
- TabSyn: 99.75% compliance (matches real Age=0 sentinel pattern almost exactly)
- TabDDPM: 66.67% (100% violation on Year + Deductible + PolicyType — quantile-transformer prep doesn't round to discrete bins)
- GaussianCopula: 88.96% (98.7% wrong Deductible — copula doesn't preserve discrete categories that look numeric)
- TVAE: 99.91% but **mode-collapsed** to single class

## Where each thing lives

```
backend/
  services/
    privacy.py             distance-based privacy (DCR / NNDR / MIA / baseline protection)
    privacy_attacks.py     Anonymeter (3 GDPR attacks) + DOMIAS + ensemble verdict   (NEW)
    detection.py           XGBoost + LogReg real-vs-synthetic discriminators
    rule_packs.py          YAML pack engine + repair primitives + detect/apply
    trust_report.py        HTML report renderer; six pillar sections + verdict tier
  rule_packs/
    insurance.yaml         auto/property insurance domain rules
    clinical.yaml          HbA1c + pregnancy + vitals
    fraud_oracle.yaml      Kaggle Oracle Insurance Fraud Detection schema             (NEW)
  api/
    generate.py            /api/generate orchestrator: runs all six pillars,
                           writes the session, returns the JSON envelope
```

## Pipeline flow on every `/api/generate` call

```
user request
   │
   ▼
synthesize(schema, source_stats, n, model_id)        ← services/synthesis.py
   │  (SDV-fit model preferred; falls back to statistical sampling)
   ▼
apply_edge_cases(rules)                              ← services/edge_cases.py
   │  (enforce "10% of rows have hba1c > 12" etc.)
   ▼
apply_pack(synth_df)                                 ← services/rule_packs.py
   │  (detect pack: insurance / clinical / fraud_oracle;
   │   check rules; auto-repair violations; recheck)
   ▼
validate(source_stats, synth_df, source_df)          ← services/validation.py
compute_utility(real, synth)                         ← services/utility.py
compute_diagnostics(real, synth, utility)            ← services/diagnostics.py
audit_sample(synth)                                  ← services/llm_auditor.py
compute_anonymeter_risks(real, synth, holdout)       ← services/privacy_attacks.py   (NEW)
compute_density_mia(real, synth, holdout)            ← services/privacy_attacks.py   (NEW)
compose_privacy_ensemble(privacy, attacks, dmia)     ← services/privacy_attacks.py   (NEW)
   │
   ▼
session[id] = { all pillar results }
   │
   ▼
render_html_report(session)                          ← services/trust_report.py
   │  (six pillar sections + suitability tier +
   │   risks list + next-steps list)
   ▼
HTML response
```

The new modules slot in cleanly: each takes pandas DataFrames as input, returns a dict, and the trust-report renderer reads from the session by key (defensively handles missing keys, so missing dependencies degrade rather than crash).

## How to explain it in one sentence

> *"We extend the team's evaluation pipeline with three GDPR-regulator-aligned privacy attacks (Anonymeter), one density-based membership-inference attack (DOMIAS), and an ensemble verdict that conservatively flags any dataset where at least one attack signal crosses a threshold — wired through the same session-based trust report the rest of the pipeline uses."*
