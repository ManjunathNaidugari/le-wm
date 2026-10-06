# Thesis implementation milestones

Updated 6 October 2026. The [agreed thesis plan](THESIS_PLAN.md) defines scope,
protocol, evidence and acceptance conditions. The main comparison is with an
established non-world-model RGB PointNav agent. Our direct policy supports
development and an ablation of predictive look-ahead.

## 1. Reliable learned navigation and benchmark protocol — week 1

- [x] Habitat/Gibson expert pilot and RGB/top-down recording infrastructure.
- [x] RunPod pretrained encoder execution and reviewed feature cache.
- [x] Real-feature 32-example overfit and learned-controller integration run.
- [ ] Preserve worker fix in experiment commit; reconcile final manifest.
- [ ] Fit all 431 retained pilot transitions and diagnose learned full-route behavior.
- [ ] Verify an external RGB PointNav baseline and freeze matched sensor/action settings.
- [ ] Materialize train/development/final buildings, checking external-model provenance.
- [ ] Collect/audit/cache a measured development dataset; train a useful proposal policy.

Current 1/15 learned success is integration evidence, not acceptance of learned
navigation. Expert success and synthetic fixtures are not learned results.

## 2. Predictive dynamics and navigation — weeks 2–3

- [ ] Prepare aligned next-latent and local-motion targets.
- [ ] Collect separately labeled alternative-action/recovery data on training buildings.
- [ ] Train latent and motion prediction; validate against simple controls.
- [ ] Implement bounded policy-guided look-ahead, scoring and replanning.
- [ ] Record predictions, selected actions and latency; verify development rollouts.

Deliver a controller whose world-model predictions actually affect action choice.

## 3. Unseen-building evaluation and presentation — weeks 3–4

- [ ] Freeze evaluation protocol and checkpoints.
- [ ] Compare complete system, external agent and no-look-ahead ablation.
- [ ] Report success/SPL, distance, collisions, failures, latency and uncertainty.
- [ ] Produce figures and RGB/top-down videos with predicted/actual motion.
- [ ] Finish thesis discussion, reproduction instructions and presentation.

Use `FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP` throughout. The agent uses RGB
and a ground-truth relative goal; camera-only localization is outside this scope.
Final evaluation buildings are never used for our training or tuning.
