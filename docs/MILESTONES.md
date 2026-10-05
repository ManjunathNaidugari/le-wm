# Thesis implementation milestones

1. **Small Gibson pilot and actual simulation video.**
   Establish official episode execution, trustworthy recording, reference metrics,
   visible navigation and repeatability on development buildings. Implementation
   and local checks are ready. The user reports 15 successes across Adrian,
   Albertville and Anaheim on RunPod, plus exact repeatability of one episode.
   Comprehensive visual acceptance remains pending.
2. **Repository restructuring** — remove unnecessary LeWM code after the pilot.
   Existing uncommitted restructuring is preserved, not expanded by this milestone.
3. **Scene-separated expert training dataset** — agree training/development/final
   evaluation scene lists before collection. Availability inspection, saved disjoint
   splits and resumable collection are implemented; larger collection is pending.
4. **Frozen V-JEPA features and direct goal-conditioned policy** — current milestone.
   Audit, causal cached features, behavior cloning, goal-only ablation and restricted
   Habitat evaluation are implemented. Synthetic checks pass; pretrained weights,
   Linux/GPU compatibility, real-data overfit and learned rollouts must pass before
   accepting the milestone. See [DIRECT_BASELINE.md](DIRECT_BASELINE.md).
5. **Action-conditioned latent predictor, motion head and short-horizon planner.**
6. **Held-out Gibson evaluation and controller comparison videos.**

Both eventual controllers run inside Habitat. The direct baseline now uses a
frozen official V-JEPA 2 ViT-L/16 encoder and a small coordinate-goal policy.
No predictive dynamics, motion head or planner is implemented. Pilot buildings come from
the official **train** split. Reserve official val/test buildings for future final
evaluation and record the chosen split policy before training.

See [Gibson pilot execution guide](GIBSON_PILOT.md) and
[current verification evidence](GIBSON_PILOT_RESULTS.md).
