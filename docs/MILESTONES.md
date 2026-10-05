# Thesis implementation milestones

1. **Small Gibson pilot and actual simulation video** — current milestone.
   Establish official episode execution, trustworthy recording, reference metrics,
   visible navigation and repeatability on development buildings. Implementation
   and local checks are ready; Linux runtime/asset acceptance remains pending.
2. **Repository restructuring** — remove unnecessary LeWM code after the pilot.
   Existing uncommitted restructuring is preserved, not expanded by this milestone.
3. **Scene-separated expert training dataset** — agree training/development/final
   evaluation scene lists before collection. No large dataset collection now.
4. **Frozen V-JEPA features and direct goal-conditioned policy.**
5. **Action-conditioned latent predictor, motion head and short-horizon planner.**
6. **Held-out Gibson evaluation and controller comparison videos.**

Both eventual controllers run inside Habitat. The current work provides their
recording/evaluation loop and a simple controller callback; it adds no feature
encoder, learned model, training procedure or planner. Pilot buildings come from
the official **train** split. Reserve official val/test buildings for future final
evaluation and record the chosen split policy before training.

See [Gibson pilot execution guide](GIBSON_PILOT.md) and
[current verification evidence](GIBSON_PILOT_RESULTS.md).
