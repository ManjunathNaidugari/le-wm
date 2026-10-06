# Full-pilot diagnostic — 6 October 2026

Source: `pilot-1986e39/full-pilot-20261006-100412`, downloaded review bundle.
This is real frozen V-JEPA feature behavior cloning, not the proposed predictive
world model and not an unseen-building result.

| Measure | Result |
|---|---|
| Training | 431 transitions from 14 expert episodes; 200 epochs |
| Best checkpoint | Epoch 199; reported resubstitution accuracy 431/431 |
| Learned rollouts | 14/15 successes; mean SPL 0.928724 |
| Trained episode rollouts | 14/14 successes; zero recorded collisions |
| Excluded episode | Anaheim/19466: STOP after 33 actions, 4.64969 m remaining, six collisions |
| Runtime/video errors | 0 / 0; manifest complete |
| Decision latency | 322.9 ms weighted mean over 464 recorded decisions; excludes simulator/video time |

The excluded episode was removed from expert training for severe black frames.
Its failure does not isolate image quality from the absence of training examples.
The 15-degree turns and 0.25-metre forward steps were unchanged. Earlier training
on only 32 examples produced 1/15 successes and mean SPL 0.066667.

Local checks verified all 30 trajectory/video hashes, the checkpoint hash against
the rollout controller identity, and SPL arithmetic from the manifest's distances.
Training accuracy and per-episode metrics above are recorded values; this review
did not rerun the new checkpoint, decode all videos, or independently reconstruct
simulator success measurements. No held-out generalization or JEPA superiority is
claimed. Preserve the original artifacts on persistent storage, outside Git.
