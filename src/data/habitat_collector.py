"""Collect aligned discrete transitions from a real Habitat-Sim episode."""
import numpy as np
import torch
from jepa_navigation.navigation.actions import Action


def collect_episode(env, max_steps=500, controller=None):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    states = [env.reset()]
    actions, collisions = [], []
    reason = "max_steps"
    error = None
    for _ in range(max_steps):
        try:
            action = Action(controller(states[-1]) if controller is not None else env.expert_action())
        except Exception as exc:
            # Preserve the real partial rollout, but never mislabel it as success.
            from habitat_sim.errors import GreedyFollowerError
            if not isinstance(exc, GreedyFollowerError):
                raise
            reason, error = "follower_error", str(exc)
            break
        next_state, collided = env.step(action)
        actions.append(int(action))
        collisions.append(collided)
        states.append(next_state)
        if action == Action.STOP:
            reason = "stop"
            break
    positions = np.stack([s["position"] for s in states])
    final_distance = env.distance()
    success = reason == "stop" and final_distance <= env.goal_radius
    episode = {
        "schema_version": 1,
        "action_names": [a.name for a in Action],
        "observations": torch.from_numpy(np.stack([s["rgb"] for s in states])).permute(0, 3, 1, 2).contiguous(),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "positions": torch.from_numpy(positions),
        "headings": torch.tensor([s["heading"] for s in states], dtype=torch.float32),
        "relative_goals": torch.from_numpy(np.stack([s["relative_goal"] for s in states])),
        "goal_position": torch.as_tensor(env.goal).clone(),
        "collisions": torch.tensor(collisions, dtype=torch.bool),
        "shortest_path": torch.as_tensor(env.initial_path).clone(),
        "metrics": {"success": success, "steps": len(actions), "termination": reason,
                    "error": error, "path_length": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
                    "initial_geodesic_distance": env.initial_distance,
                    "final_distance": final_distance},
    }
    # Optional reference-runtime evidence, without changing the legacy schema.
    for source, target in [('rotation_xyzw', 'rotations_xyzw'),
                           ('visualization', 'visualization_frames')]:
        if source in states[0]:
            episode[target] = torch.from_numpy(np.stack([s[source] for s in states]))
    if hasattr(env, 'reference_metrics'):
        metrics = env.reference_metrics()
        episode['metrics']['success'] = reason == 'stop' and bool(metrics['success'])
        episode['metrics']['habitat_lab'] = metrics
    return episode


if __name__ == "__main__":
    raise SystemExit("Use python scripts/smoke_habitat.py from the repository root.")
