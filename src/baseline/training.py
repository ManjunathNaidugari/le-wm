"""Behavior cloning with building-separated checkpoint selection and explicit tiny mode."""
from dataclasses import dataclass, asdict
from collections import OrderedDict
from pathlib import Path
import random
import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .common import ACTIONS, read_json, sha256_file, write_json
from .cache import validate_cache
from .splits import validate_splits, training_definitions
from .policy import DirectPolicy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    projection_dim: int = 32
    hidden_dim: int = 128
    class_weights: bool = True
    tiny_steps: int = 0
    goal_only: bool = False

    def __post_init__(self):
        for key in ('epochs', 'batch_size', 'projection_dim', 'hidden_dim'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if type(self.tiny_steps) is not int or self.tiny_steps < 0:
            raise ValueError('tiny_steps must be nonnegative')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid optimizer hyperparameters')
        if not 0 <= self.seed < 2**31:
            raise ValueError('seed must be in [0,2**31)')


class CachedTransitions(Dataset):
    def __init__(self, cache_dir, manifest, identities, limit=0):
        self.root = Path(cache_dir)
        self.items, self.indices = [], []
        wanted = {tuple(k) for k in identities}
        found = set()
        self.maps = OrderedDict()
        for item in manifest['episodes']:
            key = (item['building'].casefold(), item['identity'].split('/', 1)[1])
            if key not in wanted:
                continue
            found.add(key)
            targets = np.load(self.root / item['targets'], allow_pickle=False)
            actions, goals = targets['actions'].copy(), targets['goals'].copy()
            targets.close()
            item_index = len(self.items)
            self.items.append((item, actions, goals))
            for t in range(item['steps']):
                self.indices.append((item_index, t))
        if found != wanted:
            raise ValueError(f'Split episodes missing from audited cache: {wanted - found}')
        if not self.indices:
            raise ValueError('No training/evaluation transitions')
        if limit and limit < len(self.indices):
            # Deterministic round-robin across present action classes, then time order.
            # A trajectory prefix can contain no STOP or turns, defeating the check.
            buckets = [[] for _ in ACTIONS]
            for pair in self.indices:
                buckets[int(self.items[pair[0]][1][pair[1]])].append(pair)
            selected = []
            depth = 0
            while len(selected) < limit:
                for bucket in buckets:
                    if depth < len(bucket):
                        selected.append(bucket[depth])
                        if len(selected) == limit:
                            break
                depth += 1
            self.indices = sorted(selected)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item_index, t = self.indices[index]
        item, actions, goals = self.items[item_index]
        chunk = next(c for c in item['chunks'] if c['start'] <= t < c['end'])
        key = chunk['file']
        if key not in self.maps:
            self.maps[key] = np.load(self.root / key, mmap_mode='r', allow_pickle=False)
            if len(self.maps) > 4:
                self.maps.popitem(last=False)
        self.maps.move_to_end(key)
        features = np.array(self.maps[key][t-chunk['start']], copy=True)
        return torch.from_numpy(features), torch.from_numpy(goals[t].copy()), torch.tensor(int(actions[t]), dtype=torch.long)

    def goals_and_actions(self):
        return (np.stack([self.items[i][2][t] for i, t in self.indices]),
                np.array([self.items[i][1][t] for i, t in self.indices]))


def classification_metrics(confusion, loss, count):
    matrix = confusion.cpu().numpy()
    per_action = {}
    for i, name in enumerate(ACTIONS):
        tp, actual, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision, recall = tp / max(1, predicted), tp / max(1, actual)
        per_action[name] = dict(count=actual, precision=precision, recall=recall,
                               f1=2 * precision * recall / max(1e-12, precision + recall))
    return dict(loss=loss / max(1, count), accuracy=float(matrix.trace() / max(1, count)),
                samples=count, per_action=per_action, confusion_matrix=matrix.tolist())


def run_epoch(model, loader, device, optimizer=None, weights=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    confusion = torch.zeros(4, 4, dtype=torch.long)
    context = torch.enable_grad() if optimizer else torch.inference_mode()
    with context:
        for features, goals, actions in loader:
            features, goals, actions = features.to(device), goals.to(device), actions.to(device)
            logits = model(features, goals)
            loss = torch.nn.functional.cross_entropy(logits, actions, weight=weights)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # Report unweighted per-transition NLL so split comparisons have a defined denominator.
            total += float(torch.nn.functional.cross_entropy(logits.detach(), actions, reduction='sum').item())
            count += len(actions)
            pairs = actions.cpu() * 4 + logits.detach().argmax(1).cpu()
            confusion += torch.bincount(pairs, minlength=16).reshape(4, 4)
    return classification_metrics(confusion, total, count)


def require_real_encoder(encoder):
    if encoder.get('test_encoder') is not False or encoder.get('backbone') != 'official_vjepa2_vit_large':
        raise ValueError('Test encoder or unknown features are forbidden; use official V-JEPA features')


def train_policy(cache_dir, split_path, output_dir, config, device='cpu'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cache = validate_cache(cache_dir)
    encoder = cache['signature']['encoder']
    require_real_encoder(encoder)
    plan = validate_splits(read_json(split_path))
    if plan['purpose'] == 'integration' and not config.tiny_steps:
        raise ValueError('Pilot integration split requires explicit tiny_steps mode')
    from .common import identity
    train_keys = [identity(e) for e in training_definitions(plan, 'train')]
    train = CachedTransitions(cache_dir, cache, train_keys, config.tiny_steps)
    if config.tiny_steps:
        development = train
        validation_label = 'tiny subset resubstitution; integration check, no generalization claim'
    else:
        development = CachedTransitions(cache_dir, cache, [identity(e) for e in training_definitions(plan, 'development')])
        validation_label = 'building-disjoint development; final evaluation never selects checkpoints'
    # Validate cache/source definitions against the saved split; source IDs alone are insufficient.
    keys_to_definition = {tuple(identity(e)): e for phase in ('train', 'development') for e in plan['splits'][phase]['episodes']}
    for item, _, _ in train.items + development.items:
        definition = keys_to_definition[(item['building'].casefold(), item['identity'].split('/', 1)[1])]
        source = torch.load(item['source_path'], map_location='cpu', weights_only=True)['metadata']
        if source['source_episode_sha256'] != definition['source_episode_sha256'] or source['source_definition'] != definition['source_definition']:
            raise ValueError('Split/cache source definitions differ')
    simulator = train.items[0][0]['effective_settings']['simulator']
    navigation = {k: train.items[0][0]['effective_settings']['navigation'][k] for k in ('forward_step_m', 'turn_degrees', 'goal_radius_m', 'max_y_delta_m')}
    for item, _, _ in train.items + development.items:
        settings = item['effective_settings']
        if settings['simulator'] != simulator or any(settings['navigation'][k] != v for k, v in navigation.items()):
            raise ValueError('Training/development simulator settings differ')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    goals, labels = train.goals_and_actions()
    counts = np.bincount(labels, minlength=4)
    model_config = dict(feature_dim=encoder['feature_dimension'], cells=2 * encoder['feature_config']['grid_size']**2,
                        projection_dim=config.projection_dim, hidden_dim=config.hidden_dim, goal_only=config.goal_only)
    model = DirectPolicy(**model_config).to(device)
    model.goal_mean.copy_(torch.as_tensor(goals.mean(0), device=device))
    model.goal_std.copy_(torch.as_tensor(np.maximum(goals.std(0), 1e-3), device=device))
    weights = None
    if config.class_weights:
        values = np.where(counts > 0, len(labels) / (4 * np.maximum(counts, 1)), 0).astype(np.float32)
        weights = torch.from_numpy(values).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    dev_loader = DataLoader(development, batch_size=config.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    metadata = dict(schema_version=1, workflow='direct_navigation_policy', train_config=asdict(config), model_config=model_config,
                    encoder=encoder, cache_fingerprint=cache['fingerprint'], splits=plan,
                    split_sha256=sha256_file(split_path), simulator_config=simulator, navigation_config=navigation,
                    actions=ACTIONS, recorded_to_habitat=[1, 2, 3, 0], goal_convention='forward,left,up (metres); no goal orientation',
                    train_class_counts=dict(zip(ACTIONS, counts.tolist())),
                    normalization_source='training transitions only; tiny mode uses only its selected transitions',
                    tiny_selection='deterministic round-robin across present action classes, then chronological order',
                    selected_train_transitions=[dict(identity=train.items[i][0]['identity'], action_timestep=t) for i, t in train.indices] if config.tiny_steps else None,
                    validation_label=validation_label, integration_only=bool(config.tiny_steps),
                    expert_sources=[dict(identity=i[0]['identity'], source_sha256=i[0]['source_sha256'], provenance=i[0]['provenance']) for i in train.items + development.items],
                    packages=dict(torch=str(torch.__version__), numpy=str(np.__version__)), epochs=[])
    write_json(root / 'training.json', metadata)
    best_loss = float('inf')
    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer, weights)
        dev_metrics = run_epoch(model, dev_loader, device)
        metadata['epochs'].append(dict(epoch=epoch + 1, train=train_metrics, development=dev_metrics))
        improved = dev_metrics['loss'] < best_loss
        if improved:
            best_loss = dev_metrics['loss']
            metadata.update(best_epoch=epoch+1, best_development_loss=best_loss)
        metadata['checkpoint_epoch'] = epoch+1
        state = dict(metadata=copy.deepcopy(metadata), model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch=epoch+1)
        tmp = root / 'last.pt.partial'
        torch.save(state, tmp)
        tmp.replace(root / 'last.pt')
        if improved:
            tmp = root / 'best.pt.partial'
            torch.save(state, tmp)
            tmp.replace(root / 'best.pt')
        write_json(root / 'training.json', metadata)
        print(f"epoch={epoch+1} train_nll={train_metrics['loss']:.5f} dev_nll={dev_metrics['loss']:.5f} dev_accuracy={dev_metrics['accuracy']:.3f}", flush=True)
    return metadata


def load_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    if meta['workflow'] != 'direct_navigation_policy' or meta['schema_version'] != 1 or meta['actions'] != ACTIONS or meta['recorded_to_habitat'] != [1, 2, 3, 0]:
        raise ValueError('Policy checkpoint action/schema mismatch')
    require_real_encoder(meta['encoder'])
    if any(source['provenance'].get('synthetic') for source in meta.get('expert_sources', [])):
        raise ValueError('Synthetic training sources are forbidden in policy checkpoints')
    validate_splits(meta['splits'])
    model = DirectPolicy(**meta['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval().requires_grad_(False)
    return model, meta
