"""Artifact identities and atomic writes shared by the baseline tools."""
import hashlib
import json
from pathlib import Path
import os
from contextlib import contextmanager

ACTIONS = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']
PILOT_BUILDINGS = ['Adrian', 'Albertville', 'Anaheim']


def sha256_file(path):
    # Keep the encoder worker independent of Habitat/video imports.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def building(scene):
    return Path(scene).stem


def identity(definition):
    return [building(definition['source_scene_id']).casefold(), str(definition['source_episode_id'])]


def contained_file(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if root not in path.parents:
        raise ValueError(f'Artifact path escapes its root: {filename}')
    return path


@contextmanager
def artifact_lock(root):
    """One writer; interrupted locks require explicit review/removal, never auto-stealing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.writer.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()
