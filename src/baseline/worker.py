"""Persistent binary stdio bridge between feature and Habitat Python environments."""
import argparse
import json
import subprocess
import sys
import numpy as np
from .features import FeatureConfig, official_encoder

MAX_PAYLOAD = 128 * 1024 * 1024


def send_message(stream, header, array=None):
    data = b'' if array is None else np.ascontiguousarray(array).tobytes()
    if len(data) > MAX_PAYLOAD:
        raise ValueError('Feature IPC payload exceeds bound')
    header = dict(header, bytes=len(data))
    if array is not None:
        header.update(shape=list(array.shape), dtype=str(array.dtype))
    stream.write((json.dumps(header) + '\n').encode())
    stream.write(data)
    stream.flush()


def read_message(stream):
    line = stream.readline(65537)
    if not line:
        raise EOFError('Feature worker closed its stream')
    if len(line) > 65536 or not line.endswith(b'\n'):
        raise ValueError('Invalid feature IPC header')
    header = json.loads(line)
    size = header['bytes']
    if type(size) is not int or not 0 <= size <= MAX_PAYLOAD:
        raise ValueError('Invalid feature IPC payload size')
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError('Truncated feature IPC payload')
        data.extend(chunk)
    array = None
    if size:
        dtype = np.dtype(header['dtype'])
        if dtype not in (np.dtype('uint8'), np.dtype('float32')):
            raise ValueError('Only uint8 RGB and float32 features can cross the bridge')
        shape = header['shape']
        if any(type(x) is not int or x < 1 for x in shape) or np.prod(shape) * dtype.itemsize != size:
            raise ValueError('Feature IPC shape/payload mismatch')
        array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    return header, array


class WorkerClient:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        try:
            header, _ = read_message(self.process.stdout)
            if header.get('error'):
                raise RuntimeError(header['error'])
            self.identity = header['identity']
        except BaseException:
            self.close()
            raise

    def encode(self, clip):
        clip = np.asarray(clip)
        if clip.dtype != np.uint8:
            raise ValueError('Worker requires raw uint8 RGB; implicit casts are forbidden')
        send_message(self.process.stdin, {'op': 'encode'}, clip)
        header, result = read_message(self.process.stdout)
        if header.get('error'):
            raise RuntimeError(header['error'])
        if result is None or result.dtype != np.float32 or not np.isfinite(result).all():
            raise RuntimeError('Invalid worker features')
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def worker_command(python, config, source_dir=None, checkpoint=None, device='cuda'):
    command = [str(python), '-u', '-m', 'jepa_navigation.baseline.worker',
               '--feature-config', json.dumps(config.__dict__), '--device', device]
    if not source_dir or not checkpoint:
        raise ValueError('Official worker requires source_dir and checkpoint; no auto-download')
    command += ['--source-dir', str(source_dir), '--checkpoint', str(checkpoint)]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-config', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    output = sys.stdout.buffer
    try:
        config = FeatureConfig(**json.loads(args.feature_config))
        encoder = official_encoder(args.source_dir, args.checkpoint, config, args.device)
        send_message(output, {'identity': encoder.identity})
        while True:
            try:
                header, clip = read_message(sys.stdin.buffer)
            except EOFError:
                break
            try:
                if header['op'] != 'encode' or clip is None:
                    raise ValueError('Expected an encode request')
                result = encoder.encode(clip)
                send_message(output, {'ok': True}, result)
            except Exception as exc:
                send_message(output, {'error': f'{type(exc).__name__}: {exc}'})
    except Exception as exc:
        send_message(output, {'error': f'{type(exc).__name__}: {exc}'})
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
