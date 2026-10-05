"""Export the saved RGB states, preserving the T+1 frame count."""
from pathlib import Path
import imageio.v2 as imageio
from jepa_navigation.data.habitat_dataset import validate_trajectory


def export_video(episode, path, fps=10, composite=False):
    if fps < 1:
        raise ValueError('fps must be positive')
    validate_trajectory(episode, check_rgb_quality=False)
    frames = episode.get('visualization_frames') if composite else None
    if composite and (frames is None or len(frames) != len(episode['actions']) + 1 or
                      frames.dtype != episode['observations'].dtype or frames.ndim != 4 or
                      frames.shape[-1] != 3 or min(frames.shape[1:3]) < 2):
        raise ValueError('Composite video requires T+1 uint8 Habitat visualization frames')
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Video already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed encode cannot leave a seemingly complete video at the final path.
    tmp = path.with_name(path.stem + '.partial.mp4')
    try:
        with imageio.get_writer(str(tmp), fps=fps, codec='libx264', macro_block_size=1) as writer:
            for frame in (frames if composite else episode['observations']):
                writer.append_data(frame.numpy() if composite else frame.permute(1, 2, 0).numpy())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
