"""Locate active inputs/results and preserved historical material."""
from pathlib import Path

ARCHIVE_DIRECTORY = '2026-10-17'


def historical_root(root):
    return Path(root) / 'archive' / ARCHIVE_DIRECTORY


def input_path(value, root):
    """Prefer existing inputs; resolve former repository paths after archival."""
    root, path = Path(root), Path(value)
    if path.is_file():
        return path
    if path.is_absolute():
        try:
            relative = path.relative_to(root)
        except ValueError:
            return path
    else:
        relative = path
    active = root / relative
    if active.is_file():
        return active
    archived = historical_root(root) / relative
    return archived if archived.is_file() else active


def study_roots(root):
    return (Path(root) / 'results' / 'studies', historical_root(root) / 'results' / 'studies')


def run_directories(root):
    """List readable runs, with active paths taking precedence over duplicates."""
    seen = set()
    for parent in (Path(root)/'results/experiments', *study_roots(root)):
        for path in sorted(parent.glob('*')):
            if path.name in seen:
                continue
            if path.is_dir() and ((path/'config.json').is_file() or (path/'manifest.json').is_file()):
                seen.add(path.name)
                yield path
