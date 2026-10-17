"""JSON persistence shared by execution, analysis and dashboard services."""
import json
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())
