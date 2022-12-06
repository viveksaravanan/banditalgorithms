
#!/usr/bin/env python

import sys
import os
from pathlib import Path
import subprocess

sys.path.insert(0, 'src')

config_root = Path(__file__).resolve().parent / "config"


def main(targets):
    if 'test' in targets:
        configs = (config_root / "test").rglob("*.py")
    elif 'all' in targets:
        configs = (config_root/ "all").rglob("*.py")
    elif 'etc' in targets:
        configs = (config_root / "all" / "etc").rglob("*.py")
    elif 'ucb' in targets:
        configs = (config_root / "all" / "ucb").rglob("*.py")
    elif 'ts' in targets:
        configs = (config_root / "all" / "ts").rglob("*.py")
    elif 'linear' in targets:
        configs = (config_root / "all" /"linear").rglob("*py")

    for x in configs:   
        subprocess.call(
            args=[
                "python",
                str(x)
            ]
        )

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)