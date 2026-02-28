#!/usr/bin/env python3
"""
Hotfix: skip self-copy error in apply_features_8_9.py
Run from ~/hexagrid/api/:  python hotfix_selfcopy.py
"""
from pathlib import Path

script = Path(__file__).parent / "apply_features_8_9.py"
content = script.read_text()

OLD = '''\
        shutil.copy2(src, dst)
        ok(f"Copied: {fname}")'''

NEW = '''\
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            ok(f"Copied: {fname}")
        else:
            ok(f"Already in place: {fname}")'''

if OLD in content:
    script.write_text(content.replace(OLD, NEW, 1))
    print("✓ Fixed — re-run: python apply_features_8_9.py")
else:
    print("✓ Already fixed or not needed")
