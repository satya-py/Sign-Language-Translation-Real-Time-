"""Offline tests: zip range reader + bit-exact preprocessing vs. the paper's code.

Uses real MediaPipe-Holistic .pose files (same component layout as iSign).
Run:  python tests/test_exactness.py <dir with .pose files>
"""

import glob
import os
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import isign_zip
from features import pose_bytes_to_features, preprocess_array
from paper_pose_utils import load_pose_file, preprocess_pose

pose_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(pose_dir, "*.pose")))[:12]
assert files, "no .pose files found"

# 1. Build a zip (mixed stored/deflated), split it into 4 parts like iSign.
tmp = tempfile.mkdtemp()
zpath = os.path.join(tmp, "all.zip")
with zipfile.ZipFile(zpath, "w") as z:
    for i, f in enumerate(files):
        z.write(f, f"iSign-poses_v1.1/{os.path.basename(f)}",
                compress_type=zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED)
blob = open(zpath, "rb").read()
cuts = [0, len(blob) // 4, len(blob) // 2, 3 * len(blob) // 4, len(blob)]
parts = []
for k in range(4):
    p = os.path.join(tmp, isign_zip.PART_NAMES[k])
    open(p, "wb").write(blob[cuts[k]:cuts[k + 1]])
    parts.append(p)
src = isign_zip.MultiPartSource(parts, [os.path.getsize(p) for p in parts],
                                isign_zip.local_fetcher())
members = isign_zip.list_members(src)
assert len(members) == len(files), (len(members), len(files))
by_part = [len(isign_zip.members_in_part(members, src, k)) for k in range(4)]
assert sum(by_part) == len(files), by_part
for m in members:
    original = open(os.path.join(pose_dir, os.path.basename(m.filename)), "rb").read()
    assert isign_zip.read_member(src, m) == original, m.filename
print(f"[ok] zip reader: {len(members)} members byte-identical, per-part counts {by_part}")

# 2. Features + preprocessing identical to the paper's preprocess_pose.
for f in files:
    raw = pose_bytes_to_features(open(f, "rb").read())
    assert np.array_equal(raw, load_pose_file(f), equal_nan=True)
    assert raw.shape[1] == 300, raw.shape
    for pose_dim, motion in ((300, False), (600, True)):
        ours, n1 = preprocess_array(raw, 500, "zscore", pose_dim, motion)
        theirs, n2 = preprocess_pose(f, 500, "zscore", pose_dim=pose_dim, use_motion=motion)
        assert n1 == n2 and np.array_equal(ours, theirs), (f, pose_dim)
print(f"[ok] features: {len(files)} files, spatial (300) and motion (600) bit-identical "
      f"to paper preprocess_pose; e.g. raw shape {raw.shape}")
