#!/usr/bin/env python3
# Convert a numpy2-saved PICO capture pkl into a numpy-agnostic motion pkl that
# twist2's MotionLib can load for the feeder.
#   1) dump every array as nested Python lists -> pickle has NO numpy -> loads anywhere
#      (fixes numpy2 'numpy._core' incompat in twist2 numpy1.23)
#   2) PICO lbp=None but MotionLib requires it; feeder mimic obs does NOT use lbp ->
#      stub zeros of shape (n_frames, n_bodies, 3). Reference tracking unaffected.
# Run in an env that can read the source (gmr / numpy2).
import pickle
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
d = pickle.load(open(src, "rb"))

root_pos = np.asarray(d["root_pos"], dtype=np.float64)
root_rot = np.asarray(d["root_rot"], dtype=np.float64)
dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)
fps = float(d["fps"])
n_frames = root_pos.shape[0]
n_bodies = len(d["link_body_list"]) if d.get("link_body_list") is not None else 13

lbp = d.get("local_body_pos")
if lbp is None:
    lbp = np.zeros((n_frames, n_bodies, 3), dtype=np.float64)
else:
    lbp = np.asarray(lbp, dtype=np.float64)

out = {
    "root_pos": root_pos.tolist(),
    "root_rot": root_rot.tolist(),
    "dof_pos": dof_pos.tolist(),
    "local_body_pos": lbp.tolist(),
    "fps": fps,
}
with open(dst, "wb") as f:
    pickle.dump(out, f, protocol=2)

print("converted %s -> %s" % (src, dst))
print("  frames=%d fps=%.2f dur=%.2fs dof=%d n_bodies=%d"
      % (n_frames, fps, n_frames / fps, dof_pos.shape[1], n_bodies))
