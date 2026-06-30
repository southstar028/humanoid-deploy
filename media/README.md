# media/

Where to put the demo clips referenced from the [Demo](../README.md#demo) table.

This public repo **git-ignores `*.mp4` by design** (no large binaries — see `.gitignore`), so
prefer a link over a committed file:

1. **YouTube** — paste the watch URL into the Demo table. *(recommended)*
2. **GitHub drag-drop** — in the GitHub web editor, drag an mp4 into the README; GitHub hosts
   it as an `…/assets/…` link. Nothing is committed to the repo.

If you really want a small clip *in* the repo, force-add it here (it is ignored otherwise):

```bash
git add -f media/<clip>.mp4
```

Suggested clips (from the local deploy package):

| suggested file | source clip |
|---|---|
| `sim2sim_gate.mp4` | sim2sim parity gate — 60 s stand + turning gait + dynamic transition |
| `loopback_stand.mp4` | DDS loopback — standing |
| `loopback_gait.mp4` | DDS loopback — turning gait |
| `loopback_vr.mp4` | DDS loopback — VR-captured motion, no fall |
