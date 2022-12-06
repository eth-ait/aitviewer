---
title: Headless Rendering
layout: default
---

Headless rendering is supported. An example script below saves frames to disk on a server. 

```python
from aitviewer.headless import HeadlessRenderer
r = HeadlessRenderer()
for frame in frames:
   pose, shape = inference(frame)
   seq = SMPLSequence(smpl_layer, pose_body=np.concatenate(poses, axis=0), betas=np.concatenate(shapes, axis=0)
   r.scene.add(seq)
   r.save_frame(...)
   r.scene.remove(seq)
```

It may be useful to run the following command before rendering on a server with no connected monitors:
```bash
export DISPLAY=:0.0
Xvfb :0 -screen 0 640x480x24 &
```

See this [Github Issue](https://github.com/eth-ait/aitviewer/issues/10) for more details.
