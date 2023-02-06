---
title: Remote Viewer
layout: default
nav_order: 2
---

# Remote Viewer

![alt-text-1](/aitviewer/assets/remote_diagram.svg)

The viewer can be used as a server to visualize data sent by clients over the network.
Clients can connect to a viewer by creating a `RemoteViewer` object and they can use it to send data to the viewer for visualization.
Here is a short example of how to connect and send data to a viewer running locally (see  [`remote.py`](https://github.com/eth-ait/aitviewer/blob/main/examples/remote.py) for a complete example):

```python
from aitviewer.remote.viewer import RemoteViewer
from aitviewer.remote.renderables.meshes import RemoteMeshes

# Load some mesh data from disk.
cube = trimesh.load("resources/cube.obj")

# Connect to the remote viewer.
v = RemoteViewer("127.0.0.1")

# Create a mesh in the remote viewer for visualization.
m = RemoteMeshes(v, cube.vertices, cube.faces)
```

Differently from the normal `Viewer` object, the `RemoteViewer` object only represents a connection to a running viewer, therefore there is no need to call a `viewer.run()` method. Creating renderable objects such as `RemoteMeshes` immediately sends data to the actual viewer, which takes care of the visualization and interactions.

## Remote Renderables

The `aitviewer.remote.renderables.*` modules provide a set of classes mirroring the renderable classes in `aitviewer.renderables.*` with the word `Remote` in front of the renderable name (e.g. `RemoteMeshes` and `Meshes`). Their constructors take an extra `RemoteViewer` object as the first parameter and all other arguments are forwarded to the respective renderable class. After a remote renderable is created, the following methods can be used to add, update and remove frames of data:

- `r.add_frames()`: append new frames to the remote object, increasing the length of the sequence.
- `r.update_frames()`: update existing frames of the remote object with new data.
- `r.remove_frames()`: remove existing frames from the remote object.
- `r.delete()`: delete the remote object from the viewer.

Currently, the following remote renderable classes are supported:

- `RemoteMeshes`
- `RemoteSMPLSequence`
- `RemoteSpheres`
- `RemoteLines`
- `RemoteArrows`
- `RemoteRigidBodies`

## Custom Message Processing

It is possible to add custom functionality to a viewer when messages are received by subclassing the `Viewer` class and overriding the `viewer.process_message()` method. The signature of this method looks like this:

```python
def process_message(self, type: Message, remote_uid: int, args: list, kwargs: dict, client: tuple[str, str]):
```
The parameters of this message are used as follows:
- `type`: an enumeration representing the type of the message as defined in [`aitviewer/remote/message.py`](https://github.com/eth-ait/aitviewer/blob/main/aitiviewer/remote/message.py).
- `remote_uid`: a unique identifier for the node used by the client. Together with `client` it can be used to look up the corresponding node on the viewer using the `viewer.get_node_from_remote_uid()` method.
- `args` and `kwargs`: the positional and keyword arguments passed to the `send_message()` method.
- `client`: a `(ip, port)` tuple that represent the remote address of the client. This can be used to distinguish between messages from multiple clients.

See the example [`remote_custom_viewer.py`](https://github.com/eth-ait/aitviewer/blob/main/examples/remote_custom_viewer.py) for more details.

## Server Setup

The server functionality of the viewer is disabled by default, but can be enabled by setting the configuration property `server_enabled: True` (see [Configuration]({{ site.baseurl }}{% link configuration.md %}) for details on configuring the viewer) or by launching an empty viewer with the command `pip -m aitviewer.server`. Communication happens with WebSockets over the TCP port specified in the configuration property `server_port` which is number 8417 by default.

{: .warning }
> The communication between clients and servers happens over WebSockets without any form of authentication or encryption, additionally no measure is taken by the server to avoid damages from malicious clients. Therefore, it is highly recommended to run the viewer with the server enabled only on hosts that are not exposed to an untrusted network.
