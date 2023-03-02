import argparse

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--server", help="run the viewer part of the script, if not given run the client instead", action="store_true"
)
args = parser.parse_args()


# In this example we show how a Viewer can also send data to the clients other than just receiving data to visualize.
# This can be useful to implement custom behavior that is triggered interactively by the Viewer.
#
# Before reading this example you should be familiar with the 'remote.py' and 'remote_custom_viewer.py' examples which
# go over how to connect to a remote viewer and how to subclass the Viewer class to add custom functionality.

# Run the server or the viewer depending on the command line argument.
if not args.server:
    #
    # Client
    #

    import trimesh

    from aitviewer.remote.renderables.meshes import RemoteMeshes
    from aitviewer.remote.viewer import RemoteViewer

    cube = trimesh.load("resources/cube.obj")

    with RemoteViewer.create_new_process([__file__, "--server"]) as v:
        while True:
            # Get the next message, blocking until one is received.
            msg = v.get_message(block=True)

            # If msg is None the connection was closed by the remote viewer and we just exit.
            if msg is None:
                break

            # Print the message that we received.
            print(f"{msg}")

            # Send a cube with name and position as specified in the message.
            RemoteMeshes(
                v, cube.vertices, cube.faces, name=msg["name"], position=msg["position"], flat_shading=True, scale=0.5
            )

else:
    #
    # Server
    #

    import imgui

    from aitviewer.configuration import CONFIG as C
    from aitviewer.viewer import Viewer

    position = (0, 0, 0)
    name = "Test"

    class CustomViewer(Viewer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Add a new gui function for drawing some custom widgets.
            self.gui_controls["custom"] = self.gui_custom

        # Define a new gui function for drawing some custom widgets.
        def gui_custom(self):
            global position, name

            # Crate a new window with a default position and size.
            imgui.set_next_window_position(self.window_size[0] * 0.8, 275, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(self.window_size[0] * 0.1, 110, imgui.FIRST_USE_EVER)
            expanded, _ = imgui.begin("Custom window")
            if expanded:
                _, position = imgui.drag_float3("Position", *position, 1e-2, format="%.2f")
                _, name = imgui.input_text("Name", name, 64)

                if imgui.button("Make cube"):
                    # When the button is pressed send a message to all clients.
                    # The first argument to this function can be any python object that can be serialized with pickle.
                    # The 'get_message()' method on the client returns the object obtained by deserializing the pickle data.
                    self.send_message({"name": name, "position": position})
            imgui.end()

    # Create and run the viewer with the server enabled.
    C.update_conf({"server_enabled": True})
    v = CustomViewer()
    v.run()
