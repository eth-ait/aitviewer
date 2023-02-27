"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class Configuration(object):
    class __Configuration:
        def next_gui_id(self):
            self._gui_counter += 1
            return self._gui_counter

        def __init__(self):
            # Load the default configurations.
            self._conf = OmegaConf.load(os.path.join(os.path.dirname(__file__), "aitvconfig.yaml"))

            # Check if we have a config file in AITVRC env variable.
            if os.environ.get("AITVRC", None) is not None:
                env_conf = os.environ["AITVRC"]
                if os.path.isdir(env_conf):
                    conf = OmegaConf.load(os.path.join(env_conf, "aitvconfig.yaml"))
                else:
                    conf = OmegaConf.load(env_conf)
                self._conf.merge_with(conf)

            # Check if we have a config file in the working directory which overrides all previous configs.
            local_conf = os.path.join(os.getcwd(), "aitvconfig.yaml")
            if os.path.exists(local_conf):
                conf = OmegaConf.load(local_conf)
                self._conf.merge_with(conf)

            self._gui_counter = 0
            self._gpu_available = torch.cuda.is_available()

        def update_conf(self, conf_obj):
            """Update the configuration with another configuration file or another OmegaConf configuration object."""
            if isinstance(conf_obj, str):
                conf_obj = OmegaConf.load(conf_obj)
            else:
                assert isinstance(conf_obj, DictConfig) or isinstance(conf_obj, dict)
            self._conf.merge_with(conf_obj)

        def __getattr__(self, item):
            if hasattr(self._conf, item):
                # Some attributes of the config are converted to torch objects automatically.
                if item == "device":
                    return torch.device(self._conf.get("device", "cuda:0") if self._gpu_available else "cpu")
                elif item == "f_precision":
                    return getattr(torch, "float{}".format(self._conf.get("f_precision", 32)))
                elif item == "i_precision":
                    return getattr(torch, "int{}".format(self._conf.get("i_precision", 64)))
                else:
                    return getattr(self._conf, item)
            else:
                # Default behavior.
                return self.__getattribute__(item)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Configuration.instance:
            Configuration.instance = Configuration.__Configuration()
        return Configuration.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONFIG = Configuration()
