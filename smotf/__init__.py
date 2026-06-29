"""s-motf: state Mixture-of-Transformers + flow-matching controller.

Step 0 public API is ``load_config``: it loads ``configs/default.yaml`` into a
dotted-access object so every module reads dimensions from one source of truth
(e.g. ``cfg.d``, ``cfg.dims.base``) instead of hard-coding 256 / 12.
"""

from pathlib import Path
from typing import Any
import yaml

#al tells python what functions/classes are expoed if someone runs this 
__all__=["DotDict","load_config"]

#path resolution
_REPO_ROOT=Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG=_REPO_ROOT/"configs"/"default.yaml"

#the dotdict class, rewires python dict so that it can be access via dot-notation
class DotDict(dict):

    #whenever someone writes cfg.d and Python can't find d as a real attribute
    def __getattr__(self,key):
        try:
            value=self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

        if isinstance(value,dict) and not isinstance(value,DotDict):
            value=DotDict(value)
            self[key]=value
        return value
    #overwrites set and del as well
    def __setattr__(self, key,value):
        self[key]=value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    
def load_config(path=None):
    """load a yaml config into class DotDict"""
    path=Path(path) if path is not None else _DEFAULT_CONFIG

    with open(path,'r') as f:
        raw=yaml.safe_load(f)#safeload converts the yaml into python dict

    if not isinstance(raw,dict):
        raise ValueError(f"config at {path} did not parse to a mapping")
    
    return DotDict(raw)