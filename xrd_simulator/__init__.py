# Disable all loging of xfab as default; it is very verbose!
from xrd_simulator import utils
utils._set_xfab_logging(disabled=True)

# Expose set_device at package level for convenient GPU/CPU control
from xrd_simulator.utils import set_device

__all__ = ['set_device']