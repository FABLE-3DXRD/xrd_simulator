# Disable all logging of xfab as default; it is very verbose!
from xrd_simulator import utils
utils._set_xfab_logging(disabled=True)

# Expose configure_device at package level for convenient GPU/CPU control
from xrd_simulator.cuda import configure_device, get_selected_device

__all__ = ["configure_device", "get_selected_device"]
