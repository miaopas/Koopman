import rootutils
from lightning.pytorch import Callback

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CheckUnusedParametersCallback(Callback):
    """_summary_

    Args:
        Callback (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.has_checked_unused_parameters = False

    def on_after_backward(self, trainer, pl_module):
        if self.has_checked_unused_parameters:
            return
        log.info("Training is starting. Checking for unused parameters...")
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is None:
                log.warning(f"Warning: Parameter '{name}' was not used during the forward pass.")
        self.has_checked_unused_parameters = True
