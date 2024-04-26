from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import RichProgressBar
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "sweeper",
        "data",
        "module",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    hide: Sequence[str] = ("_import_path"),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "bold white"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    def add_branch_content(branch, config_group, style="yaml", resolve=False):
        """Adds branch content based on the type of config_group."""
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, style))

    for field in queue:
        if field in hide:
            continue

        branch = tree.add(field, style=style, guide_style=style)

        if field == "module":
            for subfield in ["layers", "model", "lit_module"]:
                module_branch = branch.add(subfield, style=style, guide_style=style)
                config_group = cfg[field].get(subfield)
                subfield_resolve = {"layers": True, "model": False, "lit_module": False}
                if config_group is not None:  # Ensure config_group exists before proceeding
                    add_branch_content(
                        module_branch, config_group, resolve=subfield_resolve[subfield]
                    )
        else:
            config_group = cfg.get(field)  # Safer access using .get()
            if config_group is not None:  # Check if config_group exists
                add_branch_content(branch, config_group, resolve=resolve)

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.run_specs.tags:
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.run_specs.tags = tags

        log.info(f"Tags: {cfg.run_specs.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.run_specs.tags, file=file)


class CustomRichProgressBar(RichProgressBar):
    """_summary_

    Args:
        RichProgressBar (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_metrics(self, trainer, pl_module):
        """_summary_

        Args:
            trainer (_type_): _description_
            pl_module (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_metrics = super().get_metrics(trainer, pl_module)
        metrics_keys = ["train/loss_step", "train/loss_epoch", "val/loss", "lr"]

        return OrderedDict((key, all_metrics[key]) for key in metrics_keys if key in all_metrics)
