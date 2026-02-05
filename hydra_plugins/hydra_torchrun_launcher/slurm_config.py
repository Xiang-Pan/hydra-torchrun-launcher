# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : slurm_config.py
"""Configuration dataclasses for SLURM/Submitit integration."""

from dataclasses import dataclass, field
from typing import List, Optional

from packaging.version import Version
from importlib.metadata import version

from hydra.core.config_store import ConfigStore

from .config import TorchDistributedLauncherConf


@dataclass
class SlurmAutoDetectConf:
    """Configuration for SLURM auto-detection mode."""

    enabled: bool = True
    override_nnodes: Optional[str] = None
    override_nproc_per_node: Optional[str] = None
    override_master_port: Optional[int] = None


@dataclass
class SlurmSubmitConf:
    """Configuration for submitit SLURM submission."""

    # Basic SLURM settings
    partition: Optional[str] = None
    nodes: int = 1
    gpus_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    mem_gb: Optional[int] = None
    timeout_min: int = 60

    # SLURM job settings
    job_name: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    exclude: Optional[str] = None
    nodelist: Optional[str] = None

    # Output settings
    stderr_to_stdout: bool = False

    # Additional SLURM options
    additional_parameters: Optional[dict] = None

    # Submitit-specific settings
    slurm_array_parallelism: int = 256
    slurm_setup: Optional[List[str]] = None
    slurm_srun_args: Optional[List[str]] = None


@dataclass
class TorchrunSlurmLauncherConf(TorchDistributedLauncherConf):
    """Configuration for the SLURM-aware torchrun launcher.

    Extends the base TorchDistributedLauncherConf with SLURM-specific options.

    Modes:
    - "auto": Auto-detect SLURM environment when running inside salloc/srun,
              falls back to direct launch if not in SLURM
    - "submitit": Submit a SLURM job via submitit that runs torchrun inside
    - "direct": Use the original torchrun behavior (no SLURM integration)
    """

    _target_: str = (
        "hydra_plugins.hydra_torchrun_launcher.slurm_launcher.TorchrunSlurmLauncher"
    )

    # Launch mode
    mode: str = "auto"  # "auto", "submitit", or "direct"

    # SLURM auto-detection settings
    slurm_auto_detect: SlurmAutoDetectConf = field(default_factory=SlurmAutoDetectConf)

    # Submitit SLURM settings
    slurm: SlurmSubmitConf = field(default_factory=SlurmSubmitConf)


# For torch >= 2.3, add the additional fields
if Version(version("torch")) >= Version("2.3"):

    @dataclass
    class TorchrunSlurmLauncherConf(TorchrunSlurmLauncherConf):
        logs_specs: Optional[str] = None
        local_ranks_filter: str = ""
        numa_binding: Optional[str] = None
        event_log_handler: str = "null"


# Register the configuration
ConfigStore.instance().store(
    group="hydra/launcher", name="torchrun_slurm", node=TorchrunSlurmLauncherConf
)
