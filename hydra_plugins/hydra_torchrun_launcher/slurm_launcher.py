# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : slurm_launcher.py
"""SLURM-aware torchrun launcher with submitit support."""

import logging
from typing import Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from hydra.types import HydraContext
from hydra.core.utils import JobReturn
from hydra.plugins.launcher import Launcher
from hydra.types import TaskFunction

from .slurm_utils import detect_slurm_environment, configure_torchrun_for_slurm

logger = logging.getLogger(__name__)


class TorchrunSlurmLauncher(Launcher):
    """A SLURM-aware torchrun launcher with multiple launch modes.

    Supports three modes:
    - "auto": Auto-detect SLURM environment when running inside salloc/srun
    - "submitit": Submit SLURM jobs via submitit that run torchrun inside
    - "direct": Use original torchrun behavior without SLURM integration
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None
        self.launch_config = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

        launcher_conf = config.hydra.launcher
        mode = launcher_conf.get("mode", "auto")

        if mode == "submitit":
            # For submitit mode, we defer setup until we're inside the SLURM job
            logger.info("SLURM launcher configured in submitit mode")
            self._setup_submitit()
        elif mode == "auto":
            self._setup_auto_mode(launcher_conf)
        else:
            # Direct mode - use standard setup
            self._setup_direct()

    def _setup_direct(self) -> None:
        """Setup for direct mode - use standard torchrun configuration."""
        from . import _core

        _core.setup(
            launcher=self,
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=self.config,
        )

    def _setup_auto_mode(self, launcher_conf: DictConfig) -> None:
        """Setup for auto mode - detect SLURM and configure accordingly."""
        slurm_auto_detect = launcher_conf.get("slurm_auto_detect", {})
        enabled = slurm_auto_detect.get("enabled", True)

        if enabled:
            override_port = slurm_auto_detect.get("override_master_port")
            slurm_env = detect_slurm_environment(override_master_port=override_port)

            if slurm_env.is_slurm:
                logger.info(
                    f"SLURM environment detected (job_id={slurm_env.job_id}). "
                    f"Auto-configuring torchrun."
                )
                self._setup_with_slurm_env(launcher_conf, slurm_env, slurm_auto_detect)
                return

        # Not in SLURM or auto-detect disabled, use direct mode
        logger.info("No SLURM environment detected. Using direct torchrun mode.")
        self._setup_direct()

    def _setup_with_slurm_env(
        self, launcher_conf: DictConfig, slurm_env, slurm_auto_detect: dict
    ) -> None:
        """Setup torchrun with SLURM-detected values."""
        from torch.distributed.run import config_from_args
        from omegaconf import open_dict

        # Convert to dict for modification
        conf_dict = OmegaConf.to_container(launcher_conf, resolve=True)

        # Apply SLURM configuration
        override_nnodes = slurm_auto_detect.get("override_nnodes")
        override_nproc = slurm_auto_detect.get("override_nproc_per_node")
        conf_dict = configure_torchrun_for_slurm(
            conf_dict,
            slurm_env,
            override_nnodes=override_nnodes,
            override_nproc_per_node=override_nproc,
        )

        # Convert back to DictConfig and update config
        with open_dict(self.config):
            for key, value in conf_dict.items():
                if key not in ("_target_", "mode", "slurm_auto_detect", "slurm"):
                    self.config.hydra.launcher[key] = value

        # Now setup with the updated config
        launch_config = config_from_args(self.config.hydra.launcher)[0]
        self.launch_config = launch_config

        logger.info(
            f"\nTorchrunSlurmLauncher params (SLURM auto-detected):\n"
            f"\tinit_method: {launch_config.rdzv_endpoint}\n"
            f"\tnnodes: {launch_config.min_nodes}:{launch_config.max_nodes}\n"
            f"\tnproc_per_node: {launch_config.nproc_per_node}"
        )

    def _setup_submitit(self) -> None:
        """Setup for submitit mode."""
        # Verify submitit is available
        try:
            import submitit  # noqa: F401
        except ImportError:
            raise ImportError(
                "submitit is required for SLURM submitit mode. "
                "Install it with: pip install 'hydra-torchrun-launcher[slurm]'"
            )
        # Actual setup happens when the job runs inside SLURM

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        launcher_conf = self.config.hydra.launcher
        mode = launcher_conf.get("mode", "auto")

        if mode == "submitit":
            return self._launch_via_submitit(job_overrides, initial_job_idx)
        else:
            # Both auto and direct modes use the standard launch
            return self._launch_direct(job_overrides, initial_job_idx)

    def _launch_direct(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """Launch using the standard torchrun mechanism."""
        from . import _core

        return _core.launch(
            launcher=self, job_overrides=job_overrides, initial_job_idx=initial_job_idx
        )

    def _launch_via_submitit(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """Submit jobs to SLURM via submitit."""
        import submitit
        from hydra.core.utils import JobReturn, JobStatus
        from omegaconf import OmegaConf
        import cloudpickle

        launcher_conf = self.config.hydra.launcher
        slurm_conf = launcher_conf.get("slurm", {})

        # Create submitit executor
        executor = submitit.AutoExecutor(folder=self.config.hydra.sweep.dir)

        # Configure SLURM parameters
        slurm_params = {
            "slurm_nodes": slurm_conf.get("nodes", 1),
            "slurm_time": slurm_conf.get("timeout_min", 60),
            "slurm_array_parallelism": slurm_conf.get("slurm_array_parallelism", 256),
        }

        if slurm_conf.get("partition"):
            slurm_params["slurm_partition"] = slurm_conf.partition
        if slurm_conf.get("gpus_per_node"):
            slurm_params["slurm_gpus_per_node"] = slurm_conf.gpus_per_node
        if slurm_conf.get("cpus_per_task"):
            slurm_params["slurm_cpus_per_task"] = slurm_conf.cpus_per_task
        if slurm_conf.get("mem_gb"):
            slurm_params["slurm_mem_gb"] = slurm_conf.mem_gb
        if slurm_conf.get("job_name"):
            slurm_params["slurm_job_name"] = slurm_conf.job_name
        if slurm_conf.get("account"):
            slurm_params["slurm_account"] = slurm_conf.account
        if slurm_conf.get("qos"):
            slurm_params["slurm_qos"] = slurm_conf.qos
        if slurm_conf.get("constraint"):
            slurm_params["slurm_constraint"] = slurm_conf.constraint
        if slurm_conf.get("exclude"):
            slurm_params["slurm_exclude"] = slurm_conf.exclude
        if slurm_conf.get("nodelist"):
            slurm_params["slurm_nodelist"] = slurm_conf.nodelist
        if slurm_conf.get("stderr_to_stdout"):
            slurm_params["slurm_stderr_to_stdout"] = slurm_conf.stderr_to_stdout
        if slurm_conf.get("additional_parameters"):
            slurm_params["slurm_additional_parameters"] = slurm_conf.additional_parameters
        if slurm_conf.get("slurm_setup"):
            slurm_params["slurm_setup"] = slurm_conf.slurm_setup
        if slurm_conf.get("slurm_srun_args"):
            slurm_params["slurm_srun_args"] = slurm_conf.slurm_srun_args

        executor.update_parameters(**slurm_params)

        logger.info(f"Submitting {len(job_overrides)} job(s) to SLURM via submitit")

        # Submit jobs
        jobs = []
        for idx, overrides in enumerate(job_overrides):
            job_idx = initial_job_idx + idx

            # Create the callable that will run inside SLURM
            submitit_job = TorchrunSubmititJob(
                config=OmegaConf.to_container(self.config, resolve=True),
                hydra_context_state=cloudpickle.dumps(self.hydra_context),
                task_function=self.task_function,
                overrides=list(overrides),
                job_idx=job_idx,
                launcher_conf=OmegaConf.to_container(launcher_conf, resolve=True),
            )

            job = executor.submit(submitit_job)
            jobs.append((job, job_idx))
            logger.info(f"Submitted job #{job_idx} with SLURM job ID: {job.job_id}")

        # Wait for jobs and collect results
        results = []
        for job, job_idx in jobs:
            try:
                result = job.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Job #{job_idx} failed: {e}")
                results.append(
                    JobReturn(
                        return_value=None,
                        overrides=list(job_overrides[job_idx - initial_job_idx]),
                        status=JobStatus.FAILED,
                    )
                )

        return results


class TorchrunSubmititJob:
    """Callable submitted to SLURM that runs torchrun inside the allocation."""

    def __init__(
        self,
        config: dict,
        hydra_context_state: bytes,
        task_function,
        overrides: list,
        job_idx: int,
        launcher_conf: dict,
    ):
        self.config = config
        self.hydra_context_state = hydra_context_state
        self.task_function = task_function
        self.overrides = overrides
        self.job_idx = job_idx
        self.launcher_conf = launcher_conf

    def __call__(self):
        """Execute torchrun inside the SLURM allocation."""
        import pickle
        import functools
        from pathlib import Path

        from omegaconf import DictConfig, OmegaConf, open_dict
        from hydra.core.singleton import Singleton
        from hydra.core.hydra_config import HydraConfig
        from hydra.core.utils import (
            JobReturn,
            configure_log,
            filter_overrides,
            run_job,
            setup_globals,
        )
        from torch.distributed.run import config_from_args

        from . import _core
        from .slurm_utils import detect_slurm_environment, configure_torchrun_for_slurm

        logger.info(f"TorchrunSubmititJob starting for job #{self.job_idx}")

        # Restore hydra context
        hydra_context = pickle.loads(self.hydra_context_state)

        # Convert config back to DictConfig
        config = OmegaConf.create(self.config)

        # Auto-detect SLURM environment (now running inside the allocation)
        slurm_auto_detect = self.launcher_conf.get("slurm_auto_detect", {})
        override_port = slurm_auto_detect.get("override_master_port")
        slurm_env = detect_slurm_environment(override_master_port=override_port)

        if slurm_env.is_slurm:
            logger.info(
                f"Inside SLURM allocation (job_id={slurm_env.job_id}). "
                f"Configuring torchrun."
            )

            # Apply SLURM configuration to launcher config
            override_nnodes = slurm_auto_detect.get("override_nnodes")
            override_nproc = slurm_auto_detect.get("override_nproc_per_node")
            updated_conf = configure_torchrun_for_slurm(
                self.launcher_conf,
                slurm_env,
                override_nnodes=override_nnodes,
                override_nproc_per_node=override_nproc,
            )

            # Update config with SLURM values
            with open_dict(config):
                for key, value in updated_conf.items():
                    if key not in ("_target_", "mode", "slurm_auto_detect", "slurm"):
                        config.hydra.launcher[key] = value

        # Setup globals
        setup_globals()

        # Configure logging
        configure_log(config.hydra.hydra_logging, config.hydra.verbose)

        # Create sweep directory
        sweep_dir = Path(str(config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        # Build launch config
        launch_config = config_from_args(config.hydra.launcher)[0]

        logger.info(
            f"\nTorchrunSubmititJob params:\n"
            f"\tinit_method: {launch_config.rdzv_endpoint}\n"
            f"\tnnodes: {launch_config.min_nodes}:{launch_config.max_nodes}\n"
            f"\tnproc_per_node: {launch_config.nproc_per_node}"
        )

        # Load sweep config with overrides
        lst = " ".join(filter_overrides(self.overrides))
        logger.info(f"\t#{self.job_idx} : {lst}")

        sweep_config = hydra_context.config_loader.load_sweep_config(
            config, self.overrides
        )

        with open_dict(sweep_config):
            sweep_config.hydra.job.id = slurm_env.job_id or f"job_id_for_{self.job_idx}"
            sweep_config.hydra.job.num = self.job_idx

        HydraConfig.instance().set_config(sweep_config)

        # Create the task function wrapper
        _task_function = functools.partial(
            _core.elastic_launch_task_function, self.task_function, launch_config
        )

        # Run the job
        ret = run_job(
            hydra_context=hydra_context,
            task_function=_task_function,
            config=sweep_config,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
            configure_logging=False,
        )

        # Extract return value for rank 0
        logger.debug("Return value: %s", ret.return_value)
        if 0 in ret.return_value:
            ret.return_value = ret.return_value[0]

        return ret

    def checkpoint(self):
        """Support for submitit checkpointing."""
        import submitit

        return submitit.helpers.DelayedSubmission(self)
