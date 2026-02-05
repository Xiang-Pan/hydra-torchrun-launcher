# -*- coding: utf-8 -*-
# @Time    : 2024/3/28
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : _core.py
import logging
import pickle
import cloudpickle
import importlib

import functools
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, open_dict
from hydra.types import HydraContext
from hydra.core.singleton import Singleton
from hydra.core.hydra_config import HydraConfig
from hydra.types import TaskFunction
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals
)

from torch.distributed.launcher.api import launch_agent
from torch.distributed.run import config_from_args

from .distributed_launcher import TorchDistributedLauncher

logger = logging.getLogger(__name__)


def _get_function_reference(func):
    """Extract module and qualified name from a function for lazy import in workers.

    Returns a tuple of (module_name, qualname, script_dir) where script_dir may be None.
    """
    import sys
    import os

    module_name = getattr(func, '__module__', None)
    qualname = getattr(func, '__qualname__', None)
    script_dir = None

    if not module_name or not qualname:
        return None

    # Handle __main__ module - need to get the actual module name from the file
    if module_name == '__main__':
        # Get the main script path
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, '__file__') and main_module.__file__:
            script_path = os.path.abspath(main_module.__file__)
            script_dir = os.path.dirname(script_path)
            script_name = os.path.splitext(os.path.basename(script_path))[0]
            module_name = script_name

    return (module_name, qualname, script_dir)


def setup(
        launcher: TorchDistributedLauncher,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
) -> None:
    launcher.config = config
    launcher.hydra_context = hydra_context
    launcher.task_function = task_function

    launch_config = config_from_args(config.hydra.launcher)[0]
    launcher.launch_config = launch_config

    logger.info(f"\nTorchrunLauncher params:\n"
                f"\tinit_method: {launch_config.rdzv_endpoint}\n"
                f"\tnnodes: {launch_config.min_nodes}:{launch_config.max_nodes}\n"
                f"\tnproc_per_node: {launch_config.nproc_per_node}")


def launch(
        launcher: TorchDistributedLauncher,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
) -> Sequence[JobReturn]:
    """
    :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
    :param initial_job_idx: Initial job idx in batch.
    :return: an array of return values from run_job with indexes corresponding to the input list indexes.
    """
    setup_globals()
    assert launcher.config is not None
    assert launcher.hydra_context is not None
    assert launcher.task_function is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)
    sweep_dir = Path(str(launcher.config.hydra.sweep.dir))
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for idx, overrides in enumerate(job_overrides):
        idx = initial_job_idx + idx
        lst = " ".join(filter_overrides(overrides))
        logger.info(f"\t#{idx} : {lst}")
        sweep_config = launcher.hydra_context.config_loader.load_sweep_config(
            launcher.config, list(overrides)
        )
        with open_dict(sweep_config):
            # This typically coming from the underlying scheduler (SLURM_JOB_ID for instance)
            # In that case, it will not be available here because we are still in the main process.
            # but instead should be populated remotely before calling the task_function.
            sweep_config.hydra.job.id = f"job_id_for_{idx}"
            sweep_config.hydra.job.num = idx

        HydraConfig.instance().set_config(sweep_config)

        _task_function = functools.partial(
            elastic_launch_task_function, launcher.task_function, launcher.launch_config
        )

        ret = run_job(
            hydra_context=launcher.hydra_context,
            task_function=_task_function,
            config=sweep_config,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
            configure_logging=False  # Delay configure of logging
        )

        # We assume that main process has rank 0
        # Return value from launch_agent with type Dict[int, Any], where the key is **global rank**.
        logger.debug("Return value: %s", ret.return_value)
        if 0 in ret.return_value:
            ret.return_value = ret.return_value[0]
        runs.append(ret)
        configure_log(
            launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose
        )
    return runs


def elastic_launch_task_function(task_function, launch_config, task_config):
    # Try to get function reference for lazy import (avoids pickling torch._Ops)
    func_ref = _get_function_reference(task_function)

    if func_ref is not None:
        # Pass function reference instead of pickled function
        return launch_agent(
            launch_config,
            wrapped_task_function_by_ref,
            [func_ref, cloudpickle.dumps(task_config), cloudpickle.dumps(Singleton.get_state())],
        )
    else:
        # Fallback to original cloudpickle approach
        return launch_agent(
            launch_config,
            wrapped_task_function,
            [cloudpickle.dumps(task_function), cloudpickle.dumps(task_config), cloudpickle.dumps(Singleton.get_state())],
        )


def wrapped_task_function(dumped_task_function, dumped_task_config, dumped_singleton_state):
    # Restoring (legacy cloudpickle approach)
    task_function = pickle.loads(dumped_task_function)
    task_config = pickle.loads(dumped_task_config)
    singleton_state = pickle.loads(dumped_singleton_state)

    Singleton.set_state(singleton_state)
    config = HydraConfig.instance().cfg
    configure_log(config.hydra.job_logging, config.hydra.verbose)
    return task_function(task_config)


def wrapped_task_function_by_ref(func_ref, dumped_task_config, dumped_singleton_state):
    # Import function by reference (avoids pickling issues with torch._Ops)
    import sys

    module_name, qualname, script_dir = func_ref

    # Add script directory to path if provided (needed for __main__ module imports)
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    module = importlib.import_module(module_name)

    # Navigate to the function using qualname (handles nested classes/functions)
    task_function = module
    for attr in qualname.split('.'):
        task_function = getattr(task_function, attr)

    task_config = pickle.loads(dumped_task_config)
    singleton_state = pickle.loads(dumped_singleton_state)

    Singleton.set_state(singleton_state)
    config = HydraConfig.instance().cfg
    configure_log(config.hydra.job_logging, config.hydra.verbose)
    return task_function(task_config)
