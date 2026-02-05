# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : slurm_utils.py
"""SLURM detection utilities for auto-configuring torchrun in SLURM environments."""

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SlurmEnvironment:
    """Container for SLURM environment information."""

    is_slurm: bool
    job_id: Optional[str] = None
    num_nodes: Optional[int] = None
    node_rank: Optional[int] = None
    master_addr: Optional[str] = None
    master_port: int = 29500
    gpus_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    nodelist: Optional[str] = None
    local_id: Optional[int] = None
    ntasks_per_node: Optional[int] = None


def parse_slurm_nodelist(nodelist: str) -> List[str]:
    """Parse SLURM nodelist format into individual hostnames.

    Handles formats like:
    - node01
    - node[01-03]
    - node[01,03,05]
    - node[01-03,05,07-09]
    - gpu-node-[001-004]

    Args:
        nodelist: SLURM nodelist string (e.g., "node[01-03]" or "node01,node02")

    Returns:
        List of individual hostnames
    """
    # First try using scontrol if available (most reliable)
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback to regex parsing
    return _parse_nodelist_regex(nodelist)


def _parse_nodelist_regex(nodelist: str) -> List[str]:
    """Parse nodelist using regex when scontrol is not available."""
    hosts = []

    # Handle comma-separated top-level entries
    # This regex matches either a simple hostname or a hostname with bracket notation
    pattern = r"([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)"
    entries = re.findall(pattern, nodelist)

    for entry in entries:
        # Check if this entry has bracket notation
        bracket_match = re.match(r"([a-zA-Z0-9_-]+)\[([^\]]+)\]", entry)
        if bracket_match:
            prefix = bracket_match.group(1)
            range_spec = bracket_match.group(2)
            hosts.extend(_expand_range_spec(prefix, range_spec))
        else:
            # Simple hostname
            hosts.append(entry)

    return hosts


def _expand_range_spec(prefix: str, range_spec: str) -> List[str]:
    """Expand a range specification like '01-03,05,07-09' into hostnames."""
    hosts = []
    parts = range_spec.split(",")

    for part in parts:
        if "-" in part:
            # Range like 01-03
            start, end = part.split("-", 1)
            # Preserve leading zeros
            width = len(start)
            start_num = int(start)
            end_num = int(end)
            for i in range(start_num, end_num + 1):
                hosts.append(f"{prefix}{str(i).zfill(width)}")
        else:
            # Single number
            hosts.append(f"{prefix}{part}")

    return hosts


def detect_slurm_environment(
    override_master_port: Optional[int] = None,
) -> SlurmEnvironment:
    """Detect SLURM environment variables and return structured info.

    Args:
        override_master_port: Optional port to use instead of auto-detected/default

    Returns:
        SlurmEnvironment with detected values
    """
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")

    if not job_id:
        return SlurmEnvironment(is_slurm=False)

    # Parse number of nodes
    num_nodes = None
    nnodes_str = os.environ.get("SLURM_NNODES")
    if nnodes_str:
        try:
            num_nodes = int(nnodes_str)
        except ValueError:
            logger.warning(f"Could not parse SLURM_NNODES: {nnodes_str}")

    # Parse node rank (node ID within the job)
    node_rank = None
    nodeid_str = os.environ.get("SLURM_NODEID")
    if nodeid_str:
        try:
            node_rank = int(nodeid_str)
        except ValueError:
            logger.warning(f"Could not parse SLURM_NODEID: {nodeid_str}")

    # Parse nodelist and get master address
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    master_addr = None
    if nodelist:
        hosts = parse_slurm_nodelist(nodelist)
        if hosts:
            master_addr = hosts[0]

    # Determine master port
    master_port = 29500  # Default
    if override_master_port:
        master_port = override_master_port
    else:
        # Try to derive a port from the job ID to avoid collisions
        if job_id:
            try:
                # Use job ID to generate a port in range 29500-32767
                master_port = 29500 + (int(job_id) % 3268)
            except ValueError:
                pass

    # Parse GPUs per node
    gpus_per_node = None
    # Try various SLURM GPU environment variables
    gpus_str = (
        os.environ.get("SLURM_GPUS_PER_NODE")
        or os.environ.get("SLURM_GPUS_ON_NODE")
    )
    if gpus_str:
        # Handle format like "4" or "gpu:4"
        if ":" in gpus_str:
            gpus_str = gpus_str.split(":")[-1]
        try:
            gpus_per_node = int(gpus_str)
        except ValueError:
            logger.warning(f"Could not parse SLURM GPUs: {gpus_str}")

    # If SLURM_GPUS_PER_NODE is not set, try to count from CUDA_VISIBLE_DEVICES
    if gpus_per_node is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            gpus_per_node = len(cuda_visible.split(","))

    # Parse CPUs per task
    cpus_per_task = None
    cpus_str = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_str:
        try:
            cpus_per_task = int(cpus_str)
        except ValueError:
            pass

    # Parse local ID (rank on this node)
    local_id = None
    localid_str = os.environ.get("SLURM_LOCALID")
    if localid_str:
        try:
            local_id = int(localid_str)
        except ValueError:
            pass

    # Parse ntasks per node
    ntasks_per_node = None
    ntasks_str = os.environ.get("SLURM_NTASKS_PER_NODE")
    if ntasks_str:
        try:
            ntasks_per_node = int(ntasks_str)
        except ValueError:
            pass

    return SlurmEnvironment(
        is_slurm=True,
        job_id=job_id,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        nodelist=nodelist,
        local_id=local_id,
        ntasks_per_node=ntasks_per_node,
    )


def configure_torchrun_for_slurm(
    launcher_conf: dict,
    slurm_env: SlurmEnvironment,
    override_nnodes: Optional[str] = None,
    override_nproc_per_node: Optional[str] = None,
) -> dict:
    """Update torchrun launcher configuration with SLURM-detected values.

    Args:
        launcher_conf: The launcher configuration dict (from OmegaConf)
        slurm_env: Detected SLURM environment
        override_nnodes: Optional override for nnodes
        override_nproc_per_node: Optional override for nproc_per_node

    Returns:
        Updated configuration dict
    """
    if not slurm_env.is_slurm:
        return launcher_conf

    updated = dict(launcher_conf)

    # Set nnodes
    if override_nnodes:
        updated["nnodes"] = override_nnodes
    elif slurm_env.num_nodes:
        updated["nnodes"] = str(slurm_env.num_nodes)

    # Set nproc_per_node (GPUs per node)
    if override_nproc_per_node:
        updated["nproc_per_node"] = override_nproc_per_node
    elif slurm_env.gpus_per_node:
        updated["nproc_per_node"] = str(slurm_env.gpus_per_node)

    # Set master address and port
    if slurm_env.master_addr:
        updated["master_addr"] = slurm_env.master_addr
    updated["master_port"] = slurm_env.master_port

    # Set node rank
    if slurm_env.node_rank is not None:
        updated["node_rank"] = slurm_env.node_rank

    # For multi-node, use c10d rendezvous backend
    if slurm_env.num_nodes and slurm_env.num_nodes > 1:
        updated["rdzv_backend"] = "c10d"
        if slurm_env.master_addr:
            updated["rdzv_endpoint"] = (
                f"{slurm_env.master_addr}:{slurm_env.master_port}"
            )

    logger.info(
        f"Configured torchrun for SLURM:\n"
        f"\tnnodes: {updated.get('nnodes')}\n"
        f"\tnproc_per_node: {updated.get('nproc_per_node')}\n"
        f"\tmaster_addr: {updated.get('master_addr')}\n"
        f"\tmaster_port: {updated.get('master_port')}\n"
        f"\tnode_rank: {updated.get('node_rank')}"
    )

    return updated
