# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : Yaojie Shen
# @Project : hydra-torchrun-launcher
# @File    : test_slurm.py
"""Tests for SLURM detection and launcher functionality."""

import os
from unittest import mock

import pytest

from hydra_plugins.hydra_torchrun_launcher.slurm_utils import (
    SlurmEnvironment,
    detect_slurm_environment,
    parse_slurm_nodelist,
    _parse_nodelist_regex,
    configure_torchrun_for_slurm,
)


class TestParseNodelist:
    """Tests for SLURM nodelist parsing."""

    def test_single_node(self):
        """Test parsing a single node."""
        result = _parse_nodelist_regex("node01")
        assert result == ["node01"]

    def test_simple_range(self):
        """Test parsing a simple range like node[01-03]."""
        result = _parse_nodelist_regex("node[01-03]")
        assert result == ["node01", "node02", "node03"]

    def test_range_with_larger_numbers(self):
        """Test parsing ranges with larger numbers."""
        result = _parse_nodelist_regex("gpu[001-004]")
        assert result == ["gpu001", "gpu002", "gpu003", "gpu004"]

    def test_comma_separated_in_brackets(self):
        """Test parsing comma-separated values in brackets."""
        result = _parse_nodelist_regex("node[01,03,05]")
        assert result == ["node01", "node03", "node05"]

    def test_mixed_range_and_singles(self):
        """Test parsing mixed ranges and single values."""
        result = _parse_nodelist_regex("node[01-03,05,07-09]")
        assert result == [
            "node01",
            "node02",
            "node03",
            "node05",
            "node07",
            "node08",
            "node09",
        ]

    def test_preserves_leading_zeros(self):
        """Test that leading zeros are preserved."""
        result = _parse_nodelist_regex("node[001-003]")
        assert result == ["node001", "node002", "node003"]

    def test_hostname_with_dashes(self):
        """Test parsing hostnames that contain dashes."""
        result = _parse_nodelist_regex("gpu-node-[01-02]")
        assert result == ["gpu-node-01", "gpu-node-02"]

    def test_parse_slurm_nodelist_fallback(self):
        """Test that parse_slurm_nodelist falls back to regex when scontrol unavailable."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = parse_slurm_nodelist("node[01-02]")
            assert result == ["node01", "node02"]


class TestDetectSlurmEnvironment:
    """Tests for SLURM environment detection."""

    def test_not_in_slurm(self):
        """Test detection when not in SLURM environment."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env = detect_slurm_environment()
            assert env.is_slurm is False
            assert env.job_id is None

    def test_basic_slurm_detection(self):
        """Test basic SLURM environment detection."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_NNODES": "2",
            "SLURM_NODEID": "0",
            "SLURM_JOB_NODELIST": "node01",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            with mock.patch(
                "hydra_plugins.hydra_torchrun_launcher.slurm_utils.parse_slurm_nodelist",
                return_value=["node01", "node02"],
            ):
                env = detect_slurm_environment()

        assert env.is_slurm is True
        assert env.job_id == "12345"
        assert env.num_nodes == 2
        assert env.node_rank == 0
        assert env.master_addr == "node01"

    def test_gpu_detection(self):
        """Test GPU per node detection."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_GPUS_PER_NODE": "4",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment()

        assert env.gpus_per_node == 4

    def test_gpu_detection_with_type_prefix(self):
        """Test GPU detection when format is 'gpu:4'."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_GPUS_PER_NODE": "gpu:4",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment()

        assert env.gpus_per_node == 4

    def test_gpu_detection_from_cuda_visible(self):
        """Test GPU detection from CUDA_VISIBLE_DEVICES fallback."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment()

        assert env.gpus_per_node == 4

    def test_master_port_from_job_id(self):
        """Test master port derivation from job ID."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment()

        # Port should be in range 29500-32767
        assert 29500 <= env.master_port <= 32767

    def test_master_port_override(self):
        """Test master port override."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment(override_master_port=30000)

        assert env.master_port == 30000

    def test_slurm_jobid_alternative(self):
        """Test detection with SLURM_JOBID (alternative env var)."""
        slurm_env = {
            "SLURM_JOBID": "99999",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment()

        assert env.is_slurm is True
        assert env.job_id == "99999"

    def test_cpus_per_task(self):
        """Test CPUs per task detection."""
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_CPUS_PER_TASK": "8",
        }
        with mock.patch.dict(os.environ, slurm_env, clear=True):
            env = detect_slurm_environment()

        assert env.cpus_per_task == 8


class TestConfigureTorchrunForSlurm:
    """Tests for configure_torchrun_for_slurm function."""

    def test_not_slurm_returns_unchanged(self):
        """Test that config is unchanged when not in SLURM."""
        conf = {"nnodes": "1", "nproc_per_node": "1"}
        slurm_env = SlurmEnvironment(is_slurm=False)

        result = configure_torchrun_for_slurm(conf, slurm_env)

        assert result == conf

    def test_basic_configuration(self):
        """Test basic SLURM configuration."""
        conf = {"nnodes": "1", "nproc_per_node": "1", "master_addr": "localhost"}
        slurm_env = SlurmEnvironment(
            is_slurm=True,
            job_id="12345",
            num_nodes=2,
            node_rank=0,
            master_addr="node01",
            master_port=29500,
            gpus_per_node=4,
        )

        result = configure_torchrun_for_slurm(conf, slurm_env)

        assert result["nnodes"] == "2"
        assert result["nproc_per_node"] == "4"
        assert result["master_addr"] == "node01"
        assert result["master_port"] == 29500
        assert result["node_rank"] == 0

    def test_multi_node_sets_rdzv_backend(self):
        """Test that multi-node sets c10d rendezvous backend."""
        conf = {"nnodes": "1", "rdzv_backend": "static"}
        slurm_env = SlurmEnvironment(
            is_slurm=True,
            num_nodes=2,
            master_addr="node01",
            master_port=29500,
        )

        result = configure_torchrun_for_slurm(conf, slurm_env)

        assert result["rdzv_backend"] == "c10d"
        assert result["rdzv_endpoint"] == "node01:29500"

    def test_override_nnodes(self):
        """Test nnodes override."""
        conf = {"nnodes": "1"}
        slurm_env = SlurmEnvironment(
            is_slurm=True,
            num_nodes=4,
        )

        result = configure_torchrun_for_slurm(
            conf, slurm_env, override_nnodes="2:4"
        )

        assert result["nnodes"] == "2:4"

    def test_override_nproc_per_node(self):
        """Test nproc_per_node override."""
        conf = {"nproc_per_node": "1"}
        slurm_env = SlurmEnvironment(
            is_slurm=True,
            gpus_per_node=8,
        )

        result = configure_torchrun_for_slurm(
            conf, slurm_env, override_nproc_per_node="4"
        )

        assert result["nproc_per_node"] == "4"


class TestTorchrunSlurmLauncherDiscovery:
    """Tests for TorchrunSlurmLauncher plugin discovery."""

    def test_discovery(self):
        """Test that the SLURM launcher can be discovered."""
        from hydra.core.plugins import Plugins
        from hydra.plugins.launcher import Launcher
        from hydra_plugins.hydra_torchrun_launcher.slurm_launcher import (
            TorchrunSlurmLauncher,
        )

        assert TorchrunSlurmLauncher.__name__ in [
            x.__name__ for x in Plugins.instance().discover(Launcher)
        ]

    def test_config_registered(self):
        """Test that torchrun_slurm config is registered."""
        from hydra_plugins.hydra_torchrun_launcher.slurm_config import (
            TorchrunSlurmLauncherConf,
        )

        # Verify the config class has the correct target
        assert TorchrunSlurmLauncherConf._target_ == (
            "hydra_plugins.hydra_torchrun_launcher.slurm_launcher.TorchrunSlurmLauncher"
        )
        # Verify default mode
        conf = TorchrunSlurmLauncherConf()
        assert conf.mode == "auto"
