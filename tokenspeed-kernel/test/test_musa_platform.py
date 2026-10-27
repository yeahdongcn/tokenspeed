# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Moore Threads MUSA platform detection and capability gating.

These run on any host: detection is exercised against a stubbed ``torch.musa``,
and the capability checks build the platform value directly.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import tokenspeed_kernel.platform as platform_module
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    PlatformInfo,
)


def _musa_namespace(major: int = 3, minor: int = 1) -> SimpleNamespace:
    props = SimpleNamespace(
        major=major,
        minor=minor,
        name="MTT S5000",
        total_memory=48 * 1024**3,
        multi_processor_count=104,
        max_threads_per_multi_processor=2048,
        max_shared_memory_per_block=64 * 1024,
        memory_clock_rate=0,
        memory_bus_width=0,
    )
    return SimpleNamespace(
        get_device_properties=lambda device: props,
        current_device=lambda: 0,
        device_count=lambda: 1,
        is_available=lambda: True,
    )


@pytest.fixture
def detect_musa():
    def detect(major=3, minor=1):
        musa = _musa_namespace(major, minor)
        with patch.object(platform_module.torch, "musa", musa, create=True):
            return platform_module._detect_musa_platform()

    return detect


def test_s5000_detection(detect_musa):
    platform = detect_musa()
    assert platform.is_musa
    assert platform.vendor == "musa"
    assert platform.arch_version == ArchVersion(3, 1)
    assert platform.device_name == "MTT S5000"
    assert platform.device_count == 1
    assert platform.sm_count == 104
    assert platform.max_shared_memory_per_sm == 64 * 1024
    assert platform.interconnect.topology == "single_gpu"
    assert not platform.is_nvidia
    assert not platform.is_amd
    assert not platform.is_hopper_plus
    assert not platform.is_blackwell_plus
    assert platform.generation_name == "musa:3.1"


def test_musa_runs_the_portable_kernel_dialects(detect_musa):
    platform = detect_musa()
    assert platform.kernel_solutions == frozenset({"triton", "torch", "composite"})


def test_musa_is_detected_ahead_of_a_forced_cuda_available(detect_musa):
    """A project that forces ``torch.cuda.is_available()`` must not be read as CUDA."""
    musa = _musa_namespace()
    with (
        patch.object(platform_module.torch, "musa", musa, create=True),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert platform_module._detect_platform().vendor == "musa"


def test_a_host_with_no_device_raises():
    with (
        patch.object(platform_module, "_musa_is_available", return_value=False),
        patch.object(platform_module, "_npu_is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="MUSA"):
            platform_module._detect_platform()


def test_cuda_registration_needs_its_solution_to_run_on_musa(musa_platform):
    """Admission policy: an executable dialect crosses the vendor gate.

    This is a statement about the kernel source. The launcher is not covered:
    ``test_occupancy_on_an_unsupported_vendor_names_the_gap`` pins what happens
    when one needs a driver service the platform does not implement.
    """
    cuda_only = CapabilityRequirement(vendors=frozenset({"nvidia"}))
    assert not cuda_only.satisfied_by(musa_platform)
    assert cuda_only.satisfied_by(musa_platform, solution="triton")
    # A vendor compiler has no MUSA equivalent even with no arch claim.
    assert not cuda_only.satisfied_by(musa_platform, solution="cute_dsl")
    assert not cuda_only.satisfied_by(musa_platform, solution="flashinfer")


def test_arch_gated_registration_stays_vendor_gated(musa_platform):
    """An arch floor is stated on the original vendor's scale, so it is not crossable."""
    cuda_hopper = CapabilityRequirement(
        vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
    )
    assert not cuda_hopper.satisfied_by(musa_platform, solution="triton")

    per_vendor = CapabilityRequirement(
        vendors=frozenset({"nvidia", "amd"}),
        vendor_min_arch_versions={
            "nvidia": ArchVersion(10, 0),
            "amd": ArchVersion(12, 5),
        },
    )
    assert not per_vendor.satisfied_by(musa_platform, solution="triton")

    bounded = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
    )
    assert not bounded.satisfied_by(musa_platform, solution="triton")


def test_musa_registration_matches_without_a_solution(musa_platform):
    assert CapabilityRequirement(vendors=frozenset({"musa"})).satisfied_by(
        musa_platform
    )
    # A portable Triton registration written for AMD is the same source as one
    # written for NVIDIA, so it crosses the vendor gate the same way.
    assert CapabilityRequirement(vendors=frozenset({"amd"})).satisfied_by(
        musa_platform, solution="triton"
    )
    # Vendor-compiler dialects do not, whichever vendor they were written for.
    for dialect in ("gluon", "cute_dsl", "cuda"):
        assert not CapabilityRequirement(vendors=frozenset({"amd"})).satisfied_by(
            musa_platform, solution=dialect
        )


def test_musa_features_are_conservative(detect_musa):
    platform = detect_musa()
    assert platform.sm_features == frozenset({"tensor_core:f16"})
    for feature in (
        "tensor_core:f8",
        "tensor_core:f4",
        "memory:tma",
        "memory:async_copy",
    ):
        assert not CapabilityRequirement(
            required_features=frozenset({feature})
        ).satisfied_by(platform, solution="triton")


def test_other_platforms_are_unaffected(musa_platform, h100_platform):
    """The cross-vendor rule is opt-in per platform."""
    assert h100_platform.kernel_solutions is None
    assert CapabilityRequirement(vendors=frozenset({"nvidia"})).satisfied_by(
        h100_platform
    )
    # An AMD registration still does not run on an NVIDIA platform.
    assert not CapabilityRequirement(vendors=frozenset({"amd"})).satisfied_by(
        h100_platform, solution="triton"
    )
    assert not CapabilityRequirement(vendors=frozenset({"nvidia"})).satisfied_by(
        musa_platform
    )


def test_platform_info_defaults_keep_existing_vendors_gated():
    """A platform that declares no dialect is exactly vendor-gated."""
    platform = PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(9, 5),
        device_name="AMD Instinct MI350X",
        device_count=8,
        total_memory=288 * 1024**3,
        memory_bandwidth=8000.0,
        sm_count=384,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=65536,
    )
    assert platform.kernel_solutions is None
    assert not CapabilityRequirement(vendors=frozenset({"nvidia"})).satisfied_by(
        platform, solution="triton"
    )


def test_a_dialect_the_platform_cannot_run_is_rejected(musa_platform):
    """A registration with no capability at all is still held to the dialects."""
    unclaimed = CapabilityRequirement()
    assert unclaimed.satisfied_by(musa_platform, solution="triton")
    assert not unclaimed.satisfied_by(musa_platform, solution="cuda")
    assert not unclaimed.satisfied_by(musa_platform, solution="gluon")
    # An unnamed solution cannot be judged by dialect and stays selectable.
    assert unclaimed.satisfied_by(musa_platform)


class _FakeCompiledKernel:
    """Stand-in for a compiled kernel on a backend this host cannot query.

    A real handle needs a device of that vendor, which no CI host has for every
    vendor; the branch under test reads only these fields.
    """

    def __init__(self, backend: str) -> None:
        self.metadata = _FakeMetadata(backend)

    def _init_handles(self) -> None:
        pass


class _FakeMetadata:
    def __init__(self, backend: str) -> None:
        self.target = _FakeTarget(backend)
        self.num_ctas = 1


class _FakeTarget:
    def __init__(self, backend: str) -> None:
        self.backend = backend


def test_occupancy_on_an_unsupported_vendor_names_the_gap():
    """A launcher that needs residency must fail loudly, not guess a grid.

    Portable-Triton admission is a claim about kernel source; a launcher that
    queries the driver for a service this vendor does not implement has to say
    so. ``max_active_ctas`` is the one such service a registered kernel uses
    today (the QSA score grid).
    """
    with pytest.raises(ValueError, match="occupancy is not implemented"):
        platform_module.max_active_ctas(_FakeCompiledKernel("musa"))
