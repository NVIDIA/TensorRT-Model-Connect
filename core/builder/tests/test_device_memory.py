# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's full-size memory pool switch against a fake CUDA driver."""

from __future__ import annotations

import pytest

from tensorrt_model_connect import device_memory

GIB = 1 << 30


class FakeDriver:
    """The driver calls the switch makes; pointers are the ctypes byref objects the module passes."""

    def __init__(self, *, integrated: int = 1, custom_pool: bool = False, fail: str | None = None) -> None:
        self.integrated, self.custom_pool, self.fail = integrated, custom_pool, fail
        self.created: list[int] = []
        self.current = 0xC0 if custom_pool else 0xD0
        self.destroyed: list[int] = []

    def _result(self, name: str) -> int:
        return 1 if self.fail == name else 0

    def cuInit(self, flags):
        return self._result("cuInit")

    def cuDeviceGet(self, device, ordinal):
        device._obj.value = ordinal
        return self._result("cuDeviceGet")

    def cuDeviceGetAttribute(self, value, attribute, device):
        assert attribute == 18
        value._obj.value = self.integrated
        return self._result("cuDeviceGetAttribute")

    def cuDeviceGetMemPool(self, pool, device):
        pool._obj.value = self.current
        return 0

    def cuDeviceGetDefaultMemPool(self, pool, device):
        pool._obj.value = 0xD0
        return 0

    def cuDeviceTotalMem_v2(self, total, device):
        total._obj.value = 122 * GIB
        return 0

    def cuMemPoolCreate(self, pool, props):
        self.created.append(props._obj.maxSize)
        pool._obj.value = 0xE0
        return self._result("cuMemPoolCreate")

    def cuDeviceSetMemPool(self, device, pool):
        if self.fail == "cuDeviceSetMemPool":
            return 1
        self.current = pool.value
        return 0

    def cuMemPoolDestroy(self, pool):
        self.destroyed.append(pool.value)
        return 0


@pytest.fixture
def driver(monkeypatch):
    def install(**options):
        fake = FakeDriver(**options)
        monkeypatch.setattr(device_memory, "_driver", lambda: fake)
        return fake

    return install


def test_an_integrated_gpu_on_its_default_pool_gets_a_pool_as_large_as_its_memory(driver):
    fake = driver()
    assert device_memory.use_full_size_memory_pool() is True
    assert fake.created == [122 * GIB] and fake.current == 0xE0


@pytest.mark.parametrize("options", [{"integrated": 0}, {"custom_pool": True}, {"fail": "cuInit"},
                                     {"fail": "cuDeviceGetAttribute"}])
def test_discrete_gpus_own_pools_and_driver_errors_are_left_alone(driver, options):
    fake = driver(**options)
    assert device_memory.use_full_size_memory_pool() is False
    assert fake.created == [] and fake.current == (0xC0 if options.get("custom_pool") else 0xD0)


def test_a_pool_the_driver_does_not_accept_is_destroyed(driver):
    fake = driver(fail="cuDeviceSetMemPool")
    assert device_memory.use_full_size_memory_pool() is False
    assert fake.destroyed == [0xE0] and fake.current == 0xD0


def test_no_cuda_driver_changes_nothing(monkeypatch):
    monkeypatch.setattr(device_memory, "_driver", lambda: None)
    assert device_memory.use_full_size_memory_pool() is False
