"""Tests for debug_runner.py cache state machine logic.

Extracts and tests the pure-numpy cache/mask/position logic from TrtRunner
without needing TRT or a real engine. Validates that the state machine
matches the family-local debug runner cache behavior.

Trace: ARCH-DBG-001, UD-DBG-01
Intent: Validate that the Python debug runner's cache position, attention mask, and KV append/shift logic matches the family-local debug runner cache contract.
Preconditions: No TRT or GPU required; pure-numpy replica of cache state machine.
Postconditions: Position IDs, attention masks, and cache contents are correct through prefill, autoregressive, and cache-full overflow phases.
"""

from __future__ import annotations

import numpy as np
import pytest


class CacheStateMachine:
    """Pure-numpy replica of TrtRunner's cache state tracking.

    This mirrors the position_id, attention_mask, and cache append/shift
    logic exactly as implemented in debug_runner.py TrtRunner.step().
    """

    def __init__(self, max_cache_length: int, attention_size: int = 4,
                 num_layers: int = 1):
        self.max_cache_length = max_cache_length
        self.attention_size = attention_size
        self.num_layers = num_layers
        self.cache_length = 0
        self.cache_k = [
            np.zeros((max_cache_length, attention_size), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self.cache_v = [
            np.zeros((max_cache_length, attention_size), dtype=np.float32)
            for _ in range(num_layers)
        ]

    def step(self, present_k: np.ndarray, present_v: np.ndarray):
        """Simulate one step: compute position, mask, update cache."""
        attention_window = self.max_cache_length + 1
        position_id = min(self.cache_length, self.max_cache_length)

        # Build attention mask
        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(self.cache_length, self.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        # Update cache
        for i in range(self.num_layers):
            pk = present_k[i].flatten()
            pv = present_v[i].flatten()
            if self.cache_length < self.max_cache_length:
                self.cache_k[i][self.cache_length] = pk
                self.cache_v[i][self.cache_length] = pv
            else:
                self.cache_k[i][:-1] = self.cache_k[i][1:]
                self.cache_k[i][-1] = pk
                self.cache_v[i][:-1] = self.cache_v[i][1:]
                self.cache_v[i][-1] = pv

        self.cache_length = min(self.cache_length + 1, self.max_cache_length)
        return position_id, mask


class TestPositionIdSequence:
    def test_basic_sequence(self):
        sm = CacheStateMachine(max_cache_length=4)
        positions = []
        for step in range(8):
            pk = [np.ones((1, 4), dtype=np.float32) * step]
            pv = [np.ones((1, 4), dtype=np.float32) * step]
            pos, _ = sm.step(pk, pv)
            positions.append(pos)

        # Position increments up to max_cache_length, then stays there
        assert positions == [0, 1, 2, 3, 4, 4, 4, 4]

    def test_position_clamp(self):
        sm = CacheStateMachine(max_cache_length=2)
        positions = []
        for step in range(6):
            pk = [np.zeros((1, 4), dtype=np.float32)]
            pv = [np.zeros((1, 4), dtype=np.float32)]
            pos, _ = sm.step(pk, pv)
            positions.append(pos)

        assert positions == [0, 1, 2, 2, 2, 2]


class TestAttentionMask:
    def test_mask_sequence(self):
        """Verify mask pattern over steps with max_cache=3."""
        sm = CacheStateMachine(max_cache_length=3)
        masks = []
        for step in range(6):
            pk = [np.zeros((1, 4), dtype=np.float32)]
            pv = [np.zeros((1, 4), dtype=np.float32)]
            _, mask = sm.step(pk, pv)
            masks.append(mask[0].tolist())

        # attention_window = 3 + 1 = 4
        # Step 0: cache_length=0 before step => valid=0
        #   mask = [-1e9, -1e9, -1e9, 0.0]  (only current token)
        assert masks[0] == [-1e9, -1e9, -1e9, 0.0]

        # Step 1: cache_length=1 => valid=1
        #   mask = [0.0, -1e9, -1e9, 0.0]
        assert masks[1] == [0.0, -1e9, -1e9, 0.0]

        # Step 2: cache_length=2 => valid=2
        #   mask = [0.0, 0.0, -1e9, 0.0]
        assert masks[2] == [0.0, 0.0, -1e9, 0.0]

        # Step 3: cache_length=3 (clamped) => valid=3
        #   mask = [0.0, 0.0, 0.0, 0.0]
        assert masks[3] == [0.0, 0.0, 0.0, 0.0]

        # Steps 4-5: cache_length stays at 3, mask stays fully open
        assert masks[4] == [0.0, 0.0, 0.0, 0.0]
        assert masks[5] == [0.0, 0.0, 0.0, 0.0]


class TestCacheAppend:
    def test_append_within_capacity(self):
        sm = CacheStateMachine(max_cache_length=4, attention_size=2)

        for step in range(3):
            pk = [np.array([[step * 10 + 1, step * 10 + 2]], dtype=np.float32)]
            pv = [np.array([[step * 100 + 1, step * 100 + 2]], dtype=np.float32)]
            sm.step(pk, pv)

        # After 3 steps, cache has 3 entries
        assert sm.cache_length == 3
        np.testing.assert_array_equal(sm.cache_k[0][0], [1, 2])
        np.testing.assert_array_equal(sm.cache_k[0][1], [11, 12])
        np.testing.assert_array_equal(sm.cache_k[0][2], [21, 22])
        # Slot 3 should still be zeros
        np.testing.assert_array_equal(sm.cache_k[0][3], [0, 0])


class TestCacheShiftLeft:
    def test_shift_when_full(self):
        sm = CacheStateMachine(max_cache_length=3, attention_size=1)

        # Fill cache
        for step in range(3):
            pk = [np.array([[float(step)]], dtype=np.float32)]
            pv = [np.array([[float(step)]], dtype=np.float32)]
            sm.step(pk, pv)

        assert sm.cache_length == 3
        np.testing.assert_array_equal(
            sm.cache_k[0].flatten(), [0, 1, 2])

        # Next step should shift left and append
        pk = [np.array([[3.0]], dtype=np.float32)]
        pv = [np.array([[3.0]], dtype=np.float32)]
        sm.step(pk, pv)

        assert sm.cache_length == 3  # stays clamped
        np.testing.assert_array_equal(
            sm.cache_k[0].flatten(), [1, 2, 3])

        # Another shift
        pk = [np.array([[4.0]], dtype=np.float32)]
        pv = [np.array([[4.0]], dtype=np.float32)]
        sm.step(pk, pv)
        np.testing.assert_array_equal(
            sm.cache_k[0].flatten(), [2, 3, 4])


class TestCacheLengthClamping:
    def test_clamp_at_max(self):
        sm = CacheStateMachine(max_cache_length=2)

        for step in range(10):
            pk = [np.zeros((1, 4), dtype=np.float32)]
            pv = [np.zeros((1, 4), dtype=np.float32)]
            sm.step(pk, pv)

        assert sm.cache_length == 2


class TestMultiLayer:
    def test_multi_layer_cache(self):
        sm = CacheStateMachine(
            max_cache_length=4, attention_size=2, num_layers=3)

        pk = [
            np.array([[1, 2]], dtype=np.float32),
            np.array([[3, 4]], dtype=np.float32),
            np.array([[5, 6]], dtype=np.float32),
        ]
        pv = [
            np.array([[10, 20]], dtype=np.float32),
            np.array([[30, 40]], dtype=np.float32),
            np.array([[50, 60]], dtype=np.float32),
        ]
        sm.step(pk, pv)

        np.testing.assert_array_equal(sm.cache_k[0][0], [1, 2])
        np.testing.assert_array_equal(sm.cache_k[1][0], [3, 4])
        np.testing.assert_array_equal(sm.cache_k[2][0], [5, 6])
        np.testing.assert_array_equal(sm.cache_v[0][0], [10, 20])
        np.testing.assert_array_equal(sm.cache_v[1][0], [30, 40])
        np.testing.assert_array_equal(sm.cache_v[2][0], [50, 60])


class TestCacheEdgeCases:
    """Edge-case tests for CacheStateMachine with extreme max_cache_length values."""

    def test_max_cache_length_zero_construction(self):
        """max_cache_length=0 creates zero-length cache arrays."""
        sm = CacheStateMachine(max_cache_length=0, attention_size=4)
        assert sm.max_cache_length == 0
        assert sm.cache_length == 0
        assert sm.cache_k[0].shape == (0, 4)
        assert sm.cache_v[0].shape == (0, 4)

    def test_max_cache_length_zero_step_raises(self):
        """max_cache_length=0: step raises IndexError on cache update.

        With cache_length=0 and max_cache_length=0, the condition
        cache_length < max_cache_length is False, so it enters the
        shift-left branch and tries to index [-1] on the zero-length
        cache array. This documents the current behavior — callers must
        use max_cache_length >= 1.
        """
        sm = CacheStateMachine(max_cache_length=0, attention_size=4)

        pk = [np.ones((1, 4), dtype=np.float32)]
        pv = [np.ones((1, 4), dtype=np.float32)]
        with pytest.raises(IndexError):
            sm.step(pk, pv)

    def test_max_cache_length_one_construction(self):
        """max_cache_length=1 creates a single-slot cache."""
        sm = CacheStateMachine(max_cache_length=1, attention_size=2)
        assert sm.cache_k[0].shape == (1, 2)
        assert sm.cache_v[0].shape == (1, 2)

    def test_max_cache_length_one_first_step(self):
        """max_cache_length=1: first step appends to empty cache, position=0."""
        sm = CacheStateMachine(max_cache_length=1, attention_size=2)

        pk = [np.array([[10.0, 20.0]], dtype=np.float32)]
        pv = [np.array([[30.0, 40.0]], dtype=np.float32)]
        pos, mask = sm.step(pk, pv)

        assert pos == 0
        # attention_window = 1 + 1 = 2
        # valid = min(0, 1) = 0 before step, so mask = [-1e9, 0.0]
        assert mask.shape == (1, 2)
        assert mask[0, 0] == -1e9
        assert mask[0, 1] == 0.0

        # Cache should contain the appended value
        assert sm.cache_length == 1
        np.testing.assert_array_equal(sm.cache_k[0][0], [10.0, 20.0])
        np.testing.assert_array_equal(sm.cache_v[0][0], [30.0, 40.0])

    def test_max_cache_length_one_second_step_shifts(self):
        """max_cache_length=1: second step shifts out the first entry."""
        sm = CacheStateMachine(max_cache_length=1, attention_size=2)

        # First step: append
        pk = [np.array([[1.0, 2.0]], dtype=np.float32)]
        pv = [np.array([[3.0, 4.0]], dtype=np.float32)]
        sm.step(pk, pv)

        # Second step: should shift-left and replace
        pk = [np.array([[5.0, 6.0]], dtype=np.float32)]
        pv = [np.array([[7.0, 8.0]], dtype=np.float32)]
        pos, mask = sm.step(pk, pv)

        assert pos == 1  # clamped to max_cache_length
        # valid = min(1, 1) = 1, so both slots are open: [0.0, 0.0]
        assert mask.shape == (1, 2)
        assert mask[0, 0] == 0.0
        assert mask[0, 1] == 0.0

        # Cache should have the second value only
        assert sm.cache_length == 1
        np.testing.assert_array_equal(sm.cache_k[0][0], [5.0, 6.0])
        np.testing.assert_array_equal(sm.cache_v[0][0], [7.0, 8.0])

    def test_max_cache_length_one_position_clamp(self):
        """max_cache_length=1: position clamps at 1 after two steps."""
        sm = CacheStateMachine(max_cache_length=1, attention_size=2)
        positions = []
        for step in range(5):
            pk = [np.array([[float(step), float(step)]], dtype=np.float32)]
            pv = [np.array([[float(step), float(step)]], dtype=np.float32)]
            pos, _ = sm.step(pk, pv)
            positions.append(pos)

        assert positions == [0, 1, 1, 1, 1]
