<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Pre-existing bug fixes — reviewer report

This branch fixes a set of **pre-existing runtime bugs** in Megatron Energon, all independent
of (but surfaced during) the static-typing work. Each was reproduced, then fixed **test-first**:
a regression test was written that **fails on `develop`** and **passes after the fix**, and the
full unittest suite passes after every change.

Each entry below gives the location, the behaviour **before**, the **fix**, the behaviour
**after**, and the regression test.

---

## 1. `metadataset` — relative subset start with the `"end"` keyword
**Where:** `src/megatron/energon/metadataset/metadataset_v2.py`, `Subset.as_dataset_subset`.

- **Before:** the `"end"` keyword converts to `None`; in the relative (percentage) branch this
  slipped past the assertion into the numeric comparison.
  `Subset(range=("50%", "end")).as_dataset_subset()` → `TypeError: '<=' not supported between
  instances of 'int' and 'NoneType'` (cryptic).
- **Fix:** require a float end in the relative branch with a clear message; `"end"` is only valid
  for absolute integer ranges. The absolute branch still supports `[100, "end"]`.
- **After:** `Subset(range=("50%", "end"))` raises a clear `AssertionError` ("the 'end' keyword
  is only allowed for absolute integer ranges"); valid relative/absolute ranges are unchanged.
- **Test:** `tests/test_metadataset_v2.py::test_subset_relative_start_end_keyword_fails`.

## 2. `cache` — `FileCacheLazy.get` returns a tuple on a cached re-get
**Where:** `src/megatron/energon/cache/file_cache_pool.py`, `FileCacheLazy.get`.

- **Before:** `_data` holds `(value, SourceInfo)`. The warm-path early return returned the whole
  tuple instead of the value, and skipped `add_source_info`. First `get(sample)` → `b"data"`
  (one source appended); a **second** `get(sample)` on the same lazy → `(b"data", SourceInfo(...))`
  (no source appended).
- **Fix:** populate `_data` only when `None`, then always `add_source_info` and return `_data[0]`
  (mirrors the correct sibling `DirectLazy.get`).
- **After:** every `get` returns the decoded value and appends source info; first-call behaviour
  is byte-for-byte unchanged.
- **Test:** `tests/test_file_cache_pool.py::test_get_twice_returns_value_and_source_info`.

## 3. `packing` — `PackingDataset.restore_sample` double-unpack
**Where:** `src/megatron/energon/wrappers/packing_dataset.py`, `PackingDataset.restore_sample`.

- **Before:** the restore key / inner index was unpacked twice in three places. In the
  `sample_encoder` branch the duplicate re-unpacked the already-stripped `inner_idx`, so
  `assert id == type(self).__name__` failed. Restoring a packed sample raised `AssertionError`
  whenever a `TaskEncoder` overrode `postencode_sample` (i.e. `sample_encoder` is set).
- **Fix:** remove the duplicate unpack statements so each unpack runs once.
- **After:** `loader.restore_sample(sample.__restore_key__)` round-trips (key and restore key
  match) with a postencode encoder.
- **Test:** `tests/test_dataset.py::test_packing_postencode_restore_sample`.

## 4. `buffer` — stale live buffer on restore
**Where:** `src/megatron/energon/wrappers/buffer.py`, `SavableSampleBuffer.restore_state`.

- **Before:** `restore_state` restored `_restore_keys` but left the live `_buffer`. Restoring onto
  a **reused** instance (the `num_workers=0` in-place restore path) then tripped `worker_start`'s
  `assert len(self._buffer) == 0`. Reproduced by saving mid-iteration and restoring onto the same
  loader → `AssertionError`.
- **Fix:** clear `_buffer` in `restore_state` so `worker_start` rebuilds it idempotently from
  `_restore_keys`.
- **After:** save mid-iteration → restore onto the same loader → continue iterating works.
- **Test:** `tests/test_dataset.py::test_restore_reused_loader_with_shuffle_buffer`.

## 5. `map` + `batch` — off-by-one sample index on generator resume
**Where:** `src/megatron/energon/wrappers/map_dataset.py` and
`src/megatron/energon/wrappers/batch_dataset.py` (generator-resume branch).

- **Before:** resuming a partially-consumed generator used `SampleIndex.current_idx` as the
  explicit sample index. `get_next()` post-increments, so while the generator for input index `N`
  is being consumed `current_idx == N+1`. Samples emitted **after a mid-generator checkpoint** got
  a sample index (and restore key) off by one — e.g. a sample that should carry `sample_index == 2`
  carried `3`.
- **Fix:** use `current_idx - 1` in the resume branch to recover `N`.
- **After:** resumed samples carry the same `(key, sample_index)` as an uninterrupted run.
- **Tests:** `tests/test_dataset.py::test_map_generator_resume_sample_index`,
  `tests/test_dataset.py::test_batch_generator_resume_sample_index`.

## 6. `group_batch` — stale/unbound bucket key on a handled grouping error
**Where:** `src/megatron/energon/wrappers/group_batch_dataset.py`, `__iter__`.

- **Before:** `bucket_key`/`batch_size` were read **outside** the `handle_errors` block that
  computes them. When `batch_group_criterion` raised a handled (logged-and-skipped) error,
  `bucket_key` was either unbound (`UnboundLocalError` — fatal) or stale (the failed sample was
  appended to the **previous** sample's bucket).
- **Fix:** track grouping success and `continue` (skip the sample) when grouping was
  handled-error'd, consistent with sample-skipping error handling.
- **After:** a sample whose grouping errors is skipped and iteration continues; no crash, no
  misrouting.
- **Test:** `tests/test_dataset.py::test_group_batch_grouping_error_skips_sample`.

## 7. `concat` / `blend` — single-dataset `restore_sample` crash
**Where:** `src/megatron/energon/wrappers/concat_dataset.py`,
`src/megatron/energon/wrappers/blend_dataset.py`.

- **Before:** both wrappers always prepend `(ClassName, ds_idx)` to the restore key in `__iter__`,
  even with a single wrapped dataset. `BaseWrapperDataset.restore_sample`'s `len == 1` fast-path
  forwarded the key **unstripped**, so a single-dataset `ConcatDataset`/`BlendDataset` crashed the
  inner dataset's fixed-arity parse: `ValueError: too many values to unpack`.
- **Fix:** override `restore_sample` in both wrappers to always strip the `(ClassName, ds_idx)`
  prefix. The multi-dataset case is unchanged (identical to the base else-branch).
- **After:** single-dataset `ConcatDataset`/`BlendDataset` `restore_sample` round-trips.
- **Test:** `tests/test_dataset.py::test_single_dataset_concat_blend_restore_sample`.

## 8. `savable_loader` — worker skip offset lost on re-save before emit
**Where:** `src/megatron/energon/savable_loader.py`, `SavableDataLoader.restore_state_rank`.

- **Before:** the main-process per-worker counters were initialized to `sample_index - 1`,
  ignoring the restore `offset`. A restored worker still has to skip `offset` samples before it
  emits, so **re-saving before that worker emitted** a post-restore sample collapsed its offset and
  re-emitted samples. Concretely: restore → consume `< num_workers` samples → re-save → restore →
  continue produced a sample stream that diverged from the canonical continuation for the
  not-yet-emitted worker.
- **Fix:** baseline the counter at `sample_index + offset - 1` (no change when `offset == 0`).
- **After:** re-saving before all workers have emitted is transparent — the continued stream
  matches the uninterrupted continuation.
- **Test:** `tests/test_dataset_det.py::test_restore_resave_before_all_workers_emit`.

---

## 9. `iter_map` — `IterMapDataset` multi-yield restore
**Where:** `src/megatron/energon/wrappers/iter_map_dataset.py`, `IterMapDataset.__iter__`.

- **Before:** the source-key window was cleared after **every** yielded output. For a one-to-many
  `iter_map_fn` (one fetched sample → several outputs — a supported shape per the docstring's
  non-aggregating contract), the 2nd+ output's restore key lost its source sample's key, so
  `restore_sample` fed an empty input to `iter_map_fn` and raised
  `RuntimeError: Generator did not yield enough samples`.
- **Fix:** reset the source-key window on each fetched inner sample (its single source) instead of
  after each yield, so every output of a one-to-many fn carries its source key. (Aggregating /
  many-to-one restore is explicitly out of contract per the docstring, and is unaffected.)
- **After:** `restore_sample` round-trips every output for one-to-one, one-to-many, and
  one-to-zero (skip) `iter_map_fn` shapes.
- **Test:** `tests/test_dataset.py::test_itermap_restore_sample_fanout_shapes`.
