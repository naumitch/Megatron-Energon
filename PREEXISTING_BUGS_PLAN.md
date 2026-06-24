<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Pre-existing bug fixes — plan

Three genuine, pre-existing runtime bugs were surfaced while adding static type checking
(they are independent of typing — they exist on `develop`). This branch
(`nicholas/fix-pre-existing-bugs`, off `develop`) fixes them **test-first**: each fix ships
with a regression test that **fails on current `develop`** and **passes after the fix**.

Each bug was independently investigated and then adversarially verified (and reproduced)
before being included here.

> Note: this planning file is a working artifact. It can be dropped or moved into the PR
> description before opening a pull request.

## Bug 1 — `FileCacheLazy.get` returns a tuple (and skips source info) on a cached re-get
**Severity: medium.** `src/megatron/energon/cache/file_cache_pool.py`, `FileCacheLazy.get`.

`_data` holds `Optional[tuple[T, SourceInfo]]`. The warm-path early return
`if self._data is not None: return self._data` returns the **whole tuple** instead of
`self._data[0]`, and skips `add_source_info(sample, self._data[1])`. So the first `get()`
returns `T` and adds source info; a second `get()` on the same lazy returns
`(value, SourceInfo)` and adds nothing. The sibling `DirectLazy.get`
(`no_cache.py`) is the correct pattern.

- **Fix:** populate `_data` only when `None`, then always add source info and return `_data[0]`:
  ```python
  if self._data is None:
      self._data = self.pool._get_data(self.ds, self.fname, self.entry)
  assert self._data is not None
  add_source_info(sample, self._data[1])
  return self._data[0]
  ```
- **Test:** `tests/test_file_cache_pool.py::TestFileStoreCachePool::test_get_twice_returns_value_and_source_info` — call `get(sample)` twice on one `FileCacheLazy`; assert both return the decoded value (not a tuple) and append source info. Fails-before (2nd call returns a tuple, 0 sources); passes-after.

## Bug 2 — `PackingDataset.restore_sample` re-unpacks `inner_idx`, corrupting restore (encoder path)
**Severity: high.** `src/megatron/energon/wrappers/packing_dataset.py`, `PackingDataset.restore_sample`.

Three duplicated unpack statements. Two (the generator / non-generator `restore_key`
re-unpacks) are harmless dead duplicates. The third is the real bug: in the
`sample_encoder is not None` branch, `inner_idx` is correctly stripped of
`("PackingDataset", encode_idx)`, then the **duplicate** line re-unpacks the
already-stripped `inner_idx` (now headed by the inner dataset name), so
`assert id == type(self).__name__` raises `AssertionError`. Manifests only when a
`TaskEncoder` overrides `postencode_sample` (so `sample_encoder` is non-`None`) and
`restore_sample` is called on a packed sample.

- **Fix:** delete the three duplicate unpack lines so each unpack runs once.
- **Test:** `tests/test_dataset.py::TestDataset::test_packing_postencode_restore_sample` — mirror
  the existing `test_packing` fixture but make the encoder override `postencode_sample`
  (`@stateless`), then `restore_sample(samples[1].__restore_key__)` and assert key /
  restore_key match. Fails-before (`AssertionError`); passes-after.

## Bug 3 — `Subset.as_dataset_subset`: relative start + `end="end"` raises a cryptic `TypeError`
**Severity: low.** `src/megatron/energon/metadataset/metadataset_v2.py`, `Subset.as_dataset_subset`.

`_conv("end") -> None`. In the relative (float) branch the assertion
`isinstance(end, float) or end is None` permits `None`, then `0 <= end <= 1` raises
`TypeError: '<=' not supported between 'int' and 'NoneType'`. The `"end"` keyword is only
valid for absolute ranges (per the docstring), so `["50%", "end"]` is invalid input that
should fail clearly.

- **Fix:** tighten the relative-branch end assertion to require a float, with a clear message
  (the absolute branch keeps supporting `[100, end]`):
  ```python
  assert isinstance(end, float), (
      "End must be a relative percentage (e.g. '75%') if start is relative; "
      "the 'end' keyword is only allowed for absolute integer ranges"
  )
  ```
- **Test:** `tests/test_metadataset_v2.py::...::test_subset_relative_start_end_keyword_fails` —
  `assertRaises(AssertionError)` for `Subset(range=("50%", "end")).as_dataset_subset()`.
  Fails-before (raises `TypeError`, not caught); passes-after.

## Note on scope
These are behavioral fixes, kept entirely separate from the type-checking branch. Bug 1 and
Bug 3 also correspond to mypy errors deliberately left in that branch's baseline; once these
land on `develop`, those baseline entries can be removed there.
