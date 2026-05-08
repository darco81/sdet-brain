# Sprint Report: T1-08 File watcher with debouncing + delete handling

> Sprint report autorski. Linear refs (`SDE-XX`) są internal trackingiem
> i nie są publicznie linkowane.


**Linear:** [SDE-25](https://linear.app/sdet-it/issue/SDE-25/t1-08-file-watcher-with-debouncing-delete-handling)
**Started:** 2026-04-30 16:34 (CET)
**Done:** 2026-04-30 16:46 (CET)
**CC time:** ~12 min
**Dariusz manual:** 0 min

## What shipped

- `src/sdet_brain/ingestion/watcher.py`: `BrainWatcher`
  (`FileSystemEventHandler` subclass) with thread-safe debounced
  re-ingest queue, dedicated worker thread, graceful drain on stop,
  context-manager interface, plus the public helpers
  `is_relevant_path` and the `WatcherStats` counter dataclass.
  - `on_modified` / `on_created` -> debounced enqueue.
  - `on_deleted` -> immediate `delete_by_filter` against Qdrant.
  - `on_moved` -> delete + re-enqueue at the destination.
  - File filter: `.md` only, hidden parts skipped (`.git`, `.DS_Store`,
    etc.), `node_modules` / `__pycache__` / `.venv` skipped.
- `src/sdet_brain/cli/watcher_cli.py`: `sdet-brain-watcher` CLI.
  Reads `WATCH_PATHS` (comma-separated) from env or `--paths`,
  defaults debounce to `WATCHER_DEBOUNCE_MS`. `SIGINT` / `SIGTERM`
  signal handlers trip a `threading.Event` so the main loop returns
  cleanly and the watcher drains pending events on exit.
- `docker/docker-compose.yml`: opt-in `sdet-brain-watcher` service
  under the `watcher` profile, with a host bind mount overridable
  through `SDET_BRAIN_CORPUS_HOST`.
- `tests/ingestion/test_watcher.py`: 9 tests.
  - 4 unit tests for `is_relevant_path`.
  - Debounce collapse (5 rapid `on_modified` -> 1 ingest call).
  - Delete event triggers `delete_by_filter`.
  - Filter ignores hidden / non-Markdown / `node_modules`.
  - Directory events suppressed.
  - End-to-end smoke against the live Qdrant container.
- README "Live sync mode" section.
- CHANGELOG `[Unreleased]` entry.

## Atomic commit

- `<sha> feat(ingestion): file watcher with debouncing + delete handling`

## Numbers

- Files added: 3 (watcher + CLI + tests).
- Files modified: 3 (CHANGELOG, README, docker-compose).
- Tests added: 9 (77 total).
- Quality gates: ruff clean, mypy strict 42 source files clean,
  pytest 77/77 in 9.69 s.

## Lessons learned

- Keeping the debounce queue on the watcher itself (rather than
  pushing to a separate executor) made the worker loop tiny and
  testable: `_process_due()` is exposed so tests advance time with
  `monkeypatch.setattr(time, "monotonic", ...)` instead of sleeping.
- Watchdog's `Observer.join()` can take a few seconds on macOS when
  FSEvents has no inflight notifications; the worker thread goes
  daemon to avoid blocking interpreter shutdown when the process is
  killed mid-drain.
- `on_moved` is rare in editors (most "save" flows are
  modify-in-place) but matters for `git checkout` swaps inside the
  watch tree - cheaper to handle once than to triage why renamed
  files double-show in `list_sources`.

## Out-of-scope items captured

- The CLI's hard-coded brand corpus paths in `_default_source_config`
  mirror the ingest CLI - same follow-up applies (lift to env vars
  for VPS portability).
- The Docker `watcher` profile assumes a host bind mount; production
  VPS deploys (Tier 3) will likely run watcher offline and use the
  REST `/ingest` endpoint instead. Documented but not yet wired.

## Next task

- T1-09 (SDE-26) Initial corpus ingest. Unblocked.
