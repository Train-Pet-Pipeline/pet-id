# Changelog

All notable changes to pet-id are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

pet-id is an **independent CLI tool** (spec §5.2) — no peer-dep on
pet-schema / pet-infra / other pet-* packages. Matrix registration is for
version alignment reporting only.

## [0.2.0] - 2026-04-23

Phase 9 — ecosystem optimization pass for pet-id. No dependency
governance work (pet-id has no pet-* deps); findings are in-repo
housekeeping + first-time CI onboarding.

### Added
- `.github/workflows/ci.yml` — lint (ruff check + ruff format --check) +
  type check (mypy src/ under strict) + pytest. Installs `dev + detector
  + reid` extras; pose / narrative / tracker extras deliberately deferred
  to a future heavy-runner job (multi-GB ML deps, tests already patch()
  the backends).
- `.github/workflows/no-wandb-residue.yml` — positive-list scan for a
  bare `\bwandb\b` in first-party code. Forward-looking guard; pet-id
  has never carried W&B code.
- `docs/architecture.md` — 9-chapter architecture doc matching the
  template adopted by the other 7 ecosystem repos; §1 / §2 / §6 stress
  the independence from pet-*.
- `CHANGELOG.md` — this file.

### Changed
- `Makefile` — `make test` now reports coverage for **both** `purrai_core`
  and `pet_id_registry` (was only `purrai_core`); `make lint` runs
  `mypy src/` (was `mypy src/purrai_core` only).
- `README.md` "Relation to sibling repos" section rewritten from stale
  bullet list (claimed pet-schema / pet-infra / pet-demo/core runtime
  deps that don't exist) to a positive statement of independence
  matching spec §5.2 and the actual code.

### Fixed
- `src/pet_id_registry/enroll.py` ruff I001 import-order violation
  (`from typing import Any` was stranded between stdlib and third-party
  blocks). Baseline now ruff-green.

## [0.1.0] - 2026-04-21

Initial bootstrap + first-round PetCard registry + petid CLI.

### Added
- Bootstrap commit from `pet-demo/core@fab10f5`:
  - `purrai_core` algorithm core (detector / re-id / pose / narrative /
    tracker backends + pipelines + utils).
- `pet_id_registry` package (first-round v0.1.0):
  - `card.py` — PetCard + RegisteredView Pydantic models + content-
    addressed `compute_pet_id` (L2-normalized embedding → sha256[:8]).
  - `library.py` — disk-backed library CRUD + identify.
  - `enroll.py` — photo / directory / video enrollment pipeline.
  - `cli.py` — `petid` entry point (register / identify / list / show /
    delete).
  - `backends/osnet_embedder.py` — OSNet adapter.
- 90 tests (mypy strict = true, ruff + ruff format clean).
- 5 optional extras (`detector` / `reid` / `pose` / `narrative` /
  `tracker`) + meta `all`.

### Notes
- pet-id carries no `pet-*` imports — the repo is an intentionally
  isolated CLI tool (spec §5.2 "独立 CLI 工具").
- Previously no CI — lint + test relied on hand-run Makefile targets.
  Fixed in 0.2.0.
