# pet-id Architecture

## §1 Repository Responsibility

**pet-id is an intentionally independent CLI tool** (ecosystem-optimization spec §5.2 "独立 CLI 工具") for pet identity enrollment + identification.

Unlike every other repo in the Train-Pet-Pipeline ecosystem (pet-schema / pet-infra / pet-data / pet-annotation / pet-train / pet-eval / pet-quantize / pet-ota), pet-id **does not import any `pet-*` package at runtime**. It participates in the ecosystem only via `compatibility_matrix.yaml` registration for version-alignment reporting — there is no peer-dep install order, no plugin entry point, no registry registration.

It ships two top-level packages:

1. **`purrai_core`** — algorithm core, seeded from `pet-demo/core@fab10f5` at bootstrap. Contains detector / re-id / pose / narrative / tracker backends + pipelines + utils. Can be consumed standalone by other CLIs in the future.
2. **`pet_id_registry`** — registry + CLI layer that consumes `purrai_core`. Exposes the `petid` entry point (`register / identify / list / show / delete`) and a disk-backed gallery of PetCards.

Pipeline position:

```
(standalone tool; not in the train → eval → quantize → ota chain)

photos / videos ──► petid register ──► PetCard + embedding library on disk
                                              │
                                              ▼
                         query image ──► petid identify ──► {pet_id, score, name}
```

**Does:**
- Detect a pet in one or more photos (or sample frames from a video), crop the largest bbox, compute an OSNet embedding, hash it into a content-addressed `pet_id`, store a `PetCard` + multi-view embedding on disk.
- Identify pets in a still image by matching embeddings against the enrolled library above a configurable similarity threshold.
- Read every numeric (sample FPS, max views, similarity threshold, library root) from `params.yaml`.

**Does not:**
- Import or depend on any `pet-*` runtime package (spec §5.2).
- Ship as a plugin under `pet-infra` orchestrator — there is no `@EVALUATORS` / `@CONVERTERS` / `@OTA` registration.
- Maintain a `ModelCard` contract — `PetCard` is a separate Pydantic model in `pet_id_registry.card` with its own schema_version.
- Cover on-device deployment — pet-id is a host-side CLI tool.

---

## §2 I/O Contract

### Upstream dependencies

pet-id pulls only third-party PyPI packages. No `pet-*` imports.

| Dependency | Mode | Pin / constraint |
|---|---|---|
| `numpy`, `pydantic`, `pyyaml`, `tenacity`, `python-json-logger`, `opencv-python-headless`, `pillow`, `click` | core runtime | `>=` floors in `pyproject.toml:dependencies` |
| `ultralytics` | optional `[detector]` | `>=8.3` |
| `torchreid`, `torch`, `torchvision` | optional `[reid]` | `>=0.2.5` / `>=2.1` |
| `mmpose`, `mmcv`, `mmengine` | optional `[pose]` | `>=1.3` / `>=2.1` / `>=0.10` |
| `transformers`, `accelerate`, `torch`, `qwen-vl-utils` | optional `[narrative]` | `>=4.45` / `>=0.30` |
| `lap`, `scipy`, `boxmot` | optional `[tracker]` | `>=0.4` / `>=1.11` / `>=10.0` |
| `pet-id[all]` | meta-extra | rolls up the 5 backend extras |

### Inputs

| Source | Consumer | Notes |
|---|---|---|
| Still image (.jpg/.jpeg/.png/.webp) | `petid register` / `petid identify` | decoded by `cv2.imread`; largest bbox cropped |
| Directory of images | `petid register` / `petid identify` | flat iteration; each file enrolls one view |
| Video (.mp4/.mov/.mkv) | `petid register` | `purrai_core.utils.video_io` samples frames at `pet_id.fps_sample` FPS, caps at `pet_id.max_views` |
| `params.yaml` | CLI command group | `detector / reid / pet_id / tracker / pose / narrative` sections |
| `--library-root` override | every command | falls back to `params.pet_id.library_root` |

### Outputs

- On-disk library under `<library-root>/<pet_id>/`:
  - `card.json` — `PetCard` Pydantic model serialized to JSON
  - `cover.jpg` — cover photo (either user-supplied or first crop)
  - `views/<view_id>.jpg` — each registered crop
  - `views/<view_id>.npy` — each OSNet embedding (float32, L2-normalized)
- `petid identify` outputs `{file, bbox, pet_id, name, score}` records to stdout (text) or JSON via `--json`.

### Downstream consumers

None within the Train-Pet-Pipeline monorepo. External callers may consume the `PetCard` JSON via any JSON-Schema-compatible tool; there is no published cross-repo schema contract.

---

## §3 Architecture Overview

### Directory tree

```
src/
├── purrai_core/                           ← algorithm core (bootstrapped from pet-demo/core)
│   ├── __init__.py                        ← __version__ = "0.2.0"
│   ├── config.py                          ← Config dataclass + load_config(params.yaml)
│   ├── types.py                           ← BBox, Detection, etc.
│   ├── interfaces/                        ← typing.Protocol definitions
│   │   ├── detector.py                    ← Detector
│   │   ├── reid.py                        ← ReID
│   │   ├── tracker.py                     ← Tracker
│   │   ├── pose.py                        ← Pose
│   │   └── narrative.py                   ← Narrative
│   ├── backends/                          ← concrete implementations (import real ML libs)
│   │   ├── yolov10_detector.py
│   │   ├── osnet_reid.py
│   │   ├── bytetrack_tracker.py
│   │   ├── mmpose_pose.py
│   │   ├── pose_schema.py
│   │   └── qwen2vl_narrative.py
│   ├── pipelines/
│   │   └── full_pipeline.py               ← compose detector → tracker → re-id → pose → narrative
│   ├── stitch/
│   │   └── id_stitch.py                   ← cross-clip id re-identification
│   └── utils/
│       ├── video_io.py                    ← decode video, sample frames at configured FPS
│       ├── logging.py                     ← JSON logging setup
│       └── retry.py                       ← tenacity wrappers
│
└── pet_id_registry/                       ← CLI + registry layer
    ├── __init__.py                        ← __version__ = "0.2.0" (lockstep with purrai_core)
    ├── card.py                            ← PetCard + RegisteredView + compute_pet_id
    ├── protocols.py                       ← Embedder Protocol
    ├── library.py                         ← disk-backed CRUD + identify
    ├── enroll.py                          ← photo / dir / video enrollment pipeline
    ├── cli.py                             ← petid Click group
    └── backends/
        └── osnet_embedder.py              ← OSNetReid → Embedder adapter

params.yaml                                ← detector / tracker / reid / pet_id / pose / narrative (host default)
params.cpu.yaml                            ← CPU-profile overrides
params.mps.yaml                            ← Apple Silicon MPS-profile overrides
tests/                                     ← 92 tests (8 under interfaces/ contract tests)
Makefile                                   ← setup / test / lint / test-nogpu / clean
Dockerfile                                 ← repro container
.github/workflows/
├── ci.yml                                 ← Phase 9 added — lint + mypy + pytest (dev,detector,reid)
└── no-wandb-residue.yml                   ← Phase 9 added — positive-list W&B scan
```

### High-level dataflow

```
user
 │
 ▼
petid register <input>                     petid identify <image>
 │                                          │
 ▼                                          ▼
_classify_input (image / dir / video) ◄───┤
 │                                          │
 ▼                                          ▼
build_detector(params.detector) ──► yolov10_detector.YOLOv10Detector
 │                                          │
 ▼                                          │
build_embedder(params.reid) ──► OSNetReid + OSNetEmbedderAdapter
 │                                          │
 ▼                                          ▼
enroll.enroll_{photos,video}          for each detected bbox:
   │                                      │
   ├─► crop largest bbox                  ├─► embed crop → float32 vector
   ├─► embed crop                         ├─► library.identify(q, threshold)
   ├─► compute_pet_id(embedding)          └─► emit {pet_id, name, score}
   ├─► build PetCard
   └─► library.save(card)
          │
          ▼
     <library-root>/<pet_id>/{card.json, cover.jpg, views/*}
```

---

## §4 Core Modules

### 4.1 `pet_id_registry.card` — PetCard contract

- `PetCard` (Pydantic, `extra="forbid"`): `pet_id`, `name`, `species`, `created_at`, `schema_version`, `cover_photo_uri`, `views[] (min_length=1)` + optional demographics (breed / sex / birthdate / weight_kg / markings / owner_name / medical_notes) + free-form `extra: dict`.
- `RegisteredView` (frozen): `view_id`, `pose_hint`, `crop_uri`, `embedding_uri`.
- `compute_pet_id(embedding: np.ndarray) -> str` — asserts `||embedding||₂ ≈ 1.0` (atol=1e-3); normalizes to contiguous little-endian `float32`; returns `sha256(bytes)[:8]`. See §8.4 for why this is non-optional.

### 4.2 `pet_id_registry.library` — disk-backed gallery

CRUD surface:

- `save(card)` — write `<root>/<pet_id>/card.json` + copy crops + save embeddings; `PetAlreadyExistsError` unless `force=True`.
- `load(pet_id)`, `list()`, `delete(pet_id)` — straightforward.
- `identify(query_embedding, threshold)` — iterates every enrolled `pet_id`, computes max cosine similarity across all registered views, returns the best match above `threshold` or `None`.

### 4.3 `pet_id_registry.enroll` — enrollment pipeline

Two public entry points:

- `enroll_photos(image_paths, name, species, detector, embedder, library, created_at, cover_photo, force, metadata)` — iterate photos, crop largest bbox per image, embed each crop, average or keep-per-view, emit `PetCard`.
- `enroll_video(video_path, name, species, detector, embedder, library, fps_sample, max_views, created_at, cover_photo, force, metadata)` — sample at `fps_sample` FPS up to `max_views` frames, delegate to the photo pipeline under the hood.

Both raise `NoDetectionsError` when input exhausts without any usable detection.

### 4.4 `pet_id_registry.cli` — petid Click entry point

Commands: `register / identify / list / show / delete`. All commands accept `--library-root` (falls back to `params.pet_id.library_root`). `identify` supports a `--json` flag for machine-readable output. `delete` prompts for confirmation unless `--yes` is passed.

The CLI imports backends lazily inside `build_detector` / `build_embedder` so invoking `petid list` or `petid delete` doesn't pay the `ultralytics` + `torchreid` import cost.

### 4.5 `purrai_core.interfaces.*` — structural Protocols

Each backend category has a `typing.Protocol` defining the minimum shape consumers expect. Concrete backends satisfy the protocol *structurally* — no inheritance required. This is what keeps `pet_id_registry` decoupled from `purrai_core` backend specifics: the registry types against `Detector` Protocol, not against `YOLOv10Detector`.

### 4.6 `purrai_core.backends.*` — concrete implementations

Each file wraps a real ML library (ultralytics / torchreid / bytetrack / mmpose / qwen-vl-utils). Per CLAUDE.md "Backend policy": **all backends import real libraries** — production mock substitution is explicitly forbidden. Tests may `patch()` at the boundary when they don't have the ML stack available.

---

## §5 Extension Points

### Adding a new backend for an existing interface

1. Drop `src/purrai_core/backends/<name>.py` implementing the corresponding Protocol (e.g., `purrai_core.interfaces.detector.Detector`).
2. Declare any new PyPI deps under a new or existing extras group in `pyproject.toml:[project.optional-dependencies]`.
3. Wire into `cli.py` — either extend `build_detector` with a config-driven switch, or add a new `build_*` factory.
4. Add contract tests under `tests/interfaces/test_<name>_contract.py` asserting the Protocol surface.

### Adding a new interface category

1. Create `src/purrai_core/interfaces/<name>.py` with a `typing.Protocol`.
2. Drop at least one backend under `backends/` + a new extras group.
3. Decide whether `pet_id_registry.enroll` or `pet_id_registry.cli` needs to consume it — if not, the interface can live purely in `purrai_core` for other future CLIs to use.

### Adding a new petid CLI command

Add a new `@main.command(...)` inside `cli.py` following the existing pattern (`register_cmd`, `identify_cmd`, etc.). Use `click.pass_context` to reach `ctx.obj["params"]` and reuse `Library(root)` construction.

---

## §6 Dependency Management

### Pin style

- **Third-party only** — every dep is a standard PyPI package with `>=` floors in `pyproject.toml`. No `git+https://` URLs, no peer-dep contracts.
- **Extras for heavyweight backends** — `detector` / `reid` / `pose` / `narrative` / `tracker` separate multi-GB ML installs from the core install; `all` is a meta-extra rolling up all 5.
- **No peer-dep on `pet-*`** — intentional (spec §5.2). Matrix registration is for version alignment reporting only, not install order.

### Install order

No ordering contract. `pip install pet-id` or `pip install pet-id[detector,reid]` is a single resolution. CI installs `dev + detector + reid` extras (enough for the CLI / enrollment / identify paths that tests exercise via mocked backends).

### Version bump policy

- **patch** — docstring / comment changes; no user-visible CLI / PetCard / library-layout change.
- **minor** — new CLI command; new extras group; new public function in `pet_id_registry` or `purrai_core`; docs / Makefile / CI surface change (e.g., this 0.2.0 bump adds CI workflows).
- **major** — PetCard schema bump (with migration); extras group removal; CLI command rename/removal; library directory layout change.

`tests/test_version.py` enforces:
- `pet_id_registry.__version__ == importlib.metadata.version("pet-id")`
- `purrai_core.__version__ == pet_id_registry.__version__` (lockstep — both packages ship from the same pet-id distribution).

---

## §7 Local Dev and Test

```bash
# Shared pet-pipeline conda env (same env all 8 sibling repos use)
conda activate pet-pipeline

# Full install including all ML backends
make setup                 # → pip install -e ".[dev,all]"

# OR install just the subset CI exercises
pip install -e ".[dev,detector,reid]"

# Targets
make test                  # pytest -v --cov=purrai_core --cov=pet_id_registry
make test-nogpu            # pytest -v -m "not gpu"
make lint                  # ruff check + ruff format --check + mypy src/
make clean                 # drop caches + build artifacts

# CLI smoke
petid --help
petid register tests/fixtures/sample.mp4 --name Sample --species cat
petid list
petid identify tests/fixtures/clips/frame_001.jpg
petid delete <pet_id> --yes
```

T6.3 mini-E2E per the ecosystem-optimization plan is **not applicable** for pet-id (independent tool; no orchestrator / registry integration surface). T6 regression is `make lint + make test` only.

---

## §8 Known Complex Points (Preserved for Good Reasons)

### 8.1 Two top-level packages (`pet_id_registry` + `purrai_core`) instead of one

**Why preserved:** `purrai_core` was seeded from `pet-demo/core@fab10f5` on bootstrap (commit `37243bc`) — it's the algorithm core and is self-contained enough to be reusable by other CLIs in the future (internal or external). `pet_id_registry` is the *thin* registry + CLI layer that consumes it. Collapsing them into a single `pet_id` namespace would force any future consumer of the raw algorithms to take the full CLI + disk-library dependency surface.

**What would be lost by removing:** Either the CLI layer fuses with algorithms (anti-pattern: CLI-specific concerns leak into the algorithm core) or the algorithms become private to a CLI-only package (no external reuse).

**Condition to revisit:** A decision that `purrai_core` will never have a second consumer — at which point the two-package layout is overhead.

### 8.2 5 optional extras (`detector` / `reid` / `pose` / `narrative` / `tracker`) + meta `all`

**Why preserved:** Each backend pulls hundreds of megabytes of vendor ML libraries. A developer who only uses `petid register` + `petid identify` needs `detector` + `reid` — not `mmpose` or `qwen-vl-utils`. Keeping them split means `pip install pet-id[detector,reid]` is ~500 MB while `pip install pet-id[all]` can exceed 5 GB.

**What would be lost by removing:** Either every CLI invocation pays a multi-GB install cost (friction for simple registry workflows) or the tool stops supporting pose/narrative/tracker (feature loss).

**Condition to revisit:** A lightweight common ML base ships on PyPI that covers all 5 use cases without size bloat (not imminent).

### 8.3 Intentional independence from `pet-*` ecosystem (spec §5.2)

**Why preserved:** pet-id predates the peer-dep discipline the other 8 repos adopted in Phase 3–8. Bringing it into the peer-dep / plugin model now would require converting `PetCard` into a pet-schema entity (new versioned schema surface), making `petid` a pet-infra plugin entry point (runtime registry registration), and re-reviewing every extras group against pet-* compatibility matrices. None of these produce user-visible value — pet-id's consumers are humans running a CLI, not orchestrators.

**What would be lost by removing (i.e., by merging pet-id into the ecosystem):** Either pet-schema / pet-infra breaking changes start blocking pet-id releases, or pet-id has to track every `compatibility_matrix.yaml` bump — overhead without benefit.

**Condition to revisit:** A recipe actually needs to invoke `petid` as a pipeline stage — at which point adding a thin orchestrator-facing shim package is probably cleaner than retrofitting pet-id itself.

### 8.4 `compute_pet_id` L2-normalize assertion + little-endian float32 canonicalization

**Why preserved:** `pet_id` is meant to be content-addressed — same pet enrolled on an amd64 laptop and an arm64 server must collapse to the same id. Two reproducibility traps:

1. Different host architectures have different native float endianness → `sha256(raw_bytes)` would drift. Casting to `<f4` (little-endian float32) + `ascontiguousarray` before hashing makes the byte layout deterministic.
2. Callers that pass an un-normalized embedding would produce a `pet_id` sensitive to vector magnitude — two photos of the same pet with different brightness could hash differently. Asserting `norm ≈ 1.0 (atol=1e-3)` catches the bug immediately rather than corrupting the library silently.

**What would be lost by removing:** Either cross-host id drift (library migration between machines corrupts) or silent pet_id collisions/mis-keys when callers forget to normalize. Both are very hard to debug after the fact.

**Condition to revisit:** `pet_id` stops being content-addressed (e.g., switches to a UUID + embedding-vs-id sidecar) — unlikely.

### 8.5 `_SCHEMA_VERSION = "1.0.0"` literal in `pet_id_registry.enroll`

**Why preserved:** `PetCard.schema_version` is a code contract — a schema bump requires both a code change (new field / new validator) and a literal bump, and the two need to stay in lockstep within one commit. Moving the literal into `params.yaml` would invite an operator to "upgrade" the schema version by editing a YAML file without touching the code that implements the new shape — a silent lie on-disk.

**What would be lost by removing:** Either schema version becomes a runtime-tunable (wrong abstraction) or every code-level bump ships with a params.yaml diff that operators have to merge manually.

**Condition to revisit:** The project adopts a formal schema-migration framework (e.g., alembic-for-PetCard) where the version moves behind that abstraction.

### 8.6 Algorithm backends import real libraries; tests `patch()` at the boundary

**Why preserved:** Per CLAUDE.md "Backend policy": production mock substitution is forbidden. A `YOLOv10Detector` backend must literally call `ultralytics` — otherwise the "backend" is lying about what it does. Tests that don't want to pay the ML-stack cost `patch()` the concrete backend at the Python import boundary; they are explicit about running against a stub.

**What would be lost by removing:** A class of silent production bugs where a "fake" backend passes tests but fails in production because the real ML library has a different API surface than the stub.

**Condition to revisit:** None — this is a durable policy.

---

## §9 Phase 10+ Follow-ups

1. **Heavy-backends CI job** — `ci.yml` only installs `dev + detector + reid` extras; pose / narrative / tracker backends are covered only by `patch()`-based unit tests. A second `ci-heavy` job on a self-hosted or large GitHub runner that actually installs `[pose,narrative,tracker]` and runs the matching contract tests would catch real-library API drift. Deferred because (a) runner cost, (b) tests already mock the boundary, and (c) no known incidents from the current setup.

2. **`petid identify` video support** — currently rejects video input with "extract a frame and retry". A future command could sample frames at configurable FPS and merge per-frame identifications into a single result (take-majority-or-best-score), mirroring the enrollment video pipeline.

3. **`PetCard` schema versioning** — `_SCHEMA_VERSION = "1.0.0"` has been stable since bootstrap. When the first breaking change lands (e.g., adding a required field), formalize the migration path: either every read-time `load()` checks the version and upgrades on the fly, or a separate `petid migrate <library-root>` command.

4. **`Library.identify` efficiency** — current implementation iterates every enrolled pet and every registered view per query. For libraries beyond ~1000 pets, this becomes linear-scan-expensive. Swap the storage for an approximate-nearest-neighbor index (faiss / hnswlib) when that pain point actually shows up.

5. **`params.cpu.yaml` / `params.mps.yaml` profile documentation** — both profile files exist in the repo root but their intended use (how to select one; are they opt-in overrides or replacements?) is undocumented. Either document in README or merge into a single `params.yaml` + environment-variable toggle.
