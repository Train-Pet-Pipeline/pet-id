# pet-id first-round design — thin enrollment + identify library

**Date:** 2026-04-21
**Status:** Approved (brainstorming gate passed)
**Owner:** pet-id
**Scope:** one repo (`pet-id`), one feature branch → dev → main

## 1. Context

`pet-id` was bootstrapped (commit `37243bc`) from `pet-demo/core`, inheriting `purrai_core`: yolov10 detector, bytetrack tracker, OSNet ReID, mmpose, and the streaming `full_pipeline`. Inherited code is preserved verbatim per `pet-id/CLAUDE.md` ("purrai_core package imports preserved from bootstrap; pet-id features go in new sibling packages").

What's missing to justify the repo's name: a way to turn those per-frame embeddings into **persistent, addressable pet identities**. A family user wants to enroll a pet once and later ask "who is this?" on a new photo or video frame.

This is the first round. Scope is deliberately thin: a standalone library + CLI for **enrollment** (build a `PetCard`) and **identify** (match a query image against the gallery). Downstream integration with `purrai_core.pipelines.full_pipeline` is out of scope for this round and will be picked up separately.

## 2. Goals

1. User can enroll a pet from **photos, a video, or both** with a single CLI command, providing a name and optional metadata (species, breed, sex, age, etc.). The tool produces a persistent `PetCard` stored in a local gallery directory.
2. User can run `identify` on a new photo or video frame and get back the matched pet's name (or `unknown`) for each detected pet bbox.
3. Gallery is a **flat-file library** on disk: one subdirectory per pet, containing the card JSON, a cover image, and per-view crops + embeddings. No database, no service.
4. All numerics (detection thresholds, sampling FPS, view cap, match threshold) read from `params.yaml`. No hardcoded magic numbers.
5. Code lives in a **new sibling package** `src/pet_id_registry/`; `purrai_core` is unchanged.

## 3. Non-goals (out of scope for first round)

- Integration into `purrai_core.pipelines.full_pipeline` (future round: label each tracked bbox with the identified pet name + render skeleton).
- Interactive multi-pet disambiguation at enrollment. First-round rule: pick the **largest bbox per frame**. Multi-pet scenes are addressed by asking the user to capture each pet separately.
- Advanced view-selection heuristics (farthest-first diversity sampling, Laplacian blur filtering, exposure filters). First round uses uniform FPS sampling capped at `max_views`.
- Automatic re-embedding on model upgrade, trash / garbage collection, atomic delete with recovery.
- Promoting `PetCard` into `pet-schema` as a cross-repo contract. First round keeps it local to pet-id; Phase 2 may upstream it alongside `VisionAnnotation.pet_id`.
- GUI / web UI / household or multi-owner bookkeeping.
- Training a pet-identity-specialized ReID head. First round uses the inherited OSNet (person-ReID trained) as a generic visual-similarity feature extractor. Single-photo enrollment is supported but accuracy at extreme angles depends on future training work (Phase 3).

## 4. User-visible surface

### CLI — `petid`

```
petid register <input> --name NAME [--species cat|dog|other]
                                   [--breed BREED] [--sex SEX]
                                   [--birthdate YYYY-MM-DD] [--weight-kg FLOAT]
                                   [--markings TEXT] [--owner-name TEXT]
                                   [--medical-notes TEXT]
                                   [--cover-photo PATH]
                                   [--library-root PATH]
                                   [--force]
petid identify <input> [--library-root PATH] [--json]
petid list            [--library-root PATH] [--json]
petid show <pet_id>   [--library-root PATH] [--json]
petid delete <pet_id> [--library-root PATH] [--yes]
```

`<input>` for `register` and `identify` accepts:

- A single image file (`.jpg / .png / .webp`).
- A single video file (`.mp4 / .mov / .mkv`).
- A directory (treated as "all images within, sorted by filename"; mixed video files inside are ignored in first round).

`petid register --help` appends this shooting-guide snippet verbatim:

> Capture tip: record a 5–10 second video walking a full circle around the pet, **or** take 5+ photos from different angles (front, left side, right side, top-down, sitting). More views → better recall on new angles.

### Gallery layout on disk

```
<library_root>/
├── index.json                      # {"pet_id": "name", …} derived view; rebuildable from cards
└── <pet_id>/
    ├── card.json                   # PetCard.model_dump_json()
    ├── cover.jpg                   # user-supplied --cover-photo, or first view crop as default
    └── views/
        ├── 0001.jpg                # 128×256 crop (OSNet input size)
        ├── 0001.npy                # L2-normalized float32 embedding, shape (embedding_dim,)
        ├── 0002.jpg
        ├── 0002.npy
        └── …
```

`<library_root>` defaults to `pet_id.library_root` from `params.yaml`; overridable per command with `--library-root`. No other tunables (`fps_sample`, `max_views`, `similarity_threshold`, detector/reid config) expose CLI overrides in first round — edit `params.yaml` to change them.

### `PetCard` (Pydantic v2 model, local to pet-id)

**Required:** `pet_id: str`, `name: str`, `species: Literal["cat", "dog", "other"]`, `created_at: datetime`, `schema_version: str`, `cover_photo_uri: str`, `views: list[RegisteredView]` (length ≥ 1).

**Optional:** `breed: str | None`, `sex: Literal["male", "female", "unknown"] | None`, `birthdate: date | None`, `weight_kg: float | None`, `markings: str | None`, `owner_name: str | None`, `medical_notes: str | None`, `extra: dict[str, Any] = {}`.

### `RegisteredView`

`view_id: str` (zero-padded sequential, e.g., `"0003"`), `pose_hint: str | None` (reserved; not populated in first round), `crop_uri: str` (relative path under gallery), `embedding_uri: str`.

### `pet_id` assignment

`pet_id = sha256(first_view_embedding.tobytes())[:8]`, computed after the embedding is cast to `np.float32`, forced to little-endian (`.astype("<f4")`), and made contiguous (`.tobytes(order="C")`). This keeps ids portable across hosts and across future dtype changes. Content-addressed; pet_id never collides in practice (family-scale library). `--force` on `register` overwrites an existing pet_id.

`schema_version` starts at `"1.0.0"` for the first-round release. Any change to `PetCard` / `RegisteredView` shape bumps semantically.

## 5. Algorithms

### Enrollment

**Image input:** For each image, run yolov10 on the full frame. Take the largest bbox. Crop → resize to 128×256 → OSNet embed → L2-normalize. Store as one view. Images where yolov10 detects nothing are **skipped with a warning**, not fatal.

**Video input:** Sample frames at `pet_id.fps_sample` frames per second (e.g., 2 FPS over a 10-second video → 20 candidate frames). For each sampled frame, same as image input. After collecting candidates, **truncate to the first `pet_id.max_views`** (uniform order; first round does not re-rank for diversity).

**Card construction:** `pet_id` from first view embedding; `cover_photo_uri` = `--cover-photo` if provided, else the crop from view `0001`. Write crops + embeddings under `<library_root>/<pet_id>/views/`, write `card.json`, update `index.json`.

Atomic-enough: write into a temp subdirectory under `<library_root>/.tmp/` and `os.replace` into final name. First round does not implement rollback beyond that.

### Identify

Load the query input. Accepted shapes:

- **Single image file** → processed as one frame.
- **Video file** → first round rejects with a clear error ("`identify` takes a still image in first round; extract a frame and retry"). Video scanning is a future-round addition.
- **Directory** → every image file inside is processed in filename-sorted order; one output record per file × bbox. Video files in the directory are ignored with a warning.

Run yolov10 on each frame → one result per bbox. For each bbox: crop → resize → OSNet embed.

For each detected bbox, compute:

```
best_score = max over all views v of all cards c: cosine(query_emb, v.embedding)
best_card  = argmax
label = best_card.name if best_score >= similarity_threshold else "unknown"
```

Output per bbox: `{bbox, pet_id | null, name, score}`. `--json` prints structured; default prints human-readable one-per-line.

### Gallery bookkeeping

- `list`: read `index.json`, fallback to scanning `<library_root>/*/card.json` if index is missing. Output: `pet_id  name  species  #views  created_at`.
- `show`: pretty-print a specific `card.json`.
- `delete`: `rmtree(<library_root>/<pet_id>)`, rebuild `index.json`. `--yes` skips confirmation.

## 6. Configuration (`params.yaml` additions)

```yaml
pet_id:
  library_root: "artifacts/pet_id_library"
  fps_sample: 2
  max_views: 8
  similarity_threshold: 0.55
```

`detector.class_whitelist`, `detector.conf_threshold`, `reid.embedding_dim`, `reid.device`, `reid.model_name` are already configured and reused. `reid.similarity_threshold` is the default in `OSNetReid`'s per-track matcher and is **not** reused for gallery identify — gallery identify uses the dedicated `pet_id.similarity_threshold` because the failure modes differ (per-track is online re-id, gallery is cold matching against enrolled identities).

## 7. Error handling

Three hard-fail conditions, all exit non-zero with a human-readable message to stderr:

1. Input path does not exist, is unreadable, or OpenCV cannot decode the video container.
2. Every frame/photo in the input produces zero detections (after the whole input is traversed). Message: "no pet detected in input; check detector confidence or input quality".
3. `register` without `--force` and `<library_root>/<pet_id>/` already exists.

Per-frame soft skips are warnings to stderr; the run continues.

Identify on an empty library returns `unknown (library empty)` for every bbox and exits 0 — not an error.

No bare `except:` or `except Exception: pass`. Retries for model loading reuse `purrai_core.utils.retry` when applicable.

## 8. Module boundaries (`src/pet_id_registry/`)

| Module | Responsibility | Key exports |
|---|---|---|
| `card.py` | Pydantic models + species enum | `PetCard`, `RegisteredView`, `PetSpecies` |
| `library.py` | On-disk gallery CRUD + `identify(query_embedding)` | `Library` |
| `enroll.py` | Orchestrate detection + embedding for photos/video, build and save `PetCard` | `enroll_photos()`, `enroll_video()` |
| `cli.py` | `click` app wiring all subcommands | `main()` (registered as `petid` entry point) |

Each file aims to stay under ~150 lines. Detector / embedder dependencies are taken via constructor injection using structural typing (`typing.Protocol`) so tests can substitute fakes without monkeypatching imports.

Two protocols are introduced (both in `enroll.py` or a dedicated `protocols.py`):

```python
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Bbox]: ...

class Embedder(Protocol):
    embedding_dim: int
    def embed_crop(self, crop: np.ndarray) -> np.ndarray: ...   # returns L2-normalized float32 (embedding_dim,)
```

`Bbox` is a simple dataclass/TypedDict `{x1, y1, x2, y2, class_id, confidence}` (borrow `purrai_core`'s existing type if present). Adapter wrappers translate `OSNetReid` and `Yolov10Detector` signatures into these protocols; the enroll/library code only sees the protocols.

## 9. Testing strategy

**Unit:**

- `PetCard` / `RegisteredView` validation (required fields, enum values, schema_version format).
- `Library.save/load/list/delete` round-trip with tmp dirs.
- `Library.identify` max-cosine correctness: best-match hits, tie-breaking is deterministic, below-threshold returns `None`.
- `RegisteredView.pose_hint` round-trips as `None` (guards against a later PR silently populating this reserved field without a schema bump).

**Integration (fake backends via Protocol injection):**

- Enroll 3 photos → `PetCard` has 3 views, files on disk match, `pet_id` is reproducible from input.
- Enroll a 10-second video → FPS sampling + `max_views` cap behave as specified.
- Frames with zero detections → skipped, final card has fewer views, run succeeds.
- Entire input zero detections → exits non-zero with the documented message.
- Identify against a 2-card gallery: query close to card A → returns A; query far from both → returns `unknown`; gallery empty → returns `unknown (library empty)` exits 0.
- `pet_id` collision on `register` without `--force` → fails; with `--force` → overwrites.

**CLI (`click.testing.CliRunner` + tmp library root):**

- All five subcommands end-to-end, covering `--json` output shape and exit codes.
- `--library-root` override verified independent of `params.yaml` default.

**Out of automated tests (first round):** real-model accuracy numbers (no pet-identity benchmark yet), GPU-specific paths (tests force CPU), large-library scaling.

## 10. Engineering caveats (documented, not fixed in this round)

- OSNet is trained on person-ReID data. When applied to pets, it functions as a generic body-level visual-similarity feature extractor, not a true identity embedder. Single-photo enrollment will under-recall on unseen angles. Capture guidance (multi-angle / short video) is the first-round mitigation; a pet-specialized ReID head is a Phase 3 item.
- Farthest-first selection and blur filtering would improve per-view quality per stored byte. Deferred by YAGNI until the first round ships and user feedback shows whether it's needed.

## 11. Release

Single feature branch `feature/pet-id-first-round` off `dev`. PR targets `dev`. Standard pet-repo flow: CI green + 1 reviewer approve → merge → back-to-main release PR → tag `v0.1.0` (pet-id has never tagged before, starts at 0.1.0).

## 12. Cross-references

- Parent monorepo guide: `pet-infra/docs/DEVELOPMENT_GUIDE.md` (git workflow, params.yaml rule, error handling).
- Bootstrap source: `pet-demo` (retained `purrai_core`).
- Multi-model refactor spec: `pet-infra/docs/superpowers/specs/2026-04-20-multi-model-pipeline-design.md` (pet-id is placed between Phase 1 Foundation and Phase 2 Data & Annotation).
