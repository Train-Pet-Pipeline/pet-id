# pet-id

Pet identity system for Train-Pet-Pipeline.

## Status

Bootstrap — seeded from `pet-demo/core` (commit hash will be recorded in initial commit). Provides baseline detection + ReID + tracking + pose + narrative backends. Pet-id-specific registration/identification features land in follow-up commits.

## Scope

- Multi-view photo registration of pets → embedding library
- Frame-level identification: frame → pet_id
- Integration point for downstream behavioral attribution (not in scope for this repo)

## Relation to sibling repos

- Import contracts from `pet-schema`
- Runs under `pet-infra` plugin runtime (Phase 1+)
- Reuses detector / ReID / tracker backends from pet-demo/core

## Required reading

Parent monorepo guide: `../pet-infra/docs/DEVELOPMENT_GUIDE.md`
