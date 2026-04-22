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

## pet-id registry (v0.1.0)

### Install

```bash
pip install -e ".[detector,reid]"
```

Entry point `petid` is registered automatically.

### Enroll a pet

```bash
# one photo
petid register photos/mimi_01.jpg --name Mimi --species cat

# a folder of photos
petid register photos/mimi/ --name Mimi --species cat --breed "Domestic Shorthair"

# a short video
petid register videos/mimi_circle.mp4 --name Mimi --species cat --weight-kg 4.2
```

**Capture tip:** a 5–10 second video walking a full circle around the pet, OR
5+ photos from different angles (front / left / right / top / sitting).

### Identify

```bash
petid identify query.jpg
# → query.jpg bbox=[42, 30, 310, 240] → Mimi (score=0.812)
petid identify query.jpg --json
```

### Browse

```bash
petid list
petid show <pet_id>
petid delete <pet_id> --yes
```

### Configuration

All numerics (library root, FPS sampling, view cap, match threshold) live in
`params.yaml` under the `pet_id:` block. See
`docs/superpowers/specs/2026-04-21-pet-id-first-round-design.md` for the full
design.

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)
