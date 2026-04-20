# pet-id

## Required reading

- Parent monorepo guide: `../pet-infra/docs/DEVELOPMENT_GUIDE.md`
- pet-demo CLAUDE.md (bootstrap source): `../pet-demo/CLAUDE.md`

## Directory layout (bootstrap)

- `src/purrai_core/` — inherited from pet-demo/core, unchanged on bootstrap
- `tests/` — inherited tests

Future pet-id-specific packages will be added alongside (e.g., `src/pet_id_registry/`).

## Git workflow

`feature/* → dev → main`, same as all sibling repos. Initial bootstrap commits are the only exception to direct-push-to-main.

## Backend policy

Same as pet-demo: all backends import real libraries. Tests may `patch()`; no production mock substitution.

## Commit format

`feat|fix|refactor|test|docs(pet-id): short description`

## Key constraints

- All numerics from `params.yaml` — never hardcode
- Protocols are structural (typing.Protocol)
- `purrai_core` package imports preserved from bootstrap; pet-id features go in new sibling packages
