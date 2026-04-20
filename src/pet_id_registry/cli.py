"""petid CLI — register / identify / list / show / delete."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import click
import yaml

from pet_id_registry.card import PetSex, PetSpecies
from pet_id_registry.enroll import NoDetectionsError, enroll_photos, enroll_video
from pet_id_registry.library import Library, PetAlreadyExistsError

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


def _load_params(path: Path) -> dict[str, Any]:
    """Load params.yaml from disk."""
    return yaml.safe_load(path.read_text())


def build_detector(cfg: dict[str, Any]):
    """Build the pet detector from params.detector."""
    from purrai_core.backends.yolov10_detector import YOLOv10Detector

    return YOLOv10Detector(cfg)


def build_embedder(cfg: dict[str, Any]):
    """Build the reid embedder adapter from params.reid."""
    from purrai_core.backends.osnet_reid import OSNetReid
    from pet_id_registry.backends.osnet_embedder import OSNetEmbedderAdapter

    return OSNetEmbedderAdapter(OSNetReid(cfg))


def _classify_input(path: Path) -> str:
    """Classify an input path as image / video / dir, or raise UsageError."""
    if path.is_dir():
        return "dir"
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    raise click.UsageError(f"unsupported input type: {path}")


def _collect_images(directory: Path) -> list[Path]:
    """Return sorted image files directly under directory."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


@click.group()
@click.option("--params", "params_path", type=click.Path(exists=True, path_type=Path),
              default=Path("params.yaml"), show_default=True)
@click.pass_context
def main(ctx: click.Context, params_path: Path) -> None:
    """petid: pet identity enrollment + identification CLI."""
    ctx.ensure_object(dict)
    ctx.obj["params"] = _load_params(params_path)


@main.command("register")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--species", type=click.Choice([s.value for s in PetSpecies]), required=True)
@click.option("--breed", default=None)
@click.option("--sex", type=click.Choice([s.value for s in PetSex]), default=None)
@click.option("--birthdate", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
@click.option("--weight-kg", type=float, default=None)
@click.option("--markings", default=None)
@click.option("--owner-name", default=None)
@click.option("--medical-notes", default=None)
@click.option("--cover-photo", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--library-root", type=click.Path(path_type=Path), default=None)
@click.option("--force", is_flag=True)
@click.pass_context
def register_cmd(
    ctx: click.Context,
    input_path: Path,
    name: str,
    species: str,
    breed: str | None,
    sex: str | None,
    birthdate,
    weight_kg: float | None,
    markings: str | None,
    owner_name: str | None,
    medical_notes: str | None,
    cover_photo: Path | None,
    library_root: Path | None,
    force: bool,
) -> None:
    """Register a pet from photos, a video, or a directory of photos.

    \b
    Capture tip: record a 5-10 second video walking a full circle around the pet,
    OR take 5+ photos from different angles (front, left side, right side,
    top-down, sitting). More views → better recall on new angles.
    """
    params = ctx.obj["params"]
    pet_id_cfg = params["pet_id"]
    root = Path(library_root) if library_root else Path(pet_id_cfg["library_root"])
    library = Library(root)

    detector = build_detector(params["detector"])
    embedder = build_embedder(params["reid"])

    metadata: dict[str, Any] = {}
    if breed:
        metadata["breed"] = breed
    if sex:
        metadata["sex"] = sex
    if birthdate:
        metadata["birthdate"] = birthdate.date()
    if weight_kg is not None:
        metadata["weight_kg"] = weight_kg
    if markings:
        metadata["markings"] = markings
    if owner_name:
        metadata["owner_name"] = owner_name
    if medical_notes:
        metadata["medical_notes"] = medical_notes

    kind = _classify_input(input_path)
    try:
        if kind == "image":
            card = enroll_photos(
                image_paths=[input_path], name=name, species=PetSpecies(species),
                detector=detector, embedder=embedder, library=library,
                created_at=_dt.datetime.now(_dt.UTC), cover_photo=cover_photo,
                force=force, metadata=metadata,
            )
        elif kind == "dir":
            images = _collect_images(input_path)
            if not images:
                raise click.UsageError(f"no images found in directory: {input_path}")
            card = enroll_photos(
                image_paths=images, name=name, species=PetSpecies(species),
                detector=detector, embedder=embedder, library=library,
                created_at=_dt.datetime.now(_dt.UTC), cover_photo=cover_photo,
                force=force, metadata=metadata,
            )
        elif kind == "video":
            card = enroll_video(
                video_path=input_path, name=name, species=PetSpecies(species),
                detector=detector, embedder=embedder, library=library,
                fps_sample=float(pet_id_cfg["fps_sample"]),
                max_views=int(pet_id_cfg["max_views"]),
                created_at=_dt.datetime.now(_dt.UTC), cover_photo=cover_photo,
                force=force, metadata=metadata,
            )
        else:  # pragma: no cover
            raise click.UsageError(f"unsupported input: {input_path}")
    except NoDetectionsError as e:
        raise click.ClickException(f"no pet detected: {e}")
    except PetAlreadyExistsError as e:
        raise click.ClickException(
            f"pet already exists (pet_id={e}); rerun with --force to overwrite"
        )

    click.echo(f"enrolled {card.name} [{card.pet_id}] with {len(card.views)} view(s)")
