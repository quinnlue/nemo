"""Download HuggingFace datasets and write NeMo-format manifest files.

Provides :func:`prepare_manifests` as the single entry-point used by
notebooks and scripts.  Audio arrays are decoded in the main process
(HF datasets are not pickle-safe) and written to ``.wav`` files in
parallel via a process pool.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf
from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm

from asr_benchmark.config import PROJECT_ROOT

# ── Default dataset IDs ──────────────────────────────────────────────────────
DATASET_ID = "quinnlue/asr"
NOISE_ID = "quinnlue/realclass"

# ── Output paths ─────────────────────────────────────────────────────────────
MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "ortho_dataset"
AUDIO_CACHE = MANIFEST_DIR / "audio"
NOISE_CACHE = MANIFEST_DIR / "noise"

TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
NOISE_MANIFEST = MANIFEST_DIR / "noise_manifest.jsonl"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _write_wav(wav_path_str: str, array, sr: int) -> str:
    """Write a single wav file.  Runs in a worker process."""
    sf.write(wav_path_str, array, sr)
    return wav_path_str


def _hf_split_to_manifest(
    split_dataset,
    manifest_path: Path,
    audio_cache: Path,
    max_duration: float = 25.0,
    sample_n: Optional[int] = None,
    max_workers: int = 32,
) -> int:
    """Convert a HuggingFace ASR split to a NeMo-format manifest JSONL.

    Clips longer than *max_duration* or with corrupt audio are skipped.
    """
    if sample_n:
        logger.info(f"Sampling {sample_n} utterances")
        split_dataset = split_dataset.select(range(min(sample_n, len(split_dataset))))

    records: list[dict] = []
    skipped = 0
    errors = 0
    futures: dict = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, row in enumerate(tqdm(split_dataset, desc=f"{manifest_path.stem} (decode)")):
            dur = row["audio_duration_sec"]
            if dur > max_duration:
                skipped += 1
                continue
            uid = row.get("utterance_id", f"utt_{idx}")
            wav_path = audio_cache / f"{uid}.wav"

            if not wav_path.exists():
                try:
                    array = row["audio_path"]["array"]
                    sr = row["audio_path"]["sampling_rate"]
                except (RuntimeError, Exception) as e:
                    errors += 1
                    logger.warning(f"Skipping {uid} (decode): {e}")
                    continue
                fut = pool.submit(_write_wav, str(wav_path), array, sr)
                futures[fut] = (uid, wav_path, dur, row["orthographic_text"])
            else:
                records.append({
                    "audio_filepath": str(wav_path),
                    "duration": dur,
                    "text": row["orthographic_text"],
                })

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{manifest_path.stem} (write)"):
            uid, wav_path, dur, text = futures[fut]
            try:
                fut.result()
                records.append({
                    "audio_filepath": str(wav_path),
                    "duration": dur,
                    "text": text,
                })
            except Exception as e:
                errors += 1
                logger.warning(f"Skipping {uid} (write): {e}")

    logger.info(
        f"Wrote {len(records)} entries to {manifest_path.name} "
        f"(skipped {skipped} clips > {max_duration}s, {errors} decode errors)"
    )
    pd.DataFrame(records).to_json(manifest_path, orient="records", lines=True)
    return len(records)


def _noise_to_manifest(
    noise_dataset,
    manifest_path: Path,
    noise_cache: Path,
    max_workers: int = 8,
) -> int:
    """Convert a HuggingFace noise dataset to a NeMo-format noise manifest.

    Each row is expected to have an ``"audio"`` column (dict with ``array``
    and ``sampling_rate``).  The manifest JSONL contains ``audio_filepath``,
    ``duration``, and ``offset`` (always 0.0) as required by NeMo's
    ``AudioAugmentor``.
    """
    records: list[dict] = []
    futures: dict = {}
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, row in enumerate(tqdm(noise_dataset, desc="noise (decode)")):
            uid = f"noise_{idx}"
            wav_path = noise_cache / f"{uid}.wav"

            try:
                array = row["audio"]["array"]
                sr = row["audio"]["sampling_rate"]
            except (RuntimeError, Exception) as e:
                errors += 1
                logger.warning(f"Skipping {uid} (decode): {e}")
                continue

            duration = len(array) / sr

            if not wav_path.exists():
                fut = pool.submit(_write_wav, str(wav_path), array, sr)
                futures[fut] = (uid, wav_path, duration)
            else:
                records.append({
                    "audio_filepath": str(wav_path),
                    "duration": duration,
                    "offset": 0.0,
                })

        for fut in tqdm(as_completed(futures), total=len(futures), desc="noise (write)"):
            uid, wav_path, duration = futures[fut]
            try:
                fut.result()
                records.append({
                    "audio_filepath": str(wav_path),
                    "duration": duration,
                    "offset": 0.0,
                })
            except Exception as e:
                errors += 1
                logger.warning(f"Skipping {uid} (write): {e}")

    logger.info(
        f"Wrote {len(records)} noise entries to {manifest_path.name} "
        f"({errors} decode/write errors)"
    )
    pd.DataFrame(records).to_json(manifest_path, orient="records", lines=True)
    return len(records)


# ── Public API ───────────────────────────────────────────────────────────────

def prepare_manifests(
    *,
    max_duration: float = 25.0,
    sample_n: Optional[int] = None,
    max_workers: int = 8,
) -> dict[str, Path]:
    """Download datasets and write train / val / noise manifests.

    Parameters
    ----------
    max_duration:
        Drop ASR clips longer than this many seconds.
    sample_n:
        If set, use only the first *sample_n* rows of each ASR split
        (useful for fast dev iterations).
    max_workers:
        Number of parallel workers for writing wav files.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"noise"`` mapped to manifest
    ``Path`` objects.
    """
    manifests = {
        "train": TRAIN_MANIFEST,
        "val": VAL_MANIFEST,
        "noise": NOISE_MANIFEST,
    }

    # Fast path: if all manifests already exist, skip downloading entirely.
    if all(p.exists() for p in manifests.values()):
        logger.info("All manifests already exist — skipping download and conversion")
        return manifests

    # Ensure directories exist
    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    NOISE_CACHE.mkdir(parents=True, exist_ok=True)

    # ── ASR dataset ──────────────────────────────────────────────────────
    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        ds = load_dataset(DATASET_ID)
        train_hf = ds["train"]
        val_hf = ds["validation"]

        logger.info(
            f"Loaded HuggingFace dataset '{DATASET_ID}' — "
            f"train: {len(train_hf)}, val: {len(val_hf)}"
        )

        if not TRAIN_MANIFEST.exists():
            _hf_split_to_manifest(
                train_hf, TRAIN_MANIFEST, AUDIO_CACHE,
                max_duration=max_duration, sample_n=sample_n, max_workers=max_workers,
            )
        else:
            logger.info(f"{TRAIN_MANIFEST.name} already exists — skipping")

        if not VAL_MANIFEST.exists():
            _hf_split_to_manifest(
                val_hf, VAL_MANIFEST, AUDIO_CACHE,
                max_duration=max_duration, sample_n=sample_n, max_workers=max_workers,
            )
        else:
            logger.info(f"{VAL_MANIFEST.name} already exists — skipping")

    # ── Noise dataset ────────────────────────────────────────────────────
    if not NOISE_MANIFEST.exists():
        noise_hf = load_dataset(NOISE_ID, split="train")
        logger.info(f"Loaded noise dataset '{NOISE_ID}' — {len(noise_hf)} clips")

        _noise_to_manifest(
            noise_hf, NOISE_MANIFEST, NOISE_CACHE,
            max_workers=max_workers,
        )
    else:
        logger.info(f"{NOISE_MANIFEST.name} already exists — skipping")

    return manifests
