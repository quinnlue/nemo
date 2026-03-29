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
from datasets import Audio, load_dataset
from loguru import logger
from tqdm.auto import tqdm

from asr_benchmark.config import PROJECT_ROOT

# ── Default dataset IDs ──────────────────────────────────────────────────────
DATASET_ID = "quinnlue/asr"
NOISE_ID = "quinnlue/realclass"
IMPULSE_ID = "quinnlue/rirs"

# ── Output paths ─────────────────────────────────────────────────────────────
MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "ortho_dataset"
AUDIO_CACHE = MANIFEST_DIR / "audio"
NOISE_CACHE = MANIFEST_DIR / "noise"
IMPULSE_CACHE = MANIFEST_DIR / "impulse"

TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
NOISE_MANIFEST = MANIFEST_DIR / "noise_manifest.jsonl"
IMPULSE_MANIFEST = MANIFEST_DIR / "impulse_manifest.jsonl"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _write_wav(wav_path_str: str, array, sr: int) -> str:
    """Write a single wav file.  Runs in a worker process."""
    import soundfile as sf
    sf.write(wav_path_str, array, sr)
    return wav_path_str


def _decode_and_write_wav(wav_path_str: str, audio_bytes: bytes, decode_format: str) -> str:
    """Decode raw audio bytes and write to wav.  Runs in a worker process."""
    import io
    import soundfile as sf
    import numpy as np

    with io.BytesIO(audio_bytes) as buf:
        array, sr = sf.read(buf)
    # Downmix to mono if multi-channel (avoids dimensionality mismatch in
    # augmentation ops like impulse-response convolution).
    if array.ndim > 1:
        array = np.mean(array, axis=1)
    sf.write(wav_path_str, array, sr)
    # Return duration so we don't need to re-open the file
    duration = len(array) / sr
    return wav_path_str, duration


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


def _audio_dataset_to_manifest(
    dataset,
    manifest_path: Path,
    cache_dir: Path,
    *,
    label: str = "audio",
    audio_column: str = "audio",
    uid_prefix: str = "clip",
    max_workers: int = 8,
) -> int:
    # Disable HF decoding — get raw bytes instead so segfaults stay in workers
    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    records: list[dict] = []
    futures: dict = {}
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, row in enumerate(tqdm(dataset, desc=f"{label} (decode)")):
            uid = f"{uid_prefix}_{idx}"
            wav_path = cache_dir / f"{uid}.wav"

            if wav_path.exists():
                # Already written — read duration from file header
                import soundfile as sf
                try:
                    info = sf.info(str(wav_path))
                    records.append({
                        "audio_filepath": str(wav_path),
                        "duration": info.frames / info.samplerate,
                        "offset": 0.0,
                    })
                except Exception as e:
                    errors += 1
                    logger.warning(f"Skipping {uid} (re-read): {e}")
                continue

            try:
                audio_bytes = row[audio_column]["bytes"]
                if audio_bytes is None:
                    # Some HF datasets store path instead of bytes
                    audio_path = row[audio_column]["path"]
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
            except Exception as e:
                errors += 1
                logger.warning(f"Skipping {uid} (read bytes): {e}")
                continue

            fut = pool.submit(_decode_and_write_wav, str(wav_path), audio_bytes, "")
            futures[fut] = (uid, wav_path)

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{label} (write)"):
            uid, wav_path = futures[fut]
            try:
                path_out, duration = fut.result()
                records.append({
                    "audio_filepath": path_out,
                    "duration": duration,
                    "offset": 0.0,
                })
            except Exception as e:
                errors += 1
                logger.warning(f"Skipping {uid} (worker): {e}")

    logger.info(
        f"Wrote {len(records)} {label} entries to {manifest_path.name} "
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
    """Download datasets and write train / val / noise / impulse manifests.

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
    dict with keys ``"train"``, ``"val"``, ``"noise"``, ``"impulse"``
    mapped to manifest ``Path`` objects.
    """
    manifests = {
        "train": TRAIN_MANIFEST,
        "val": VAL_MANIFEST,
        "noise": NOISE_MANIFEST,
        "impulse": IMPULSE_MANIFEST,
    }

    # Fast path: if all manifests already exist, skip downloading entirely.
    if all(p.exists() for p in manifests.values()):
        logger.info("All manifests already exist — skipping download and conversion")
        return manifests

    # Ensure directories exist
    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    IMPULSE_CACHE.mkdir(parents=True, exist_ok=True)

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

        _audio_dataset_to_manifest(
            noise_hf, NOISE_MANIFEST, NOISE_CACHE,
            label="noise", uid_prefix="noise", max_workers=max_workers,
        )
    else:
        logger.info(f"{NOISE_MANIFEST.name} already exists — skipping")

    # ── Impulse dataset ──────────────────────────────────────────────────
    if not IMPULSE_MANIFEST.exists():
        impulse_hf = load_dataset(IMPULSE_ID, split="train")
        logger.info(f"Loaded impulse dataset '{IMPULSE_ID}' — {len(impulse_hf)} clips")

        _audio_dataset_to_manifest(
            impulse_hf, IMPULSE_MANIFEST, IMPULSE_CACHE,
            label="impulse", uid_prefix="impulse", max_workers=max_workers,
        )
    else:
        logger.info(f"{IMPULSE_MANIFEST.name} already exists — skipping")

    return manifests
