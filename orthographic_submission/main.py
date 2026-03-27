"""Load a NeMo ASR adapter checkpoint and run inference on test data."""

import json
import os
from itertools import islice
from pathlib import Path
import sys

from loguru import logger
from omegaconf import DictConfig, open_dict
import torch

from nemo.collections.asr.models import ASRModel

BATCH_SIZE = 4
PROGRESS_STEP_DENOM = 100


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def _patch_transcribe_lhotse(model: ASRModel) -> None:
    """Work around NeMo hardcoding use_lhotse=True in _setup_transcribe_dataloader."""

    def _patched(config):
        if "manifest_filepath" in config:
            manifest_filepath = config["manifest_filepath"]
            batch_size = config["batch_size"]
        else:
            manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
            batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

        dl_config = {
            "use_lhotse": False,
            "manifest_filepath": manifest_filepath,
            "sample_rate": model.preprocessor._sample_rate,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": config.get("num_workers", min(batch_size, os.cpu_count() - 1)),
            "pin_memory": True,
            "channel_selector": config.get("channel_selector", None),
            "use_start_end_token": model.cfg.validation_ds.get("use_start_end_token", False),
        }
        if config.get("augmentor"):
            dl_config["augmentor"] = config["augmentor"]
        return model._setup_dataloader_from_config(config=DictConfig(dl_config))

    model._setup_transcribe_dataloader = _patched


def main(model_path: Path, data_manifest: Path):
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    logger.info("CUDA device count: {}", torch.cuda.device_count())

    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading NeMo checkpoint: {}", model_path)
    model = ASRModel.restore_from(str(model_path), map_location=device)

    with open_dict(model.cfg):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding)
    _patch_transcribe_lhotse(model)

    data_dir = data_manifest.parent

    with data_manifest.open("r") as fr:
        items = [json.loads(line) for line in fr]

    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)

    logger.info("Processing {} utterances from {}", len(items), data_manifest)

    step = max(1, len(items) // PROGRESS_STEP_DENOM)

    audio_files = [str(data_dir / item["audio_path"]) for item in items]
    utterance_ids = [item["utterance_id"] for item in items]

    logger.info("Starting transcription...")
    predictions = {}
    next_log = step
    processed = 0

    for batch_audio, batch_ids in zip(
        batched(audio_files, BATCH_SIZE), batched(utterance_ids, BATCH_SIZE)
    ):
        raw = model.transcribe(list(batch_audio), batch_size=len(batch_audio))
        if isinstance(raw, tuple):
            raw = raw[0]
        texts = [h.text if hasattr(h, "text") else h for h in raw]
        for uid, text in zip(batch_ids, texts):
            predictions[uid] = text

        processed += len(batch_audio)
        if processed >= next_log:
            logger.info("Processed {} utterances", processed)
            next_log += step

    logger.success("Transcription complete.")

    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing submission file to {}", submission_path)
    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_path = _script_dir / "ASR-Adapter-best.nemo"

    if len(sys.argv) > 2:
        data_manifest = Path(sys.argv[2])
    else:
        data_manifest = Path("data/utterance_metadata.jsonl")

    main(model_path, data_manifest)
