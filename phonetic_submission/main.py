"""
Load the model from "wav2vec2-phonetic-final" and run inference on data.
Must run notebooks/phonetic.ipynb first to train the model.
"""

from itertools import islice
import json
from pathlib import Path
import sys

import librosa
from loguru import logger
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

SR = 16000


BATCH_SIZE = 4
PROGRESS_STEP_DENOM = 100  # Update progress bar every 1 // PROGRESS_STEP_DENOM


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def main(model_dir: Path, data_manifest: Path):
    # Diagnostics
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    logger.info("CUDA device count: {}", torch.cuda.device_count())

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_dir, local_files_only=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load manifest and process data
    data_dir = data_manifest.parent

    with data_manifest.open("r") as fr:
        items = [json.loads(line) for line in fr]

    # Sort by audio duration for better batching
    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)

    logger.info(f"Processing {len(items)} utterances from {data_manifest}")

    step = max(1, len(items) // PROGRESS_STEP_DENOM)

    def predict_batch(batch_items):
        audio_arrays = [
            librosa.load(data_dir / item["audio_path"], sr=SR)[0] for item in batch_items
        ]
        inputs = processor(audio_arrays, sampling_rate=SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)

    # Predict
    predictions = {}
    next_log = step
    processed = 0
    logger.info("Starting transcription...")
    for batch in batched(items, BATCH_SIZE):
        preds = predict_batch(batch)
        for item, pred in zip(batch, preds):
            predictions[item["utterance_id"]] = pred

        processed += len(batch)
        if processed >= next_log:
            logger.info(f"Processed {processed} utterances")
            next_log += step

    logger.success("Transcription complete.")

    # Write submission file
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing submission file to {submission_path}")
    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["phonetic_text"] = predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    # Default: model lives next to main.py (used when packaged in submission zip)
    _script_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1])
    else:
        model_dir = _script_dir / "wav2vec2-phonetic-final"

    if len(sys.argv) > 2:
        data_manifest = Path(sys.argv[2])
    else:
        data_manifest = Path("data/utterance_metadata.jsonl")

    main(model_dir, data_manifest)
