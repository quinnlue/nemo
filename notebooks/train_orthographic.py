import json
import os

import lightning.pytorch as pl
import torch
import wandb
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import open_dict
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from asr_benchmark.nemo_adapter import (
    add_global_adapter_cfg,
    patch_transcribe_lhotse,
    update_model_cfg,
    update_model_config_to_support_adapter,
)
from asr_benchmark.score import english_spelling_normalizer, score_wer
from orthographic_config import BATCH_SIZE, SAMPLE, cfg

torch.set_float32_matmul_precision("high")

# ── Trainer & experiment manager ─────────────────────────────────────────────
trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

# ── Load pretrained model ────────────────────────────────────────────────────
model_cfg = ASRModel.from_pretrained(cfg.model.pretrained_model, return_config=True)
update_model_config_to_support_adapter(model_cfg, cfg)
model = ASRModel.from_pretrained(
    cfg.model.pretrained_model,
    override_config_path=model_cfg,
    trainer=trainer,
)

with open_dict(model.cfg):
    model.cfg.decoding.greedy.use_cuda_graph_decoder = False
model.change_decoding_strategy(model.cfg.decoding)

# ── Data loaders ─────────────────────────────────────────────────────────────
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)

cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
model.setup_multiple_validation_data(cfg.model.validation_ds)

# ── Optimizer ────────────────────────────────────────────────────────────────
model.setup_optimization(cfg.model.optim)

# ── Spec augmentation ────────────────────────────────────────────────────────
if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
else:
    model.spec_augmentation = None
    del model.cfg.spec_augment

# ── Adapter setup ────────────────────────────────────────────────────────────
with open_dict(cfg.model.adapter):
    adapter_name = cfg.model.adapter.pop("adapter_name")
    adapter_type = cfg.model.adapter.pop("adapter_type")
    adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
    adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)

    adapter_type_cfg = cfg.model.adapter[adapter_type]

    if adapter_module_name is not None and ":" not in adapter_name:
        adapter_name = f"{adapter_module_name}:{adapter_name}"

    adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
    if adapter_global_cfg is not None:
        add_global_adapter_cfg(model, adapter_global_cfg)

model.add_adapter(adapter_name, cfg=adapter_type_cfg)
assert model.is_adapter_available()

model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(adapter_name, enabled=True)

model.freeze()
model = model.train()
model.unfreeze_enabled_adapters()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# ── Train ────────────────────────────────────────────────────────────────────
trainer.fit(model)
wandb.finish()

# ── Save adapter state dict ──────────────────────────────────────────────────
if adapter_state_dict_name is not None:
    state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
    ckpt_path = os.path.join(state_path, "checkpoints")
    if os.path.exists(ckpt_path):
        state_path = ckpt_path
    state_path = os.path.join(state_path, adapter_state_dict_name)

    model.save_adapters(str(state_path))

# ── Evaluate best checkpoint ────────────────────────────────────────────────
nemo_ckpts = sorted((exp_log_dir / "checkpoints").glob("*.nemo"))
if not nemo_ckpts:
    raise FileNotFoundError(f"No .nemo checkpoints found in {exp_log_dir}/checkpoints/")

best_ckpt = nemo_ckpts[-1]
print(f"Loading checkpoint: {best_ckpt}")
eval_model = ASRModel.restore_from(best_ckpt, map_location="cuda")

with open_dict(eval_model.cfg):
    eval_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
eval_model.change_decoding_strategy(eval_model.cfg.decoding)

patch_transcribe_lhotse(eval_model)

# ── Run inference on validation set ──────────────────────────────────────────
val_manifest_path = cfg.model.validation_ds.manifest_filepath
with open(val_manifest_path) as f:
    val_entries = [json.loads(line) for line in f]

audio_files = [e["audio_filepath"] for e in val_entries]
references = [e["text"] for e in val_entries]

print(f"Running inference on {len(audio_files)} validation utterances...")
raw = eval_model.transcribe(
    audio_files, batch_size=BATCH_SIZE, channel_selector="average", verbose=False
)
if isinstance(raw, tuple):
    raw = raw[0]

predictions = [h.text if hasattr(h, "text") else h for h in raw]

# ── Score ────────────────────────────────────────────────────────────────────
normalizer = EnglishTextNormalizer(english_spelling_normalizer)
filtered = [(r, p) for r, p in zip(references, predictions) if normalizer(r) != ""]

references, predictions = zip(*filtered)

wer = score_wer(references, predictions)

print(f"Validation WER: {wer:.4f}")

print("\nSample predictions:")
for ref, pred in zip(references[:5], predictions[:5]):
    print(f"  REF:  {ref}")
    print(f"  PRED: {pred}")
    print()
