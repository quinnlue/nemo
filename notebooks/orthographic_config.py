"""
git clone https://github.com/quinnlue/asr.git
cd asr
sudo apt update
sudo apt install just
just create-environment
source ./.venv/bin/activate
just requirements
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=asr
uv run --with jupyter jupyter lab
Available types: ['speed', 'time_stretch', 'gain', 'silence', 'impulse', 'shift', 'noise', 'noise_norm', 'white_noise', 'rir_noise_aug', 'transcode_aug', 'random_segment']
"""

from omegaconf import OmegaConf

from asr_benchmark.config import PROJECT_ROOT
from asr_benchmark.manifest import prepare_manifests

# ── Sampling ─────────────────────────────────────────────────────────────────
# Set SAMPLE to use a smaller subset of the data for faster iteration during
# development.  Set it to None to use the full dataset.
SAMPLE = None

# ── Hardware-dependent settings ──────────────────────────────────────────────
DEVICES = 1
PRECISION = "bf16-mixed"
BATCH_SIZE = 32
NUM_WORKERS = 8

# ── Manifests ────────────────────────────────────────────────────────────────
manifests = prepare_manifests(sample_n=SAMPLE)
train_manifest_path = manifests["train"]
val_manifest_path = manifests["val"]
noise_manifest_path = manifests["noise"]
impulse_manifest_path = manifests["impulse"]

# ── Load NeMo adapter defaults ───────────────────────────────────────────────
yaml_path = PROJECT_ROOT / "asr_benchmark" / "assets" / "asr_adaptation.yaml"
cfg = OmegaConf.load(yaml_path)

# ── Training overrides ───────────────────────────────────────────────────────
overrides = OmegaConf.create(
    {
        "model": {
            "pretrained_model": "nvidia/parakeet-tdt-0.6b-v3",
            "adapter": {
                "adapter_name": "asr_children_orthographic",
                "adapter_module_name": "encoder",
                "linear": {"in_features": 1024},
            },
            "train_ds": {
                "manifest_filepath": str(train_manifest_path),
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "use_lhotse": False,
                "channel_selector": "average",
                "augmentor": {
                    "time_stretch": {
                        "min_speed_rate": 0.9,
                        "max_speed_rate": 1.1,
                        "prob": 0.5,
                        "num_rates": 5,
                        "rng": 42,
                    },
                    "noise": {
                        "manifest_path": str(noise_manifest_path),
                        "orig_sr": 16000,
                        "min_snr_db": 10.0,
                        "max_snr_db": 50.0,
                        "rng": 42,
                    },
                    "impulse": {
                        "manifest_path": str(impulse_manifest_path),
                        "rng": 42,
                    },
                },
            },
            "validation_ds": {
                "manifest_filepath": str(val_manifest_path),
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "use_lhotse": False,
                "channel_selector": "average",
            },
            "optim": {
                "lr": 0.001,
                "weight_decay": 0.0,
            },
        },
        "trainer": {
            "devices": DEVICES,
            "precision": PRECISION,
            "strategy": "auto",
            "max_epochs": 1 if SAMPLE else None,
            "max_steps": -1 if SAMPLE else 5000,
            "val_check_interval": 1.0 if SAMPLE else 500,
            "enable_progress_bar": False,
        },
        "exp_manager": {
            "exp_dir": str(PROJECT_ROOT / "models" / "orthographic_benchmark_nemo"),
            "create_wandb_logger": True,
            "wandb_logger_kwargs": {
                "project": "asr-ctc",
                "name": "orthographic-adapter",
                "tags": ["orthographic", "adapter", "parakeet-tdt-0.6b-v3"],
            },
        },
    }
)

cfg = OmegaConf.merge(cfg, overrides)
