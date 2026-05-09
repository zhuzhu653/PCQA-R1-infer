# PCQA-R1-infer

Minimal reviewer package for deterministic six-view inference with the
anonymous PCQA-R1 checkpoint.

- Checkpoint: `checkpoint-210`
- Bundled label files: `WPC` and `SJTU-PCQA`
- Expected SJTU-PCQA fold-1 metrics: PLCC `0.9372`, SRCC `0.9525`
- Expected format sanity: `valid_format_ratio=1.0`, `parse_success_ratio=1.0`, `empty_think_ratio=0.0`

The checkpoint directory should contain the standard Hugging Face files,
including `config.json`, tokenizer/processor files, and `model.safetensors`.
Download the reviewer checkpoint from Hugging Face:

```text
https://huggingface.co/jiujiu666/PCQA-R1-sjtu
```

Place the downloaded files in the `checkpoint-210` directory, or point
`--model_path` directly to the local Hugging Face snapshot directory.

Environment:

- Tested on Linux with an NVIDIA GPU and CUDA-enabled PyTorch.
- Python `3.10` is recommended.
- `eval_pcqa.py` expects a Qwen3.5-capable `transformers` build and uses `flash_attention_2` during model loading.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Minimum tested packages:

- `torch` with CUDA support
- `transformers>=5.5.0`
- `qwen-vl-utils`
- `numpy`
- `scipy`
- `tqdm`

Fold note:

- `SJTU-PCQA` uses the public 9-fold split protocol bundled in this package.
- `WPC` uses the public 5-fold split protocol bundled in this package.
- `--fold 0` is a convenience all-set label file bundled for reviewer evaluation; it is not an additional benchmark split.

Required input layout:

```text
DATA_BASE/
└── SJTU-PCQA_maps/
    └── 6view/
        ├── <point_cloud_stem>_view_0.png
        ├── <point_cloud_stem>_view_1.png
        ├── ...
        └── <point_cloud_stem>_view_5.png
```

Run:

```bash
export DATA_BASE=/path/to/projection_root
python eval_pcqa.py \
	--model_path /path/to/anonymous/checkpoint-210 \
	--dataset sjtu \
	--fold 1 \
	--image_folder "$DATA_BASE/SJTU-PCQA_maps/6view" \
	--output eval_sjtu_fold1_color_checkpoint-210.json
```
