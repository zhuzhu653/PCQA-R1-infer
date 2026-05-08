# PCQA-R1-infer

Minimal reviewer package for deterministic six-view inference with the
anonymous PCQA-R1 checkpoint.

- Checkpoint: `checkpoint-210`
- Bundled label files: `WPC` and `SJTU-PCQA`
- Expected SJTU-PCQA fold-1 metrics: PLCC `0.9372`, SRCC `0.9525`
- Expected format sanity: `valid_format_ratio=1.0`, `parse_success_ratio=1.0`, `empty_think_ratio=0.0`

The checkpoint directory should contain the standard Hugging Face files,
including `config.json`, tokenizer/processor files, and `model.safetensors`.
If the checkpoint is distributed as split release assets, download all
`model.safetensors.part-*` files into the `checkpoint-210` directory and run:

```bash
cd checkpoint-210
sha256sum -c ../CHECKPOINT_210_SHA256SUMS_1GB.txt
cat model.safetensors.part-* > model.safetensors
```

Fold note:

- `SJTU-PCQA` uses 9 official folds.
- `WPC` uses 5 official folds.
- `--fold 0` is a convenience all-set label file bundled for reviewer evaluation; it is not an additional official fold.

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

Optional WPC case identifiers are listed in `examples/reviewer_cases_wpc.json`.
