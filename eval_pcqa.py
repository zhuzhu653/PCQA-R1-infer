#!/usr/bin/env python3
"""Minimal reviewer inference for PCQA-R1."""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit
from tqdm import tqdm
from transformers import AutoProcessor
try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None
from qwen_vl_utils import process_vision_info

def logistic_func(X, beta1, beta2, beta3, beta4):
    logistic_part = 1 + np.exp(np.negative(np.divide(X - beta3, np.maximum(np.abs(beta4), 1e-10))))
    return beta2 + np.divide(beta1 - beta2, logistic_part)


def logistic_fit(gt_scores, pred_scores):
    try:
        p0 = [np.max(gt_scores), np.min(gt_scores), np.mean(pred_scores), 0.5]
        popt, _ = curve_fit(logistic_func, pred_scores, gt_scores, p0=p0, maxfev=100000000)
        return logistic_func(pred_scores, *popt)
    except (RuntimeError, ValueError):
        return pred_scores

PCQA_PROMPT = (
    "You are doing the point cloud quality assessment task. Here is the question: "
    "You are a point cloud quality assessment expert. Given 6 rendered views of a 3D point cloud, "
    "rate its visual quality on a continuous scale from 1.00 (worst) to 5.00 (best)."
)

REASONING_TAG = "pcqa_reasoning"
_OPEN_TAG = f"<{REASONING_TAG}>"
_CLOSE_TAG = f"</{REASONING_TAG}>"

QUESTION_TEMPLATE = (
    "{Question} First output the PCQA reasoning process in "
    f"{_OPEN_TAG} {_CLOSE_TAG} tags "
    "and then output the final answer with only one score in <answer> </answer> tags. "
    f"Format: {_OPEN_TAG}your detailed analysis here{_CLOSE_TAG}<answer>X</answer>"
)
SPECIAL_TAIL_PATTERN = re.compile(r"(?:<\|(?:im_end|im_start|endoftext|pad)\|>|\s)+$")
LEADING_EMPTY_THINK_PATTERN = re.compile(r"^(?:\s*<think(?:ing)?>\s*</think(?:ing)?>\s*)+")
REASONING_BLOCK_PATTERN = re.compile(
    rf"{re.escape(_OPEN_TAG)}(.*?){re.escape(_CLOSE_TAG)}", re.DOTALL
)
FORMAT_PATTERN = re.compile(
    rf"^\s*{re.escape(_OPEN_TAG)}(?=[^<]*?[A-Za-z\u4e00-\u9fff])\s*[^<\s][^<]*?{re.escape(_CLOSE_TAG)}"
    r"\s*<answer>[^<]*?</answer>\s*\Z"
)


def strip_leading_empty_think(output_text):
    return LEADING_EMPTY_THINK_PATTERN.sub("", output_text or "")


def normalize_output_for_format(output_text):
    return strip_leading_empty_think(SPECIAL_TAIL_PATTERN.sub("", output_text or ""))


def is_empty_reasoning_output(output_text):
    reasoning_match = REASONING_BLOCK_PATTERN.search(output_text or "")
    return bool(reasoning_match and reasoning_match.group(1).strip() == "")

_SELF_DIR = str(Path(__file__).resolve().parent)
_DATA_BASE = os.environ.get("DATA_BASE", str(Path(__file__).resolve().parents[1] / "data_clip"))

DATASET_CONFIG = {
    "sjtu": {
        "label_dir": f"{_SELF_DIR}/datasets/SJTU-PCQA/test",
        "label_pattern": "SJTU-PCQA_test_fold{fold}_labels.txt",
        "image_dir": f"{_DATA_BASE}/SJTU-PCQA_maps/6view",
        "num_folds": 9,
    },
    "wpc": {
        "label_dir": f"{_SELF_DIR}/datasets/WPC/test",
        "label_pattern": "WPC_test_fold{fold}_labels.txt",
        "image_dir": f"{_DATA_BASE}/WPC_maps/6view",
        "num_folds": 5,
    },
}


def load_test_labels(label_file):
    labels = []
    with open(label_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if '\t' in stripped:
                parts = stripped.split('\t')
            else:
                parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                float(parts[1])
            except ValueError:
                continue
            ply_name = parts[0]
            mos = float(parts[1])
            if ply_name.endswith('.ply'):
                ply_stem = ply_name[:-4]
            else:
                ply_stem = ply_name
            labels.append((ply_stem, mos))
    return labels


def get_view_paths(ply_stem, image_dir):
    """Return the six color-rendered view paths for one point cloud sample."""
    return [os.path.join(image_dir, f"{ply_stem}_view_{view_id}.png") for view_id in range(6)]


def resolve_label_file(dataset_name, fold):
    """Resolve the label file for a dataset/fold pair."""
    config = DATASET_CONFIG[dataset_name]
    if fold == 0:
        full_name_map = {
            "sjtu": "SJTU-PCQA",
            "wpc": "WPC",
        }
        full_name = full_name_map.get(dataset_name, dataset_name.upper())
        return os.path.join(config["label_dir"], f"{full_name}_test_all_labels.txt")
    return os.path.join(config["label_dir"], config["label_pattern"].format(fold=fold))


def shard_samples(samples, shard_index, num_shards):
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards == 1:
        return samples
    return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_index]


def parse_score(output_text):
    try:
        matches = re.findall(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
        if matches:
            answer = matches[0].strip()
        else:
            answer = output_text.strip()
        m = re.search(r'[-+]?\d+\.\d+', answer) or re.search(r'[-+]?\d+', answer)
        if m is None:
            return None
        score = float(m.group())
        score = max(1.0, min(5.0, score))
        return score
    except Exception:
        return None


def inference_pcqa(samples, model, processor, device, batch_size=2):
    results = {}
    total_generation_seconds = 0.0
    total_generated_tokens = 0
    question = QUESTION_TEMPLATE.format(Question=PCQA_PROMPT)

    _is_qwen35 = (Qwen3_5ForConditionalGeneration is not None
                  and isinstance(model, Qwen3_5ForConditionalGeneration))
    _chat_kwargs = dict(tokenize=False, add_generation_prompt=True, add_vision_id=True)
    if _is_qwen35:
        _chat_kwargs['enable_thinking'] = False

    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch = samples[i:i + batch_size]

        batch_messages = []
        for ply_stem, view_paths in batch:
            content = []
            for vp in view_paths:
                content.append({"type": "image", "image": vp})
            content.append({"type": "text", "text": question})
            message = [{"role": "user", "content": content}]
            batch_messages.append(message)

        texts = [
            processor.apply_chat_template(msg, **_chat_kwargs)
            for msg in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_start_time = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, use_cache=True, max_new_tokens=512,
                do_sample=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_generation_seconds += time.perf_counter() - batch_start_time
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        total_generated_tokens += sum(int(out_ids.shape[0]) for out_ids in generated_ids_trimmed)
        batch_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        batch_output = [strip_leading_empty_think(output) for output in batch_output]

        batch_output_for_pattern = batch_output
        for idx, ((ply_stem, _), output) in enumerate(zip(batch, batch_output)):
            score = parse_score(output)
            parse_ok = score is not None
            if score is None:
                print(f"  [WARN] 无法解析分数: {ply_stem} → {output[:150]}")
                score = 3.0
            output_for_pattern = normalize_output_for_format(batch_output_for_pattern[idx])
            results[ply_stem] = {
                "score": score,
                "output_text": output,
                "completion_tokens": int(generated_ids_trimmed[idx].shape[0]),
                "valid_format": bool(FORMAT_PATTERN.search(output_for_pattern)),
                "empty_think": is_empty_reasoning_output(output_for_pattern),
                "parse_success": parse_ok,
            }

    throughput = {
        "generation_seconds": float(total_generation_seconds),
        "samples_per_second": float(len(samples) / total_generation_seconds) if total_generation_seconds > 0 else None,
        "tokens_per_second": float(total_generated_tokens / total_generation_seconds) if total_generation_seconds > 0 else None,
        "avg_generated_tokens": float(total_generated_tokens / len(samples)) if samples else None,
    }

    return results, throughput


def summarize_pcqa_predictions(dataset_name, fold, mode, labels, pred_dict, throughput=None, print_summary=True):
    """Compute PCQA metrics from a prediction dictionary keyed by ply_stem."""
    gt_scores = []
    pred_scores = []
    completion_lengths = []
    valid_format_flags = []
    empty_think_flags = []
    parse_success_flags = []
    per_sample = []
    for ply_stem, mos in labels:
        if ply_stem in pred_dict:
            pred_entry = pred_dict[ply_stem]
            gt_scores.append(mos)
            pred_scores.append(pred_entry["score"])
            completion_lengths.append(pred_entry.get("completion_tokens", 0))
            valid_format_flags.append(1.0 if pred_entry.get("valid_format", False) else 0.0)
            empty_think_flags.append(1.0 if pred_entry.get("empty_think", False) else 0.0)
            parse_success_flags.append(1.0 if pred_entry.get("parse_success", False) else 0.0)
            per_sample.append(
                {
                    "ply_stem": ply_stem,
                    "gt": float(mos),
                    "pred": float(pred_entry["score"]),
                    "completion_tokens": int(pred_entry.get("completion_tokens", 0)),
                    "valid_format": bool(pred_entry.get("valid_format", False)),
                    "empty_think": bool(pred_entry.get("empty_think", False)),
                    "parse_success": bool(pred_entry.get("parse_success", False)),
                }
            )

    if len(gt_scores) < 5:
        print(f"  [SKIP] 可用于指标计算的样本太少: {len(gt_scores)}")
        return None

    gt_scores = np.array(gt_scores)
    pred_scores = np.array(pred_scores)
    completion_lengths = np.array(completion_lengths)
    valid_format_flags = np.array(valid_format_flags)
    empty_think_flags = np.array(empty_think_flags)
    parse_success_flags = np.array(parse_success_flags)

    srcc, _ = spearmanr(gt_scores, pred_scores)
    krcc, _ = kendalltau(gt_scores, pred_scores)
    pred_fitted = logistic_fit(gt_scores, pred_scores)
    plcc, _ = pearsonr(gt_scores, pred_fitted)
    rmse = np.sqrt(np.mean((gt_scores - pred_fitted) ** 2))
    plcc_raw, _ = pearsonr(gt_scores, pred_scores)

    throughput = throughput or {}
    generation_seconds = throughput.get("generation_seconds")
    samples_per_second = throughput.get("samples_per_second")
    tokens_per_second = throughput.get("tokens_per_second")

    if print_summary:
        print(f"\n  SRCC = {srcc:.4f}")
        print(f"  PLCC = {plcc:.4f}  (logistic-fitted; raw={plcc_raw:.4f})")
        print(f"  KRCC = {krcc:.4f}")
        print(f"  RMSE = {rmse:.4f}  (logistic-fitted)")
        print(f"  Pred range: [{pred_scores.min():.2f}, {pred_scores.max():.2f}], mean={pred_scores.mean():.2f}")
        print(f"  GT range:   [{gt_scores.min():.2f}, {gt_scores.max():.2f}], mean={gt_scores.mean():.2f}")
        print(f"  Avg completion length = {completion_lengths.mean():.1f} tokens")
        print(f"  Valid format ratio = {valid_format_flags.mean():.3f}")
        print(f"  Parse success ratio = {parse_success_flags.mean():.3f}")
        print(f"  Empty-reasoning ratio = {empty_think_flags.mean():.3f}")
        if samples_per_second is not None:
            print(f"  Throughput = {samples_per_second:.2f} samples/s, {tokens_per_second:.2f} tokens/s")

    return {
        "dataset": dataset_name,
        "fold": fold,
        "mode": mode,
        "srcc": float(srcc),
        "plcc": float(plcc),
        "plcc_raw": float(plcc_raw),
        "krcc": float(krcc),
        "rmse": float(rmse),
        "n_valid": len(gt_scores),
        "n_total": len(labels),
        "pred_mean": float(pred_scores.mean()),
        "pred_std": float(pred_scores.std()),
        "avg_completion_length": float(completion_lengths.mean()),
        "median_completion_length": float(np.median(completion_lengths)),
        "valid_format_ratio": float(valid_format_flags.mean()),
        "parse_success_ratio": float(parse_success_flags.mean()),
        "empty_think_ratio": float(empty_think_flags.mean()),
        "reasoning_empty_ratio": float(empty_think_flags.mean()),
        "generation_seconds": generation_seconds,
        "samples_per_second": samples_per_second,
        "tokens_per_second": tokens_per_second,
        "per_sample": per_sample,
    }


def merge_shard_results(dataset_name, fold, mode, shard_paths):
    """Merge eval shard JSON files and recompute metrics on the full target set."""
    label_file = resolve_label_file(dataset_name, fold)
    if not os.path.exists(label_file):
        print(f"  [SKIP] 标签文件不存在: {label_file}")
        return None

    labels = load_test_labels(label_file)
    merged_pred = {}
    duplicate_keys = []
    shard_summaries = []
    shard_generation_seconds = []
    total_completion_tokens = 0

    print(f"\n{'='*60}")
    print(f"合并分片评测: {dataset_name} fold-{fold} ({mode})")
    print(f"  标签: {label_file}")
    print(f"  Shards: {len(shard_paths)}")
    print(f"{'='*60}")

    for shard_path in shard_paths:
        with open(shard_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        result = payload.get("result", payload)
        per_sample = result.get("per_sample", [])
        shard_summaries.append(
            {
                "path": shard_path,
                "n_valid": result.get("n_valid", len(per_sample)),
                "generation_seconds": result.get("generation_seconds"),
                "shard": result.get("shard"),
            }
        )
        if result.get("generation_seconds") is not None:
            shard_generation_seconds.append(float(result["generation_seconds"]))
        for item in per_sample:
            ply_stem = item["ply_stem"]
            if ply_stem in merged_pred:
                duplicate_keys.append(ply_stem)
                continue
            completion_tokens = int(item.get("completion_tokens", 0))
            total_completion_tokens += completion_tokens
            merged_pred[ply_stem] = {
                "score": float(item["pred"]),
                "completion_tokens": completion_tokens,
                "valid_format": bool(item.get("valid_format", False)),
                "empty_think": bool(item.get("empty_think", False)),
                "parse_success": bool(item.get("parse_success", False)),
            }

    if duplicate_keys:
        preview = ", ".join(sorted(set(duplicate_keys))[:5])
        raise ValueError(f"Shard outputs contain duplicated samples: {preview}")

    label_stems = [ply_stem for ply_stem, _ in labels]
    missing_after_merge = [ply_stem for ply_stem in label_stems if ply_stem not in merged_pred]
    if missing_after_merge:
        print(f"  [WARN] 合并后仍缺少 {len(missing_after_merge)} 个标签样本，示例: {missing_after_merge[:3]}")

    parallel_generation_seconds = max(shard_generation_seconds) if shard_generation_seconds else None
    serial_generation_seconds_sum = sum(shard_generation_seconds) if shard_generation_seconds else None
    throughput = {
        "generation_seconds": parallel_generation_seconds,
        "samples_per_second": (len(merged_pred) / parallel_generation_seconds) if parallel_generation_seconds else None,
        "tokens_per_second": (total_completion_tokens / parallel_generation_seconds) if parallel_generation_seconds else None,
    }
    merged_result = summarize_pcqa_predictions(
        dataset_name, fold, mode, labels, merged_pred, throughput=throughput, print_summary=True
    )
    if merged_result:
        merged_result["merged_from"] = shard_paths
        merged_result["shards"] = shard_summaries
        merged_result["n_shards"] = len(shard_paths)
        merged_result["n_missing_after_merge"] = len(missing_after_merge)
        merged_result["serial_generation_seconds_sum"] = serial_generation_seconds_sum
    return merged_result


def save_result_json(model_path, result, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_path, "result": result}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


def evaluate_pcqa(dataset_name, fold, mode, image_folder_override,
                  model, processor, device, batch_size=2,
                  shard_index=0, num_shards=1):
    """评估单个 PCQA 数据集/fold"""
    config = DATASET_CONFIG[dataset_name]

    # 确定标签文件
    label_file = resolve_label_file(dataset_name, fold)

    image_dir = image_folder_override or config["image_dir"]

    print(f"\n{'='*60}")
    print(f"评估: {dataset_name} fold-{fold} ({mode})")
    print(f"  标签: {label_file}")
    print(f"  图片: {image_dir}")
    print(f"{'='*60}")

    if not os.path.exists(label_file):
        print(f"  [SKIP] 标签文件不存在: {label_file}")
        return None

    labels = load_test_labels(label_file)
    print(f"  标签数量: {len(labels)}")

    # 构建样本列表
    samples = []
    skipped = 0
    for ply_stem, mos in labels:
        view_paths = get_view_paths(ply_stem, image_dir)
        # 检查所有视角图是否存在
        missing = [p for p in view_paths if not os.path.exists(p)]
        if missing:
            skipped += 1
            if skipped <= 3:
                print(f"  [WARN] 缺失图片: {ply_stem} ({len(missing)} files)")
            continue
        samples.append((ply_stem, view_paths))

    if skipped > 0:
        print(f"  [WARN] 跳过 {skipped} 个样本（图片缺失）")
    print(f"  有效样本: {len(samples)}")

    full_valid_samples = len(samples)
    if num_shards > 1:
        samples = shard_samples(samples, shard_index, num_shards)
        print(
            f"  分片: shard {shard_index + 1}/{num_shards}, "
            f"当前分片有效样本 {len(samples)} / {full_valid_samples}"
        )

    if len(samples) < 5:
        print(f"  [SKIP] 有效样本太少")
        return None

    # 推理
    pred_dict, throughput = inference_pcqa(samples, model, processor, device, batch_size)

    result = summarize_pcqa_predictions(dataset_name, fold, mode, labels, pred_dict, throughput=throughput)
    if result and num_shards > 1:
        result["shard"] = {
            "index": shard_index,
            "num_shards": num_shards,
            "n_full_valid_samples": full_valid_samples,
            "n_shard_samples": len(samples),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="PC_RL PCQA 评估（6-view 多图推理）")
    parser.add_argument("--model_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIG.keys()),
                        help="数据集名")
    parser.add_argument("--fold", type=int, default=1, help="fold 编号 (0=全集)")
    parser.add_argument("--mode", type=str, default="color", choices=["color"],
                        help="输入模式；匿名发布仅支持 6 张彩色渲染图")
    parser.add_argument("--image_folder", type=str, default=None, help="覆盖默认图片目录")
    parser.add_argument("--max_pixels", type=int, default=401408,
                        help="最大像素数，控制图像 resize (默认 401408≈632²，与训练脚本保持一致)")
    parser.add_argument("--batch_size", type=int, default=2, help="推理 batch size，默认固定为 2 以统一评估口径")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 路径")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="多卡分片评测：当前 shard 序号，范围 [0, num_shards)。默认 0")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="多卡分片评测：总 shard 数。默认 1 表示不分片")
    parser.add_argument("--merge_shards", type=str, default=None,
                        help="只合并 shard JSON 并重算指标，多个路径用 ':' 分隔；不会加载模型")
    args = parser.parse_args()

    if args.merge_shards:
        shard_paths = [p for p in args.merge_shards.split(":") if p]
        if not shard_paths:
            raise ValueError("--merge_shards was provided but no shard path was parsed")
        result = merge_shard_results(args.dataset, args.fold, args.mode, shard_paths)
        if result:
            ckpt_name = Path(args.model_path).name
            output_path = args.output or f"eval_pcqa_{args.dataset}_fold{args.fold}_{args.mode}_{ckpt_name}.json"
            save_result_json(args.model_path, result, output_path)
        return

    print(f"加载模型: {args.model_path}")
    device = torch.device(args.device)

    config_path = os.path.join(args.model_path, "config.json")
    model_cls = Qwen3_5ForConditionalGeneration
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        name_or_path = cfg.get("_name_or_path", "")
        if model_type != "qwen3_5" and "Qwen3.5" not in name_or_path:
            raise ValueError(f"Anonymous release only supports Qwen3.5 checkpoints, got model_type={model_type!r}")
    if model_cls is None:
        raise ImportError("Qwen3.5 requires transformers>=5.5.0")
    print(f"  检测到 Qwen3.5 模型")

    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    EVAL_MAX_PIXELS = args.max_pixels
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        ip.max_pixels = EVAL_MAX_PIXELS
        ip.min_pixels = 3136
        if hasattr(ip, 'size') and hasattr(ip.size, '__getitem__'):
            ip.size["longest_edge"] = EVAL_MAX_PIXELS
            ip.size["shortest_edge"] = 3136
        print(f"  image_processor.max_pixels = {EVAL_MAX_PIXELS}")
    print("模型加载完成")

    result = evaluate_pcqa(
        args.dataset, args.fold, args.mode,
        args.image_folder,
        model, processor, device, args.batch_size,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    if result:
        # 保存结果
        ckpt_name = Path(args.model_path).name
        output_path = args.output or f"eval_pcqa_{args.dataset}_fold{args.fold}_{args.mode}_{ckpt_name}.json"
        save_result_json(args.model_path, result, output_path)


if __name__ == "__main__":
    main()
