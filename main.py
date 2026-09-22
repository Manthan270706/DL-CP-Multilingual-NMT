"""Main CLI Entrypoint for Multilingual NMT System.
Master Decision Register & Spec Section 2.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.data.adapters import DatasetAdapter
from src.data.dataset_manager import DatasetManager
from src.inference.engine import TranslationInferenceEngine
from src.registry.language_registry import LanguageRegistry
from src.training.trainer import NMTTrainer
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.utils.system_info import get_system_environment

logger = setup_logger("main")


def cmd_env_check(args):
    """Verify and print environment and GPU status."""
    env = get_system_environment()
    print("\n" + "=" * 60)
    print("MULTILINGUAL NMT - ENVIRONMENT & GPU AUDIT")
    print("=" * 60)
    print(f"Python Version : {env['python_version']}")
    print(f"Platform       : {env['platform']}")
    print(f"PyTorch        : {env.get('torch_version')}")
    print(f"CUDA Available : {env['cuda_available']}")
    if env["cuda_available"]:
        print(f"CUDA Version   : {env.get('cuda_version')}")
        for dev in env["devices"]:
            print(f"Device {dev['id']}       : {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
    print("\nPackage Versions:")
    for k, v in sorted(env["packages"].items()):
        print(f"  {k:<15}: {v}")
    print("=" * 60 + "\n")


def cmd_prepare_data(args):
    """Prepare, clean, and split datasets for specified or all directions."""
    config_mgr = ConfigManager()
    cfg_datasets = config_mgr.get_datasets_config()
    manager = DatasetManager()

    directions = (
        [f"{args.src}-{args.tgt}"]
        if args.src and args.tgt
        else LanguageRegistry.get_supported_directions()
    )

    for direction in directions:
        src, tgt = direction.split("-")
        logger.info(f"Preparing dataset for direction '{src}->{tgt}'...")

        # Create or load dataset
        if args.synthetic:
            raw_df = DatasetAdapter.create_synthetic_verification_corpus(
                src, tgt, n_samples=args.num_samples or 100
            )
            ds_name = "synthetic_verified"
            provenance = {"synthetic": True}
        else:
            raw_df, provenance = DatasetAdapter.load_for_direction_with_metadata(
                src_lang=src,
                tgt_lang=tgt,
                max_samples=args.num_samples,
            )
            ds_name = provenance.get("dataset_name", f"canonical_{src}_{tgt}")

        train_df, val_df, test_df, manifest = manager.prepare_dataset_pipeline(
            raw_df=raw_df,
            src_lang=src,
            tgt_lang=tgt,
            dataset_name=ds_name,
            provenance_metadata=provenance,
        )
        print(
            f"Prepared {direction}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} | Total Cleaned={manifest['split_counts']['total_cleaned']}"
        )


def cmd_dataset_audit(args):
    """Execute complete real-world dataset acquisition, dual leakage validation, and freeze manifest generation."""
    print("\n" + "=" * 80)
    print("MULTILINGUAL NMT — CANONICAL REAL-WORLD DATASET AUDIT & FREEZE")
    print("=" * 80 + "\n")

    manager = DatasetManager()
    audit_records = []

    for direction in LanguageRegistry.get_supported_directions():
        src, tgt = direction.split("-")
        logger.info(f"Auditing canonical real-world corpus for {direction.upper()}...")

        raw_df, provenance = DatasetAdapter.load_for_direction_with_metadata(
            src_lang=src,
            tgt_lang=tgt,
            max_samples=args.num_samples,
        )

        train_df, val_df, test_df, manifest = manager.prepare_dataset_pipeline(
            raw_df=raw_df,
            src_lang=src,
            tgt_lang=tgt,
            dataset_name=provenance["dataset_name"],
            provenance_metadata=provenance,
        )

        audit_records.append({
            "Direction": direction.upper(),
            "Dataset Name": provenance["dataset_name"],
            "Config": provenance.get("configuration", "default"),
            "Provider": provenance.get("provider", "N/A"),
            "License": provenance.get("license", "N/A"),
            "Original Size": provenance.get("original_size", len(raw_df)),
            "Cleaned Total": manifest["split_counts"]["total_cleaned"],
            "Train Size": len(train_df),
            "Val Size": len(val_df),
            "Test Size": len(test_df),
            "Pair Leakage": manifest["leakage_audit"]["pair_overlaps"],
            "Source Leakage": manifest["leakage_audit"]["source_overlaps"],
            "Freeze Manifest": manifest["paths"]["test"].replace("test.csv", "freeze_manifest.json"),
        })

    print("\n" + "=" * 80)
    print("DATASET AUDIT & FREEZE SUMMARY TABLE")
    print("=" * 80)
    df_audit = pd.DataFrame(audit_records)
    print(df_audit[["Direction", "Dataset Name", "Config", "Original Size", "Cleaned Total", "Train Size", "Val Size", "Test Size"]].to_string(index=False))
    print("=" * 80 + "\n")


def cmd_train(args):
    """Fine-tune a model for a specific language pair."""
    config_mgr = ConfigManager()
    full_cfg = config_mgr.get_full_config()
    train_cfg = full_cfg["training"]["training"]
    eval_cfg = full_cfg["evaluation"]

    # Merge generation settings into train_cfg
    train_cfg.update(eval_cfg["generation"])
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs
    if args.batch_size:
        train_cfg["per_device_train_batch_size"] = args.batch_size

    manager = DatasetManager()
    train_df, val_df, test_df = manager.load_processed_splits(args.src, args.tgt)

    exp_id = args.exp_id or f"{args.model.upper()}_{args.src.upper()}_{args.tgt.upper()}_RUN001"
    trainer = NMTTrainer(
        model_id=args.model,
        src_lang=args.src,
        tgt_lang=args.tgt,
        config=train_cfg,
        experiment_id=exp_id,
    )

    trainer.run_training(train_df, val_df, test_df)


def cmd_translate(args):
    """Translate text from CLI."""
    engine = TranslationInferenceEngine()
    res = engine.translate_single(
        text=args.text,
        src_lang=args.src,
        tgt_lang=args.tgt,
        model_id=args.model,
        num_beams=args.beams or 5,
    )
    print("\n" + "=" * 50)
    print(f"Source ({args.src}) : {args.text}")
    print(f"Target ({args.tgt}) : {res['translated_text']}")
    print(f"Model       : {res['model_id']}")
    print(f"Latency     : {res['latency_ms']} ms")
    print("=" * 50 + "\n")


def cmd_app(args):
    """Launch Streamlit web application."""
    import subprocess

    app_path = ROOT_DIR / "app" / "streamlit_app.py"
    port = args.port or 8501
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    logger.info(f"Launching Streamlit application on port {port}...")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Multilingual Neural Machine Translation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # env-check
    p_env = subparsers.add_parser("env-check", help="Check Python, PyTorch, CUDA, and dependencies")
    p_env.set_defaults(func=cmd_env_check)

    # dataset-audit
    p_audit = subparsers.add_parser(
        "dataset-audit",
        help="Audit, clean, validate dual zero-leakage, and freeze canonical real datasets",
    )
    p_audit.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional max samples limit (None = full canonical corpus)",
    )
    p_audit.set_defaults(func=cmd_dataset_audit)

    # prepare-data
    p_data = subparsers.add_parser("prepare-data", help="Acquire, clean, and split datasets")
    p_data.add_argument("--src", type=str, default="en", help="Source language (default: en)")
    p_data.add_argument("--tgt", type=str, help="Target language (hi, mr, fr, de)")
    p_data.add_argument(
        "--synthetic", action="store_true", help="Use verified local synthetic corpus"
    )
    p_data.add_argument("--num-samples", type=int, default=200, help="Number of samples")
    p_data.set_defaults(func=cmd_prepare_data)

    # train
    p_train = subparsers.add_parser("train", help="Run fine-tuning experiment")
    p_train.add_argument(
        "--model", type=str, required=True, help="Model ID (e.g. nllb200, mbart50, marian_en_hi)"
    )
    p_train.add_argument("--src", type=str, default="en", help="Source language")
    p_train.add_argument("--tgt", type=str, required=True, help="Target language")
    p_train.add_argument("--exp-id", type=str, help="Unique experiment ID")
    p_train.add_argument("--epochs", type=int, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, help="Batch size per device")
    p_train.set_defaults(func=cmd_train)

    # translate
    p_trans = subparsers.add_parser("translate", help="Translate text using inference engine")
    p_trans.add_argument("--text", type=str, required=True, help="Text to translate")
    p_trans.add_argument("--src", type=str, default="en", help="Source language")
    p_trans.add_argument("--tgt", type=str, required=True, help="Target language")
    p_trans.add_argument("--model", type=str, default="nllb200", help="Model ID")
    p_trans.add_argument("--beams", type=int, default=5, help="Beam width")
    p_trans.set_defaults(func=cmd_translate)

    # app
    p_app = subparsers.add_parser("app", help="Launch Streamlit translation interface")
    p_app.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    p_app.set_defaults(func=cmd_app)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
