import argparse
import shutil
from datetime import datetime
from pathlib import Path


KEEP_NAMES = {
    "src",
    "tests",
    "requirements.txt",
    "README.md",
    "PROJECT_TUTORIAL.md",
    "run_train.py",
    "run_predict.py",
    "run_compare_channels.py",
    "run_digit_pipeline.py",
    "run_health_scan.py",
    "run_visualize.py",
    "run_report.py",
    "run_freeze.py",
    "lexicon_commands.txt",
    "artifacts_delivery_final",
    "predictions_delivery_final_lex",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze project by keeping only delivery-relevant files.")
    parser.add_argument("--root_dir", type=str, default=r"d:\\huxi", help="Workspace root")
    parser.add_argument("--project_dir", type=str, default=r"d:\\huxi\\morse_train", help="Project directory")
    parser.add_argument(
        "--trash_dir",
        type=str,
        default=None,
        help="Trash directory, default: <root>/_freeze_trash_YYYYMMDD",
    )
    parser.add_argument("--dry_run", action="store_true", help="Show what would be moved (default mode)")
    parser.add_argument("--apply", action="store_true", help="Apply move-to-trash cleanup")
    return parser.parse_args()


def _default_trash_dir(root_dir: Path) -> Path:
    return root_dir / f"_freeze_trash_{datetime.now().strftime('%Y%m%d')}"


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    project_dir = Path(args.project_dir)
    trash_dir = Path(args.trash_dir) if args.trash_dir else _default_trash_dir(root_dir)

    if not project_dir.exists():
        raise FileNotFoundError(f"Project dir not found: {project_dir}")

    apply_mode = bool(args.apply)
    dry_run = True if not apply_mode else False
    if args.dry_run and not args.apply:
        dry_run = True

    candidates = []
    for p in sorted(project_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.name in KEEP_NAMES:
            continue
        candidates.append(p)

    print("mode:", "DRY_RUN" if dry_run else "APPLY")
    print("project_dir:", project_dir)
    print("trash_dir:", trash_dir)
    print("keep_count:", len(KEEP_NAMES))
    print("move_count:", len(candidates))
    for p in candidates:
        print(" -", p)

    if dry_run:
        return

    trash_project = trash_dir / project_dir.name
    trash_project.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in candidates:
        dst = trash_project / src.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        moved += 1

    print("moved:", moved)
    print("done")


if __name__ == "__main__":
    main()
