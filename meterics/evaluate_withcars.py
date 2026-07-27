"""Evaluate dynamic radio-map reconstruction predictions with vehicles."""

import argparse

try:
    from ._common import add_common_arguments, evaluate, print_and_save, resolve_gt_dir
except ImportError:
    from _common import add_common_arguments, evaluate, print_and_save, resolve_gt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PA-DCNet DRM predictions.")
    add_common_arguments(parser, default_target="carsDPM")
    args = parser.parse_args()
    summary, count, skipped = evaluate(args.pred_dir, resolve_gt_dir(args), limit=args.limit)
    print_and_save(summary, count, skipped, args.output)


if __name__ == "__main__":
    main()
