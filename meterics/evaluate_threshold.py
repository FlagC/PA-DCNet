"""Evaluate threshold-transformed static radio-map predictions."""

import argparse

try:
    from ._common import add_common_arguments, evaluate, print_and_save, resolve_gt_dir
except ImportError:
    from _common import add_common_arguments, evaluate, print_and_save, resolve_gt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate thresholded PA-DCNet SRM predictions.")
    add_common_arguments(parser, default_target="DPM")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--apply-to-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-to-gt", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary, count, skipped = evaluate(
        args.pred_dir,
        resolve_gt_dir(args),
        limit=args.limit,
        threshold=args.threshold,
        threshold_prediction=args.apply_to_pred,
        threshold_target=args.apply_to_gt,
    )
    print_and_save(summary, count, skipped, args.output)


if __name__ == "__main__":
    main()
