"""Run local public-fruit regression in four independently released stages."""
import argparse
import json
from pathlib import Path
from herb_spectra.public_fruit_regression import prepare, selection, evaluation, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'selection', 'evaluation', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.stage in ('selection', 'evaluation') and args.release is None:
        parser.error('A root stage release JSON is required')
    stages = {'prepare': lambda: prepare(args.output),
              'selection': lambda: selection(args.output, args.release),
              'evaluation': lambda: evaluation(args.output, args.release),
              'score': lambda: score(args.output)}
    print(json.dumps(stages[args.stage](), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
