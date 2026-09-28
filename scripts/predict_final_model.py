"""Reproduz a previsão mensal a partir dos artefatos de um ajuste final."""
import argparse
import json
from pathlib import Path

from worcap.delivery import infer_pressure
from worcap.experiment import require, sha256
from worcap.finalize import export_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads((args.model_dir/'recipe.json').read_text(encoding='utf-8'))
    require(sha256(args.sample) == recipe['sample_sha256'], 'Sample diferente da receita')
    require(not args.output.exists(), 'Arquivo de saída já existe')
    predicted = infer_pressure(args.model_dir, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    verification = export_csv(predicted, args.sample, args.output)
    original = args.model_dir/'verification.json'
    if original.exists():
        expected = json.loads(original.read_text(encoding='utf-8'))
        require(verification['sha256'] == expected['sha256'], 'CSV difere do ajuste original')
    args.output.with_suffix('.verification.json').write_text(
        json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
