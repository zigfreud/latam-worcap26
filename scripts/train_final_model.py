"""Reajusta o componente principal com dados oficiais locais."""
import argparse
from pathlib import Path

from worcap.delivery import run_pressure_final

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
run_pressure_final(a.output)
