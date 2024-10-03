import pyzx as zx
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.util import tracefunc

sys.setprofile(tracefunc)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--qubit", type=int, default=20)
    argparse.add_argument("--gates", type=int, default=100)
    args = argparse.parse_args()

    g_circ = zx.generate.cliffordT(args.qubit, args.gates)
    g_circ.to_tensor()