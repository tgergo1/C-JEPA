import argparse
import numpy as np
from .predictive_coding import PredictiveCodingNetwork


def run_simulation(sizes, input_vec, steps, modulation=1.0):
    """Run a predictive coding network for a number of steps."""
    net = PredictiveCodingNetwork(sizes)
    x = np.array(input_vec, dtype=float)
    for _ in range(steps):
        out = net.forward(x, modulation=modulation)
    print("Output:", out)
    return out


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run a simple simulation with PredictiveCodingNetwork."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        required=True,
        help="Comma separated layer sizes, e.g. 2,3,1",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Comma separated input vector",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of forward steps to run",
    )
    parser.add_argument(
        "--modulation",
        type=float,
        default=1.0,
        help="Neuromodulation factor",
    )
    parsed = parser.parse_args(args)

    sizes = [int(s) for s in parsed.sizes.split(",")]
    input_vec = [float(v) for v in parsed.input.split(",")]
    run_simulation(sizes, input_vec, parsed.steps, modulation=parsed.modulation)


if __name__ == "__main__":
    main()
