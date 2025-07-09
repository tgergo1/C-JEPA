import argparse
from bio_snn.datasets.digits import gather_digits_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download digits dataset and save train/test splits"
    )
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Fraction reserved for testing")
    parser.add_argument("--random-state", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    train_path, test_path = gather_digits_dataset(
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Saved training data to {train_path}")
    print(f"Saved testing data to {test_path}")


if __name__ == "__main__":
    main()
