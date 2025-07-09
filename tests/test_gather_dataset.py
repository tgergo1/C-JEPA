from bio_snn.datasets import gather_digits_dataset


def test_gather_digits_dataset(tmp_path):
    train_path, test_path = gather_digits_dataset(out_dir=tmp_path, test_size=0.2, random_state=0)
    assert train_path.exists()
    assert test_path.exists()
