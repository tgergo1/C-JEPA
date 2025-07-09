from bio_snn.training import EnergyTrainer, TrainingConfig


def test_batch_iter_produces_all_samples(tmp_path):
    cfg = TrainingConfig(
        sizes=[64, 16, 10], epochs=0, batch_size=20, checkpoint_dir=tmp_path
    )
    trainer = EnergyTrainer(cfg)
    count = 0
    for x_batch, y_batch in trainer._batch_iter():
        assert x_batch.shape[0] <= 20
        assert x_batch.ndim == 2
        assert y_batch.ndim == 1
        count += len(x_batch)
    assert count == len(trainer.X_train)


def test_training_creates_plot(tmp_path):
    cfg = TrainingConfig(sizes=[64, 16, 10], epochs=1, checkpoint_dir=tmp_path, lr=0.05)
    trainer = EnergyTrainer(cfg)
    trainer.train()
    assert any(tmp_path.glob("training_loss.png"))
