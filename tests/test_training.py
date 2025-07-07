from bio_snn.training import EnergyTrainer, TrainingConfig


def test_energy_trainer_runs(tmp_path):
    cfg = TrainingConfig(sizes=[64, 16, 10], epochs=1, checkpoint_dir=tmp_path, lr=0.05)
    trainer = EnergyTrainer(cfg)
    trainer.train()
    # check that a checkpoint file was created
    assert any(tmp_path.glob("epoch_1.npz"))
