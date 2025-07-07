import subprocess
import sys


def test_cli_simulation_runs():
    result = subprocess.run(
        [sys.executable, '-m', 'bio_snn.interface', '--sizes', '2,3,1', '--input', '1,0', '--steps', '5'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'Output:' in result.stdout


def test_cli_seed_reproducible():
    args = [sys.executable, '-m', 'bio_snn.interface', '--sizes', '2,3,1', '--input', '1,0', '--steps', '5', '--seed', '42']
    r1 = subprocess.run(args, capture_output=True, text=True).stdout
    r2 = subprocess.run(args, capture_output=True, text=True).stdout
    assert r1 == r2
