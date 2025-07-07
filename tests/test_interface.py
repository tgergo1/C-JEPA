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
