"""dense_control stages on Barça-like smoothed balls (common/smooth_ball.py), into a separate --out.

    ../run.sh approaches/dense_control/smooth.py build --out ../results/dense_smooth
    python approaches/dense_control/smooth.py train|predict|calibrate|evaluate --out ../results/dense_smooth ...

Same code, architecture, schedule and seeds as results/dense; only the simulated ball changes.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import tuning_data  # noqa: E402

if sys.argv[1:2] == ["build"]:
    import functools, os
    from common.smooth_ball import degrade_smooth
    # dense.build imports degrade at call time; XYZ_SIGMA/XYZ_NOISE/XYZ_PSIGMA/XYZ_SWITCH/XYZ_BALL_NOISE only for robustness checks (8 / 0 / 0 / 0 = deliverable)
    tuning_data.degrade = functools.partial(degrade_smooth, sigma=float(os.environ.get("XYZ_SIGMA", 8)),
                                           noise=float(os.environ.get("XYZ_NOISE", 0)),
                                           psigma=float(os.environ.get("XYZ_PSIGMA", 0)),
                                           rate=float(os.environ.get("XYZ_SWITCH", 0)),
                                           params=dict(ball_noise_m=float(os.environ["XYZ_BALL_NOISE"])) if "XYZ_BALL_NOISE" in os.environ else None)

from common import dense_control as dense  # noqa: E402

if __name__ == "__main__":
    dense.main()
