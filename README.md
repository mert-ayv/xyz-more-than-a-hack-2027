# Team Xyz: possession from tracking (More Than A Hack 2027)

Dense per-frame model on tracking only, trained on public matches whose ball track is smoothed to look like the
provided match, with an auxiliary ball-contact head; three variants (3,000 steps, 6,000 steps, width 128) averaged. Method, validation and limits: [TECHNICAL.md](TECHNICAL.md).

## Reproduce the CSV (CPU, about a minute)

Python 3.11 with `pip install -r requirements-reproduce.txt` (numpy, pandas, scipy, Torch 2.8 CPU). `MTAH_DATA_DIR` holds the supplied
`tactical/BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml`. The data is not in this repository: it is the organisers' gated bundle, see `DATA.md` in [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027).

```bash
export MTAH_DATA_DIR=/absolute/path/to/data
for m in long steps2 w128; do
  mkdir -p work/$m dense/$m/final && cp weights/$m/calibration.json work/$m/ && cp weights/$m/model.pt dense/$m/final/
  python approaches/dense_control/contact.py run approaches/dense_control/smooth_target.py --dense dense/$m --work work/$m --target work/target --csv work/$m.csv --weight 1
done
python approaches/dense_control/ens_target.py --works work/long work/steps2 work/w128 --target work/target --csv work/Xyz.csv
python evaluation/validate_submission.py work/Xyz.csv
```

Expected SHA256 `41ef82f3feebce0a2244b561f5afe16fbece9e11ec61f22a305b0ac8c41f7898`.

## Retrain the weights (GPU)

Public Metrica sample games 1–2 and five DFL matches (`MTAH_TUNING_DIR`, `MTAH_DFL_DIR`):
`approaches/dense_control/smooth.py build`, `approaches/dense_control/contact.py targets` (contact targets from the
clean DFL ball), `contact.py train|predict|calibrate` for the seven leave-one-game-out folds (3,000 steps in plan.json), then
`echo 3000 > <run>/contact/STEPS` and `contact.py final freeze|train` for the final model; repeat with `STEPS` 6000, and with `STEPS` 3000 plus `<run>/contact/WIDTH` 128. Retraining is not claimed bit-identical.
