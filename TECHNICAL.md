# Team Xyz — possession from tracking

> Technical document for the Initial Challenge (deliverable 3, SUBMISSION.md). Numbers are development-game results; none come from the provided match's labels.

## Approach

One neural sequence model reads the whole match frame by frame and a decoder turns its per-frame evidence into possession spells.

- **Input per frame (25 Hz):** ball position/velocity/acceleration/turn, ball-seen flags, every tracked player relative to the ball (distance, closing speed, short-horizon path and intercept features), nearest-player and count features per team, and restart hints from the automatic smart tagging.
- **Model:** a shared per-player encoder pooled by learned attention per team, then a 7-block dilated 1-D convolution over time (window 1,024 frames, ±127-frame halo, receptive field ≈ ±5 s). Two heads: owner (H/A/D) and six directed transitions (H→A, A→H, live↔dead). A third, training-only head predicts ball contacts per team (below); it is not used at inference.
- **Inference:** four pitch-flip views averaged (left/right, top/bottom, teams swapped accordingly).
- **Three models averaged:** the same recipe trained three ways (3,000 steps width 96; 6,000 steps; width 128). Their owner outputs and their own calibrated transition evidence are averaged before decoding.
- **Decoder:** Viterbi over H/A/D per half. Switching H↔A costs 8, live↔dead costs 8 (log units), the calibrated transition head adds evidence for a switch at the frame it fires. No minimum possession length.

**Why this and not something else.** The provided match's tracking differs from clean public tracking in two ways that matter: player tracks fragment across 46 slots and the ball is missing on ~41% of graded frames; and the ball track is strongly low-passed (median acceleration 0.04 vs 0.76 on our development games, turn cosine 0.97 vs 0.70). A model trained on clean ball paths learns from jitter that never appears in the target. So training games are first degraded to the target's tracking statistics (slot fragmentation, ball gaps, noise, dropped/jittered restart tags) and then the ball is smoothed with a σ = 8-frame Gaussian, the value that best matches the target's unlabelled ball statistics among σ 0–16. Retraining on that input moved the leave-one-game-out score on Barça-like development inputs from 0.607 to 0.675.

**Contact head.** Smoothing hides the kicks the model used to see. During training only, an extra head learns where contacts happen, from the clean (unsmoothed) DFL ball: a local acceleration peak ≥ 40 m/s² within 1.5 m of a player marks a contact for that player's team (±3 frames, weight 0.5 in the loss; Metrica games carry no contact target). The model therefore has to recover contact evidence from the smoothed inputs it will see on the target. This lifted the score from 0.675 to 0.685.

## Assets used

| Asset | Use |
|---|---|
| Tactical automatic tracking (TXT) + metadata XML | main input: players, ball, teams, pitch, halves |
| Automatic smart tagging (XML) | restart frame and team as a soft input feature |
| Tracking statistics of the provided match (no labels) | degradation rates and ball smoothing σ |
| Tactical / broadcast video, homography | investigated (camera recovery, sampled review); **not used** |
| Public five-minute reference labels | never used for training or selection; read only as a late diagnostic |

## Training data

All labelled data is external; none comes from the provided match.

| Dataset | Games | Source / licence | Labels: who and how |
|---|---|---|---|
| Metrica Sports sample data | 2 full matches | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), public, anonymised | derived by us from Metrica's event data: alive = clean tracking has the ball; team = team of the last on-ball event (pass, carry, recovery, shot, set piece, ball lost) |
| DFL / IDSSE Bundesliga open data | 5 full matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY) | Bassek, Rein, Weber et al., *Sci Data* 12, 195 (2025), CC-BY 4.0 | the tracking feed's own per-frame BallStatus and BallPossession; no extra filter |

Two further DFL matches (J03WQQ, J03WR9) were kept out of all training as a reserved check. Each final model trains on all seven games, four degradation draws each (seeds 100–103), 3,000 steps (one variant 6,000; one with width 128), batch 32, seed 7. (1,500 steps under-trained: held-out loss matched training loss on 6/7 games.)

## Possession rules

We did not write hand rules; the model learns the label conventions above and the decoder sets how much evidence a hand-over needs.

- **Change of possession:** a team switch must accumulate enough owner and transition evidence over consecutive frames to pay the switch cost. A single touch or rebound rarely does; a reception followed by control does.
- **Loose balls and challenges:** stay with the last team until the evidence turns, as in the challenge rules.
- **Dead ball:** `D` whenever the model sees play stopped. Restart team is an input hint, not a rule.
- **Abstention:** `X` only outside the two halves (40,462 frames). Inside play we always answer.

## Selection and validation

- **Protocol:** leave-one-game-out over the seven development games, each scored on three degradation draws (seeds 0–2), original labels, official spell-F1 scorer. Calibration of each fold uses an inner held-out game only. A change is promoted only if the mean gain is ≥ 0.005 and the game-bootstrap 90% lower bound is > 0.
- **Result:** frozen raw-trained model 0.607 → smooth-trained 0.675 (+0.068, lower bound +0.060, 7/7 games up) → plus contact head 0.685 (+0.010, lower bound +0.008, 7/7 games up; +0.008 at the fixed decoder costs) → trained 3,000 instead of 1,500 steps 0.702 (+0.017, lower bound +0.013, 7/7 games up; +0.017 at the fixed decoder costs) → average of the 3,000-step, 6,000-step and width-128 models 0.708 (+0.006, lower bound +0.002, 5/7 games up; +0.006, 6/7 at the fixed decoder costs).
- **Sensitivity** (measured on the smooth-trained model without the contact head): the same model scored on σ = 6 inputs 0.676 (unchanged), on σ = 10 inputs 0.667 (−0.008). If the target's smoothing differs from σ = 8, expect a small loss, not a collapse. The model barely reacts to smoothed player tracks (−0.001) or the target's frequent slot ID switches (0.000); halving the simulated ball noise (1.0 → 0.5 m) raises the score to 0.682, so the dev number carries about ±0.007 from that assumption.

**Tried and dropped** (all on the same protocol):

| Idea | Result |
|---|---|
| Blend with our earlier champion (synthetic-pretrained frame model + span classifier) | champion alone 0.546 on Barça-like inputs; every blend weight loses |
| Self-training on own target pseudo-labels | +0.0014, lower bound ≈ 0 |
| Ensemble of fold models; asymmetric decoder costs; GBM second stage | −0.0025; +0.0005; lost |
| Seed ensemble (2 seeds) | +0.0016, lower bound ≈ 0 |
| Two more DFL matches (the reserved J03WQQ/J03WR9) added to training | −0.0013, 2/7 games up |
| Retrain on target-like player tracks (smoothed, frequent ID switches) | −0.0024, 2/7 games up |
| Contact head also trained on Metrica games (proxy contacts from ball/player geometry) | −0.0004, 2/7 games up |
| Second training-only head regressing clean-ball speed, acceleration and turn | −0.0063, 0/7 games up |
| Wider contact target (σ = 4 frames instead of 1), since the smoothed ball can't time a touch to one frame | −0.0000, 3/7 games up |
| Longer training alone: 6,000 instead of 3,000 steps | +0.0022, lower bound −0.002, 3/7 games up (kept as an ensemble member) |
| Wider network alone: 128 instead of 96 channels, 3,000 steps | −0.0017, 2/7 games up (kept as an ensemble member) |
| Contact head's predictions as extra switch evidence in the decoder | +0.0029, lower bound ≈ 0, 4/7 games up |
| Re-normalising inputs to the target's own statistics | not run: the target's largest input shift is mass player dropouts during dead-ball stretches (ball unseen 89% of those frames), which the model already predicts as dead |
| White ball jitter after smoothing, to mimic the target's high-frequency floor | 1 cm already makes inputs less target-like (acceleration, spectrum); not trained |
| Rapid-exchange objectives, interval models, player tracklets, rerankers (GPU screens) | no gain on very short possessions |

## Limitations

- **σ = 8 is a stand-in.** It matches the target's ball statistics at ≤ 2 Hz; the target also has a flat high-frequency floor no Gaussian reproduces.
- **Very short possessions** (< 1 s, rapid exchanges) are mostly missed: the tracking rarely carries a clear contact, and every screen we built for them failed.
- **Label conventions differ** between Metrica (event-derived) and DFL (tracker-derived) and the organisers' ground truth; seven games is little data for that.
- **Real broadcast tracking is harder than our simulated noise.**
  - **What we ran:** a check on the 20 open SkillCorner A-League games (MIT licence, broadcast tracking at 10 fps). Nothing was tuned on them. The pipeline was unchanged: σ = 8 ball, decoder (8, 8, 1).
  - **On SkillCorner's own tracker possession field** (`possession.group`), the ensemble scores spellF1 0.351 (90% CI 0.341–0.360). That is below our older, hand-built peak decoder at 0.415, and the ensemble predicts 22% too many possessions.
  - **On labels built from SkillCorner's events, the two are close:** 0.581 vs 0.590.
  - **Not the frame rate:** a 10 fps control on our development games changes the ensemble by −0.002.
  - **Open:** the gap comes from label convention, broadcast tracking (off-screen players are extrapolated), or both. We could not separate them.
  - **Decoder:** a stickier setting, (24, 32, 1), recovers +0.028 there but costs −0.017 on the development games, so we kept (8, 8, 1). It is the first thing we would revisit with tracker-labelled real games.
- With more time: more training matches with tracker-derived labels, and learning the target's ball filter instead of assuming a Gaussian.

## Reproduction

From `research/` with the data bundle in `data/` (not in the repository):

1. Build degraded + smoothed development caches: `approaches/dense_control/smooth.py build --out <run> --seeds 0 1 2` (plus training seeds 100–103).
2. Contact targets from the clean DFL ball: `approaches/dense_control/contact.py targets --out <run>`.
3. Train and predict the seven outer folds and their inner calibration folds: `contact.py train|predict|calibrate --holdout <game> --role inner|main` (GPU recipe in `results/dense_contact/job.sh`).
4. Train the final model: `contact.py final freeze|train --out <run>/final --cache <run>/cache`, three times: plain, with `<run>/contact/STEPS` = 6000, and with `contact/WIDTH` = 128.
5. Per model: `contact.py run approaches/dense_control/smooth_target.py --dense <run> --work <run>/target --target <target cache> --csv <run>.csv --weight 1`. Then `approaches/dense_control/ens_target.py --works <three target dirs> --target <target cache> --csv Xyz.csv` and `python evaluation/validate_submission.py Xyz.csv`.

The target cache is the undegraded, unsmoothed provided tracking (`final.py build --game target`). Step 5 runs on CPU in about a minute and gives CSV SHA256 `41ef82f3feebce0a2244b561f5afe16fbece9e11ec61f22a305b0ac8c41f7898` from the three frozen final weights. Retraining on other hardware is not claimed to be bit-identical.
