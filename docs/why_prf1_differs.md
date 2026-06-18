# Why P/R/F1 differ across backends, but mAP doesn't

On the fixture `test` split (49 classes):

| way | threshold selection | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| `ours` (`Evaluation`, default `per_class`) | per-class max-F1 | 0.791 | 0.712 | 0.742 | 0.674 | 0.449 |
| `ultralytics` | one global threshold | 0.759 | 0.726 | 0.730 | 0.680 | 0.451 |
| `torchmetrics` | per-class max-F1 | 0.805 | 0.711 | 0.748 | 0.679 | 0.451 |

mAP agrees to ~0.002–0.006. P/R/F1 don't. **This is expected, not a bug** — the
reason is structural.

> The `ours` row uses the library default `confidence_optimization="per_class"`.
> Switch it to `"global"` and `ours` lands right next to `ultralytics` instead
> (P=0.741, R=0.738, F1=0.728) — same lever, see below.

![mean metrics per way](prf1_means.png)

## TL;DR

- **mAP is the *area under* the precision–recall curve.** It's a property of the
  whole curve, not of any one point. All three paths run the *same* matching
  (`iou_prior` ≈ Ultralytics' `match_predictions` ≈ COCO's per-class greedy-by-IoU),
  so they build the **same curve** → the same area → the same mAP. The confidence
  threshold can't move it.
- **P/R/F1 are a single *point* on that curve.** That point moves with (1) how you
  **select** the threshold and (2) how you **read** precision there. Same curve,
  different point ⇒ same mAP, different P/R/F1.

## The dominant lever: per-class vs. one global threshold

`ours` (default `per_class`) and `torchmetrics` both put **each class at its own
F1-optimal point**. `ultralytics` — and `ours` with
`confidence_optimization="global"` — apply **one** confidence shared by every class
(the one maximising the *mean* per-class F1), like YOLO's single `conf`. Per-class
selection can only match or beat a shared threshold, so the two per-class paths sit
at higher mean F1 (0.742 / 0.748) than the global one (0.730).

On one class's curve («ПленаГрубая», AP50 = 0.688):

![per-class P-R curve](prf1_pr_curve.png)

- **`ours` (circle) and `torchmetrics` (diamond) coincide** at the per-class max-F1
  point — **R = 0.676, P = 0.729, F1 = 0.702**.
- **The single global threshold (square)** — where `ultralytics` and `ours-global`
  put *this* class — sits further down the curve at **R = 0.798, P = 0.574,
  F1 = 0.668**. The value that's best *on average across all classes* (here 0.267) is
  too low for this class, so it trades precision for recall and loses F1.

The square and the circle are just **different points on the same curve** (the
arrow is the shift) — which is exactly why the *area*, mAP, is unchanged.

## The minor lever: raw vs. the COCO envelope

Within the per-class pair, why is `ours` P = 0.791 but `torchmetrics` P = 0.805?
Different precision *readout*:

- **`ours`** reports **raw, realizable** precision — the actual TP/(TP+FP) at a real
  confidence value (the blue step curve). What you'd measure in production.
- **`torchmetrics` / COCO** reads the **monotone precision envelope** (the red curve —
  precision made non-increasing in recall, "max to the right"), an upper bound on raw.

At the per-class F1-optimum these **almost coincide** (in the plot they're the same
point), because that optimum sits where the raw curve meets its envelope. So this
readout difference is only ~0.01 on mean precision — second-order next to the
per-class-vs-global selection above. `ultralytics` reads a 1000-point smoothed
interpolation, essentially the same curve again.

That same envelope is why **mAP still agrees**: COCO/VOC AP is the area under the
envelope, built from the same raw curve in all three paths; interpolation
redistributes precision along the curve but preserves the area.

## Why F1 clusters but P and R swing

F1 is what each path *maximises* at its point, so F1 stays in a narrow band
(0.730–0.748). What swings is **how that F1 splits into precision vs. recall**: a
single global threshold sits at higher recall / lower precision (the square); a
per-class optimum is more balanced (the circle). Same F1 ridge, different point
along it.

## Practical guidance

- **Compare detectors/backends on mAP** — it's the operating-point-invariant,
  backend-agnostic number; the three implementations agree on it to ~0.002.
- **Treat P/R/F1 as operating-point readouts.** To compare them fairly, fix the
  *same* selection rule (per-class vs. global) *and* the same precision definition
  (raw vs. envelope) on both sides — otherwise you're comparing two different points
  on the same curve.
- Per CLAUDE.md: this library's own `Evaluation` P/R/F1 are intentionally the *raw,
  realizable* numbers and are **not** meant to match YOLO's console output. The
  external backends (`compute_detection_metrics`) are the apples-to-apples path.

## Reproduce

```bash
# 3-way table
uv run --with ultralytics --with 'torchmetrics[detection]' python scripts/compare_backends.py

# the two figures in this doc
uv run --with ultralytics --with 'torchmetrics[detection]' python scripts/plot_prf1_vs_map.py
```
