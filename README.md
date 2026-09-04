# Sampling-based replanning under moving obstacles: RRT, RRT-Connect, RRT\* and RRTX

A benchmark that puts numbers on a trade every motion-planning course states qualitatively:
**finding a path quickly and finding a good path are not the same objective**, and the gap widens
once the obstacles move and you have to replan.

Four planners, one discrete-time replanning framework, the same scenario, the same seeds. Then the
same four run against a physical workspace to see whether the simulation ranking survives contact
with a camera and a projector.

---

## The result, up front

Averaged over seeded runs of a moving-obstacle scenario (`results/simulation/summary_3planners.csv`):

| Planner | Success | First solution (ms) | Total planning (ms) | Path length (px) | Replans |
|---|---|---|---|---|---|
| RRT | 100% | 6559 | 6559 | 1275 | 7.4 |
| RRT-Connect | 100% | 6095 | 6095 | 1254 | 7.1 |
| RRT\* | 100% | 8579 | 19002 | 975 | 14.9 |

**RRT-Connect reaches a feasible path about 29% faster than RRT\*. RRT\* returns paths about 22%
shorter, and pays roughly three times the total planning cost to do it.** That is the trade, stated
in milliseconds and pixels rather than adjectives.

**RRTX is the interesting failure.** It produces a first solution in 70 ms, two orders of magnitude
faster than anything else, and then succeeds only 15% of the time
(`results/simulation/benchmark_summary.csv`). Its 32 repairs per run and 2275-node tree show where
the cost went: it is rewiring constantly and, in this scenario and this implementation, not
converging to a usable path. A benchmark that only reported the winners would have hidden that.

Minimum clearance is worth reading alongside speed. RRT keeps 10.2 px of clearance; RRT-Connect and
RRT\* sit near 2.5 px. Faster and shorter paths hug obstacles harder, which matters the moment the
obstacle is real and its position is uncertain.

## Physical validation

The four planners were then run against a real workspace: a top-down camera detecting obstacles by
colour, a calibrated camera-to-projector mapping, and the planned path projected live onto the table
so a person could move a physical obstacle and watch the planner respond
(`results/physical/physical_validation_final_summary.csv`).

| Planner | Time to goal (ms) | Total planning (ms) | Executed path (px) | Nodes |
|---|---|---|---|---|
| RRT | 987 | 3636 | 2120 | 271 |
| RRT-Connect | 421 | 1763 | 1783 | 130 |
| RRT\* | 987 | 14706 | 2156 | 1922 |
| RRTX | 987 | 987 | 1743 | 3230 |

All four reached the goal. The ordering broadly held: RRT-Connect fastest to goal and cheapest in
nodes, RRT\* most expensive by an order of magnitude.

**The rig fought itself at first.** The projected path changed the colours the camera saw, which
corrupted the obstacle mask, which changed the path. A feedback loop between the two halves of the
system that does not exist in either half alone, and only appears once you close the loop. That is
the thing simulation cannot tell you.

---

## Layout

```
planners/    the benchmark harness and the planner implementations
analysis/    metric extraction, plotting, video rendering
demos/       small standalone RRT and RRT-Connect demos
results/     the CSVs behind every number in this README
figures/     the plots and scenario snapshots
```

| File | What it does |
|---|---|
| `planners/planner_comparison_full.py` | The full four-planner comparison: RRT, RRT-Connect, RRT\*, RRTX in one discrete-time replanning loop, with overlay rendering and CSV logging. |
| `planners/benchmark_sim.py` | The core benchmark loop, one scenario, seeded runs. |
| `planners/benchmark_sim_2x2.py` | Four-panel side-by-side comparison with corrected metric accounting. |
| `planners/benchmark_sim_moving_obstacles.py` | The moving-obstacle variant the headline numbers come from. |
| `analysis/plot_3planners_from_csv.py` | Rebuilds the report figures from `benchmark_detailed.csv`. |
| `analysis/physical_validation_plots.py` | The physical-rig plots from the logged run data. |
| `analysis/inspect_all_csvs.py` | Sanity check: schema and row counts for every logged CSV before you trust a plot. |
| `analysis/make_video.py` | Renders the planner comparison to video. |
| `demos/` | Minimal RRT, RRT-Connect and side-by-side demos, useful for seeing the difference in thirty seconds. |

## Running it

```bash
pip install -r requirements.txt

export RESULTS_DIR=./results/simulation
export PHYSICAL_DATA_DIR=./results/physical
export OUT_DIR=./figures

python planners/benchmark_sim_moving_obstacles.py    # produces benchmark_detailed.csv
python analysis/inspect_all_csvs.py                  # check before plotting
python analysis/plot_3planners_from_csv.py
python analysis/physical_validation_plots.py
```

## Reading the metrics

- **first_solution_ms** is time to *any* feasible path. **total_planning_ms** includes every replan
  and, for RRT\*, all the rewiring. They diverge sharply for the asymptotically optimal planners,
  and quoting only the first is how benchmarks flatter themselves.
- **replans** counts how often the moving obstacle invalidated the current path. **repairs** is
  RRTX-specific rewiring.
- **min_clearance_px** is the closest the executed path came to an obstacle. Short paths are usually
  bought with clearance.

## Limitations

- One scenario family and one workspace size. The ranking is conditioned on both; a narrow-corridor
  scenario would likely reward RRT-Connect's bidirectional search even more.
- RRTX's 15% success rate is a result for *this implementation in this scenario*, not a claim about
  the algorithm. Treat it as a lead to investigate, not a verdict.
- Clearance is measured in pixels, in the workspace's own frame, not in metres.
- The physical validation is a single session per planner, not a repeated trial.

## Attribution

Course project for AER1516, Robot Motion Planning, University of Toronto Institute for Aerospace
Studies, Winter 2026. The planner benchmark and the analysis in this repository are my work; the
project was carried out in a four-person team and the physical camera-and-projector rig was built
with teammates.

## Licence

MIT, see `LICENSE`.

---

Sheik Farhaan Elaheebocus · [linkedin.com/in/farhaan-elaheebocus](https://www.linkedin.com/in/farhaan-elaheebocus)
