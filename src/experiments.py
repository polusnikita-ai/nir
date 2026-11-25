"""Проведение парных вычислительных экспериментов и экспорт результатов в CSV/JSON."""

from __future__ import annotations

import csv
import json
import os
import statistics
from typing import Dict, List

from .models import ScenarioConfig
from .simulator import MultiRobotSimulator
from .visualizer import plot_experiment_charts


def run_benchmark_suite(out_dir: str = "results", num_seeds: int = 15, run_scalability: bool = True) -> Dict:
    """Серия парных экспериментов: 3 топологии × 2 режима × 4 значения alpha × num_seeds сценариев."""
    os.makedirs(out_dir, exist_ok=True)
    topologies = ["full", "ring", "line"]
    alphas = [0.25, 0.5, 0.75, 1.0]
    regimes = [False, True]  # False = однородный, True = неоднородный

    raw_rows: List[Dict] = []
    for topo in topologies:
        for het in regimes:
            for alpha in alphas:
                for s in range(num_seeds):
                    cfg = ScenarioConfig(
                        arena_size=100.0,
                        base_pos=(50.0, 50.0),
                        num_robots=6,
                        num_tasks=30,
                        num_obstacles=4,
                        topology=topo,
                        heterogeneous=het,
                        alpha=alpha,
                        seed=1000 + s,
                        dt=0.5,
                    )
                    sim = MultiRobotSimulator(cfg, record_frames=False)
                    raw_rows.append(sim.run())

    # Сохранение всех запусков в CSV
    csv_path = os.path.join(out_dir, "experiments_raw.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    # Агрегация сводной таблицы по конфигурациям (включая медиану и разброс ΔK)
    summary_rows: List[Dict] = []
    for topo in topologies:
        for het in regimes:
            base_map = {
                r["seed"]: r
                for r in raw_rows
                if r["topology"] == topo and r["heterogeneous"] == het and abs(r["alpha"] - 1.0) < 1e-6
            }
            for alpha in alphas:
                subset = [
                    r
                    for r in raw_rows
                    if r["topology"] == topo and r["heterogeneous"] == het and abs(r["alpha"] - alpha) < 1e-6
                ]
                deltas = [r["coverage"] - base_map[r["seed"]]["coverage"] for r in subset]
                ept_vals = [r["energy_per_task"] for r in subset if r["energy_per_task"] is not None]
                std_dk = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                summary_rows.append(
                    {
                        "topology": topo,
                        "heterogeneous": het,
                        "alpha": alpha,
                        "runs": len(subset),
                        "mean_coverage": round(statistics.mean(r["coverage"] for r in subset), 4),
                        "mean_delta_k": round(statistics.mean(deltas), 4),
                        "median_delta_k": round(statistics.median(deltas), 4),
                        "std_delta_k": round(std_dk, 4),
                        "min_delta_k": round(min(deltas), 4),
                        "max_delta_k": round(max(deltas), 4),
                        "positive_deltas": sum(1 for d in deltas if d > 1e-9),
                        "zero_deltas": sum(1 for d in deltas if abs(d) <= 1e-9),
                        "negative_deltas": sum(1 for d in deltas if d < -1e-9),
                        "mean_energy": round(statistics.mean(r["total_energy_spent"] for r in subset), 2),
                        "mean_energy_per_task": round(statistics.mean(ept_vals), 2) if ept_vals else 0.0,
                        "mean_distance": round(statistics.mean(r["total_distance"] for r in subset), 2),
                        "mean_rounds": round(statistics.mean(r["rounds"] for r in subset), 2),
                        "mean_sim_time_sec": round(statistics.mean(r["sim_time_sec"] for r in subset), 2),
                        "mean_messages": round(statistics.mean(r["messages_sent"] for r in subset), 1),
                        "mean_bytes": round(statistics.mean(r["bytes_sent"] for r in subset), 1),
                        "mean_sync_steps": round(statistics.mean(r["sync_steps"] for r in subset), 1),
                        "mean_wall_time_ms": round(statistics.mean(r["wall_time_ms"] for r in subset), 2),
                        "total_penetrations": sum(r["obstacle_penetrations"] for r in subset),
                        "unconverged_rounds": sum(r["unconverged_rounds"] for r in subset),
                        "total_violations": sum(r["violations"] for r in subset),
                    }
                )

    summary_csv = os.path.join(out_dir, "experiments_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = os.path.join(out_dir, "experiments_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    charts = plot_experiment_charts(summary_rows, out_dir)

    scalability_rows: List[Dict] = []
    if run_scalability:
        scalability_rows = run_scalability_suite(out_dir=out_dir, num_seeds=5)

    return {
        "raw_csv": csv_path,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "charts": charts,
        "summary_rows": summary_rows,
        "scalability_rows": scalability_rows,
    }


def run_scalability_suite(out_dir: str = "results", num_seeds: int = 5) -> List[Dict]:
    """Дополнительная серия масштабируемости по числу роботов n ∈ {5, 10, 20} и точек m ∈ {20, 50, 100}."""
    os.makedirs(out_dir, exist_ok=True)
    robot_counts = [5, 10, 20]
    task_counts = [20, 50, 100]
    alphas = [0.75, 1.0]

    rows: List[Dict] = []
    for n in robot_counts:
        for m in task_counts:
            for alpha in alphas:
                runs: List[Dict] = []
                for s in range(num_seeds):
                    cfg = ScenarioConfig(
                        arena_size=100.0,
                        base_pos=(50.0, 50.0),
                        num_robots=n,
                        num_tasks=m,
                        num_obstacles=4,
                        topology="ring",
                        heterogeneous=True,
                        alpha=alpha,
                        seed=2000 + s,
                        dt=0.5,
                    )
                    runs.append(MultiRobotSimulator(cfg, record_frames=False).run())
                rows.append(
                    {
                        "num_robots": n,
                        "num_tasks": m,
                        "topology": "ring",
                        "alpha": alpha,
                        "runs": len(runs),
                        "mean_coverage": round(statistics.mean(r["coverage"] for r in runs), 4),
                        "mean_rounds": round(statistics.mean(r["rounds"] for r in runs), 2),
                        "mean_sync_steps": round(statistics.mean(r["sync_steps"] for r in runs), 1),
                        "mean_messages": round(statistics.mean(r["messages_sent"] for r in runs), 1),
                        "mean_bytes": round(statistics.mean(r["bytes_sent"] for r in runs), 1),
                        "mean_energy": round(statistics.mean(r["total_energy_spent"] for r in runs), 2),
                        "mean_wall_time_ms": round(statistics.mean(r["wall_time_ms"] for r in runs), 2),
                        "total_violations": sum(r["violations"] for r in runs),
                    }
                )

    sc_csv = os.path.join(out_dir, "scalability_summary.csv")
    with open(sc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sc_json = os.path.join(out_dir, "scalability_summary.json")
    with open(sc_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return rows
