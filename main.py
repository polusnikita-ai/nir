"""Точка входа CLI для запуска имитационной модели многороботной системы."""

from __future__ import annotations

import argparse
import json
import os

from src.experiments import run_benchmark_suite
from src.interfaces import HttpSmartContractLedger, InMemoryLedger
from src.models import ScenarioConfig
from src.simulator import MultiRobotSimulator
from src.visualizer import render_demo_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Имитационная модель многороботной системы (модульная платформа с поддержкой MRTA, CBAA, ставок и реестра)."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare", "demo", "experiments", "all"],
        default="demo",
        help="Режим запуска: single, compare (4 алгоритма), demo (видео 25 с), experiments, all",
    )
    parser.add_argument(
        "--allocator",
        choices=["cbaa", "round_robin", "greedy", "optimal"],
        default="cbaa",
        help="Подключаемый алгоритм распределения задач (Задачи 1Б и 2)",
    )
    parser.add_argument("--robots", type=int, default=6, help="Количество роботов")
    parser.add_argument("--tasks", type=int, default=20, help="Количество контрольных точек")
    parser.add_argument("--topology", choices=["full", "ring", "line"], default="ring", help="Топология сети")
    parser.add_argument("--alpha", type=float, default=0.5, help="Вес расстояния в функции ставки [0..1]")
    parser.add_argument("--beta-load", type=float, default=0.0, help="Вес загрузки робота в ставке (Задача 3)")
    parser.add_argument("--ledger-url", default="", help="URL API смарт-контракта для регистрации раундов (Задача 4)")
    parser.add_argument("--seed", type=int, default=42, help="Начальное зерно генератора случайных чисел")
    parser.add_argument("--duration", type=float, default=25.0, help="Длительность видео, сек")
    parser.add_argument("--video-out", default="demo/simulation_demo.mp4", help="Путь сохранения видео MP4/GIF")
    parser.add_argument("--results-dir", default="results", help="Папка сохранения результатов экспериментов")
    parser.add_argument("--seeds", type=int, default=15, help="Число сценариев на конфигурацию в экспериментах")
    parser.add_argument("--log-out", default="", help="Путь сохранения подробного JSON-журнала в режиме single")
    parser.add_argument("--no-scalability", action="store_true", help="Пропустить серию масштабируемости")
    args = parser.parse_args()

    ledger = (
        HttpSmartContractLedger(endpoint_url=args.ledger_url, dry_run=False)
        if args.ledger_url
        else InMemoryLedger()
    )

    if args.mode == "single":
        cfg = ScenarioConfig(
            num_robots=args.robots,
            num_tasks=args.tasks,
            topology=args.topology,
            alpha=args.alpha,
            beta_load=args.beta_load,
            allocator_name=args.allocator,
            seed=args.seed,
        )
        sim = MultiRobotSimulator(cfg, record_frames=False, log_messages=bool(args.log_out), ledger=ledger)
        metrics = sim.run()
        if args.log_out:
            out_dir = os.path.dirname(args.log_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.log_out, "w", encoding="utf-8") as f:
                json.dump(sim.get_full_logs(), f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.mode == "compare":
        comparison = []
        for alloc_name in ("round_robin", "greedy", "optimal", "cbaa"):
            cfg = ScenarioConfig(
                num_robots=args.robots,
                num_tasks=args.tasks,
                topology=args.topology,
                alpha=args.alpha,
                beta_load=args.beta_load,
                allocator_name=alloc_name,
                seed=args.seed,
                dt=0.5,
            )
            sim = MultiRobotSimulator(cfg, record_frames=False)
            m = sim.run()
            comparison.append(
                {
                    "allocator": alloc_name,
                    "coverage": round(m["coverage"] * 100.0, 2),
                    "completed": m["completed_tasks"],
                    "energy_spent": m["total_energy_spent"],
                    "energy_per_task": m["energy_per_task"],
                    "distance": m["total_distance"],
                    "rounds": m["rounds"],
                    "messages": m["messages_sent"],
                    "violations": m["violations"],
                }
            )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))

    if args.mode in ("demo", "all"):
        cfg = ScenarioConfig(
            num_robots=args.robots,
            num_tasks=args.tasks,
            topology=args.topology,
            alpha=args.alpha,
            beta_load=args.beta_load,
            allocator_name=args.allocator,
            seed=args.seed,
            dt=0.25,
        )
        metrics = render_demo_video(args.video_out, cfg=cfg, duration_sec=args.duration, fps=24)
        print(f"Видео сохранено: {args.video_out}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.mode in ("experiments", "all"):
        res = run_benchmark_suite(
            out_dir=args.results_dir,
            num_seeds=args.seeds,
            run_scalability=not args.no_scalability,
        )
        print(f"Эксперименты завершены. Сводка: {res['summary_csv']}")
        print(f"Графики: {', '.join(res['charts'])}")


if __name__ == "__main__":
    main()
