"""Движок имитационного моделирования многороботной системы."""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from .allocators import build_allocator
from .interfaces import AssignmentLedger, BidFunction, EnergyLoadBidFunction, InMemoryLedger, TaskAllocator
from .models import (
    FrameSnapshot,
    Obstacle,
    ObstaclePathPlanner,
    Point,
    Robot,
    ScenarioConfig,
    Task,
)
from .network import CommunicationNetwork


def generate_scenario(cfg: ScenarioConfig) -> Tuple[List[Robot], List[Task], List[Obstacle]]:
    """Детерминированная генерация сцены: роботов, контрольных точек и препятствий."""
    rng = random.Random(cfg.seed)
    bx, by = cfg.base_pos

    # 1. Препятствия (не перекрывают базу и центральную зону старта)
    obstacles: List[Obstacle] = []
    attempts = 0
    while len(obstacles) < cfg.num_obstacles and attempts < 500:
        attempts += 1
        ox = rng.uniform(15.0, cfg.arena_size - 15.0)
        oy = rng.uniform(15.0, cfg.arena_size - 15.0)
        rad = rng.uniform(4.5, 7.5)
        if math.hypot(ox - bx, oy - by) < rad + 17.0:
            continue
        if any(math.hypot(ox - o.x, oy - o.y) < rad + o.radius + 8.0 for o in obstacles):
            continue
        obstacles.append(Obstacle(ox, oy, rad))

    # 2. Роботы в центральной зоне вокруг базы [35, 65] x [35, 65]
    robots: List[Robot] = []
    for i in range(cfg.num_robots):
        angle = 2.0 * math.pi * i / max(1, cfg.num_robots)
        r_dist = rng.uniform(6.0, 14.0)
        rx = bx + r_dist * math.cos(angle)
        ry = by + r_dist * math.sin(angle)
        if cfg.heterogeneous:
            gamma = rng.uniform(0.8, 1.2)
            eta = rng.uniform(0.8, 1.2)
            extra_energy = rng.uniform(80.0, 200.0)
        else:
            gamma = 1.0
            eta = 1.0
            extra_energy = 140.0
        reserve = 10.0
        init_e = gamma * math.hypot(rx - bx, ry - by) + reserve + extra_energy
        robots.append(
            Robot(
                robot_id=i,
                x=rx,
                y=ry,
                base_x=bx,
                base_y=by,
                energy=init_e,
                initial_energy=init_e,
                reserve_energy=reserve,
                energy_per_meter=gamma,
                inspection_factor=eta,
                speed=6.0,
            )
        )

    # 3. Контрольные точки обследования вне препятствий
    tasks: List[Task] = []
    attempts = 0
    while len(tasks) < cfg.num_tasks and attempts < 2000:
        attempts += 1
        tx = rng.uniform(6.0, cfg.arena_size - 6.0)
        ty = rng.uniform(6.0, cfg.arena_size - 6.0)
        if any(math.hypot(tx - o.x, ty - o.y) < o.radius + 3.0 for o in obstacles):
            continue
        if math.hypot(tx - bx, ty - by) < 8.0:
            continue
        if any(math.hypot(tx - t.x, ty - t.y) < 6.0 for t in tasks):
            continue
        h_j = rng.uniform(2.0, 8.0)
        tasks.append(
            Task(
                task_id=len(tasks),
                x=tx,
                y=ty,
                inspection_time=2.5,
                base_energy=h_j,
            )
        )

    return robots, tasks, obstacles


class MultiRobotSimulator:
    """Дискретно-временная имитационная модель многороботной системы."""

    def __init__(
        self,
        cfg: ScenarioConfig,
        record_frames: bool = False,
        log_messages: bool = False,
        allocator: Optional[TaskAllocator] = None,
        bid_fn: Optional[BidFunction] = None,
        ledger: Optional[AssignmentLedger] = None,
    ) -> None:
        self.cfg = cfg
        self.record_frames = record_frames
        self.allocator = allocator or build_allocator(cfg.allocator_name, alpha=cfg.alpha, beta_load=cfg.beta_load)
        self.bid_fn = bid_fn or EnergyLoadBidFunction(alpha=cfg.alpha, beta_load=cfg.beta_load)
        self.ledger = ledger if ledger is not None else InMemoryLedger()
        self.last_ledger_tx: str = ""
        self.robots, self.tasks, self.obstacles = generate_scenario(cfg)
        self.planner = ObstaclePathPlanner(self.obstacles, arena_size=cfg.arena_size, safety_margin=0.75)
        self.network = CommunicationNetwork(cfg.num_robots, cfg.topology, log_messages=log_messages)
        self.diagonal = math.hypot(cfg.arena_size, cfg.arena_size)

        self.sim_time: float = 0.0
        self.rounds_executed: int = 0
        self.total_sync_steps: int = 0
        self.violations: int = 0
        self.unconverged_rounds: int = 0
        self.obstacle_penetrations: int = 0
        self.wall_time_sec: float = 0.0
        self.frames: List[FrameSnapshot] = []
        self.round_logs: List[Dict] = []
        self.task_logs: List[Dict] = []

        self._waypoints: Dict[int, List[Point]] = {r.robot_id: [] for r in self.robots}
        self._inspection_energy_spent: float = 0.0

    def _capture_frame(
        self,
        phase: str,
        active_links: Optional[List[Tuple[int, int]]] = None,
        bid_annotations: Optional[List[Dict]] = None,
        phase_detail: str = "",
        contested_tasks: Optional[List[int]] = None,
        round_summary: Optional[List[str]] = None,
    ) -> None:
        if not self.record_frames:
            return
        self.frames.append(
            FrameSnapshot(
                sim_time=self.sim_time,
                round_idx=self.rounds_executed,
                phase=phase,
                robots=[
                    {
                        "id": r.robot_id,
                        "x": r.x,
                        "y": r.y,
                        "energy": r.energy,
                        "initial_energy": r.initial_energy,
                        "reserve": r.reserve_energy,
                        "state": r.state,
                        "assigned": r.assigned_task_id,
                        "traj": list(r.trajectory[-90:]),
                    }
                    for r in self.robots
                ],
                tasks=[
                    {
                        "id": t.task_id,
                        "x": t.x,
                        "y": t.y,
                        "completed": t.completed,
                        "assigned_to": t.assigned_to,
                    }
                    for t in self.tasks
                ],
                active_links=list(active_links or []),
                bid_annotations=list(bid_annotations or []),
                messages_total=self.network.messages_sent,
                bytes_total=self.network.bytes_sent,
                completed_count=sum(1 for t in self.tasks if t.completed),
                total_tasks=len(self.tasks),
                phase_detail=phase_detail,
                contested_tasks=list(contested_tasks or []),
                round_summary=list(round_summary or []),
                ledger_tx=self.last_ledger_tx,
                allocator_name=self.allocator.name,
            )
        )

    def _verify_round_invariants(
        self,
        assignments: Dict[int, Optional[int]],
        converged: bool,
    ) -> None:
        """Проверка инвариантов допустимости распределения и сходимости протокола."""
        if not converged:
            self.unconverged_rounds += 1
            self.violations += 1
        seen_tasks = set()
        task_map = {t.task_id: t for t in self.tasks}
        for r in self.robots:
            tid = assignments.get(r.robot_id)
            if tid is None:
                continue
            if tid in seen_tasks:
                self.violations += 1
            seen_tasks.add(tid)
            t = task_map[tid]
            if t.completed or not r.can_execute(t, planner=self.planner):
                self.violations += 1

    def _check_obstacle_clearance(self, robot: Robot) -> None:
        for obs in self.obstacles:
            if math.hypot(robot.x - obs.x, robot.y - obs.y) < obs.radius - 1e-6:
                self.obstacle_penetrations += 1
                self.violations += 1

    def _step_along_waypoints(self, robot: Robot, dt: float) -> bool:
        """Продвижение робота вдоль запланированной полилинии обхода препятствий."""
        wps = self._waypoints.get(robot.robot_id, [])
        if not wps:
            return True

        remaining_step = robot.speed * dt
        moved_total = 0.0
        while remaining_step > 1e-12 and wps:
            wx, wy = wps[0]
            dx = wx - robot.x
            dy = wy - robot.y
            dist = math.hypot(dx, dy)
            if dist <= remaining_step + 1e-12:
                robot.x, robot.y = wx, wy
                moved_total += dist
                remaining_step -= dist
                wps.pop(0)
            else:
                ratio = remaining_step / dist
                robot.x += dx * ratio
                robot.y += dy * ratio
                moved_total += remaining_step
                remaining_step = 0.0

        robot.distance_traveled += moved_total
        robot.energy = max(0.0, robot.energy - robot.energy_per_meter * moved_total)
        robot.trajectory.append((robot.x, robot.y))
        self._check_obstacle_clearance(robot)
        return len(wps) == 0

    def _verify_mission_invariants(self) -> None:
        """Проверка финальных инвариантов миссии: резерв, возврат на базу и баланс энергии."""
        bx, by = self.cfg.base_pos
        theoretical_spent = self._inspection_energy_spent
        for r in self.robots:
            if r.energy < r.reserve_energy - 1e-5:
                self.violations += 1
            if math.hypot(r.x - bx, r.y - by) > 1e-3:
                self.violations += 1
            theoretical_spent += r.energy_per_meter * r.distance_traveled

        actual_spent = sum(r.initial_energy - r.energy for r in self.robots)
        if abs(actual_spent - theoretical_spent) > 1e-2:
            self.violations += 1

    def run(self) -> Dict:
        """Запуск полной миссии: раунды торгов, исполнение, возврат на базу."""
        t0 = time.perf_counter()
        self._capture_frame(
            "AUCTION",
            active_links=self.network.edges,
            phase_detail="ИНИЦИАЛИЗАЦИЯ СЦЕНЫ И СЕТИ СВЯЗИ",
        )

        while True:
            open_tasks = [t for t in self.tasks if not t.completed]
            if not open_tasks:
                break

            self.rounds_executed += 1
            for r in self.robots:
                r.state = "NEGOTIATING"
                r.assigned_task_id = None

            alloc_res = self.allocator.allocate(
                robots=self.robots,
                open_tasks=open_tasks,
                network=self.network,
                diagonal=self.diagonal,
                round_idx=self.rounds_executed,
                planner=self.planner,
                bid_fn=self.bid_fn,
                record_history=self.record_frames,
            )
            assignments = alloc_res.assignments
            self.total_sync_steps += alloc_res.sync_steps
            self._verify_round_invariants(assignments, alloc_res.converged)

            active_assignments = {rid: tid for rid, tid in assignments.items() if tid is not None}
            receipt = None
            if active_assignments and self.ledger is not None:
                receipt = self.ledger.register_round(
                    round_idx=self.rounds_executed,
                    assignments=active_assignments,
                    metadata={
                        "allocator": self.allocator.name,
                        "allocator": self.allocator.name,
            "topology": self.cfg.topology,
                        "alpha": self.cfg.alpha,
                    },
                )
                self.last_ledger_tx = receipt.tx_hash

            # Записываем шаги фазы аукциона (выбор, обмен ставками, разрешение конфликтов, консенсус)
            if self.record_frames:
                for step_info in alloc_res.step_history:
                    self._capture_frame(
                        "AUCTION",
                        active_links=step_info["active_links"],
                        bid_annotations=step_info["bid_annotations"],
                        phase_detail=step_info["phase_detail"],
                        contested_tasks=step_info["contested_tasks"],
                        round_summary=step_info["round_summary"],
                    )

            self.round_logs.append(
                {
                    "round": self.rounds_executed,
                    "allocator": self.allocator.name,
                    "assigned": active_assignments,
                    "sync_steps": alloc_res.sync_steps,
                    "converged": alloc_res.converged,
                    "ledger_tx": receipt.tx_hash if receipt else None,
                    "messages_cumulative": self.network.messages_sent,
                    "bytes_cumulative": self.network.bytes_sent,
                }
            )

            if not active_assignments:
                break

            # Применяем согласованные назначения и строим свободные от препятствий маршруты
            task_map = {t.task_id: t for t in self.tasks}
            for r in self.robots:
                tid = active_assignments.get(r.robot_id)
                if tid is not None:
                    t = task_map[tid]
                    t.assigned_to = r.robot_id
                    r.assigned_task_id = tid
                    r.state = "MOVING"
                    r.inspect_remaining = t.inspection_time
                    path, _ = self.planner.shortest_path((r.x, r.y), (t.x, t.y))
                    self._waypoints[r.robot_id] = path[1:]
                else:
                    r.state = "IDLE"
                    self._waypoints[r.robot_id] = []

            # Исполнение назначений раунда до завершения всеми назначенными роботами
            while any(r.state in ("MOVING", "INSPECTING") for r in self.robots):
                self.sim_time += self.cfg.dt
                for r in self.robots:
                    if r.state == "MOVING" and r.assigned_task_id is not None:
                        arrived = self._step_along_waypoints(r, self.cfg.dt)
                        if arrived:
                            r.state = "INSPECTING"
                    elif r.state == "INSPECTING" and r.assigned_task_id is not None:
                        r.inspect_remaining -= self.cfg.dt
                        if r.inspect_remaining <= 1e-9:
                            t = task_map[r.assigned_task_id]
                            t.completed = True
                            t.assigned_to = None
                            insp_cost = t.base_energy * r.inspection_factor
                            r.energy = max(0.0, r.energy - insp_cost)
                            self._inspection_energy_spent += insp_cost
                            r.tasks_completed += 1
                            self.task_logs.append(
                                {
                                    "round": self.rounds_executed,
                                    "task_id": t.task_id,
                                    "robot_id": r.robot_id,
                                    "completed_time_sec": round(self.sim_time, 2),
                                    "robot_energy_after": round(r.energy, 3),
                                }
                            )
                            r.assigned_task_id = None
                            r.state = "IDLE"
                self._capture_frame("MOVING", phase_detail="ДВИЖЕНИЕ С ОБХОДОМ ПРЕПЯТСТВИЙ И ОБСЛЕДОВАНИЕ")

        # Возвращение всех роботов на базу после завершения рабочего цикла
        for r in self.robots:
            r.state = "RETURNING"
            path, _ = self.planner.shortest_path((r.x, r.y), (r.base_x, r.base_y))
            self._waypoints[r.robot_id] = path[1:]

        while any(r.state == "RETURNING" for r in self.robots):
            self.sim_time += self.cfg.dt
            for r in self.robots:
                if r.state == "RETURNING":
                    arrived = self._step_along_waypoints(r, self.cfg.dt)
                    if arrived:
                        r.state = "AT_BASE"
            self._capture_frame("RETURN_BASE", phase_detail="ВОЗВРАЩЕНИЕ ГРУППЫ НА БАЗУ С СОХРАНЕНИЕМ РЕЗЕРВА")

        self._verify_mission_invariants()
        self._capture_frame("FINISHED", phase_detail="МИССИЯ УСПЕШНО ЗАВЕРШЕНА")
        self.wall_time_sec = time.perf_counter() - t0
        return self.get_metrics()

    def get_metrics(self) -> Dict:
        completed = sum(1 for t in self.tasks if t.completed)
        total = len(self.tasks)
        total_energy_spent = sum(r.initial_energy - r.energy for r in self.robots)
        total_distance = sum(r.distance_traveled for r in self.robots)
        min_final_energy = min(r.energy for r in self.robots)
        return {
            "seed": self.cfg.seed,
            "num_robots": self.cfg.num_robots,
            "num_tasks": total,
            "topology": self.cfg.topology,
            "heterogeneous": self.cfg.heterogeneous,
            "alpha": self.cfg.alpha,
            "completed_tasks": completed,
            "coverage": completed / total if total else 0.0,
            "total_energy_spent": round(total_energy_spent, 3),
            "energy_per_task": round(total_energy_spent / completed, 3) if completed > 0 else None,
            "total_distance": round(total_distance, 3),
            "min_final_energy": round(min_final_energy, 3),
            "rounds": self.rounds_executed,
            "sync_steps": self.total_sync_steps,
            "messages_sent": self.network.messages_sent,
            "bytes_sent": self.network.bytes_sent,
            "sim_time_sec": round(self.sim_time, 2),
            "wall_time_ms": round(self.wall_time_sec * 1000.0, 2),
            "unconverged_rounds": self.unconverged_rounds,
            "obstacle_penetrations": self.obstacle_penetrations,
            "violations": self.violations,
        }

    def get_full_logs(self) -> Dict:
        """Полный журнал наблюдателя: метрики, раунды, задачи, состояние роботов и сообщения."""
        return {
            "config": {
                "arena_size": self.cfg.arena_size,
                "base_pos": self.cfg.base_pos,
                "num_robots": self.cfg.num_robots,
                "num_tasks": self.cfg.num_tasks,
                "num_obstacles": self.cfg.num_obstacles,
                "topology": self.cfg.topology,
                "heterogeneous": self.cfg.heterogeneous,
                "alpha": self.cfg.alpha,
                "seed": self.cfg.seed,
                "dt": self.cfg.dt,
            },
            "metrics": self.get_metrics(),
            "round_logs": self.round_logs,
            "task_logs": self.task_logs,
            "robot_logs": [
                {
                    "robot_id": r.robot_id,
                    "initial_energy": round(r.initial_energy, 3),
                    "final_energy": round(r.energy, 3),
                    "reserve_energy": round(r.reserve_energy, 3),
                    "energy_per_meter": round(r.energy_per_meter, 4),
                    "inspection_factor": round(r.inspection_factor, 4),
                    "distance_traveled": round(r.distance_traveled, 3),
                    "tasks_completed": r.tasks_completed,
                    "trajectory_points": len(r.trajectory),
                }
                for r in self.robots
            ],
            "ledger_records": self.ledger.get_records() if self.ledger else [],
            "message_log": self.network.message_log,
        }
