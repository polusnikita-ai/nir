"""Реализации подключаемых алгоритмов распределения задач (Задачи 1Б и 2).

Поддерживаются:
- RoundRobinAllocator (Задача 1Б): циклическое назначение без оценки расстояний;
- GreedyNearestAllocator (Задача 1Б): выбор допустимого робота с минимальной стоимостью;
- OptimalAssignmentAllocator (Задача 1Б): глобально оптимальное назначение (венгерский алгоритм);
- DecentralizedAuctionAllocator (Задача 2 + Задача 3): децентрализованный аукцион CBAA с подключаемой ставкой.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .allocator import DecentralizedAuctionRound
from .interfaces import AllocationResult, BidFunction, EnergyLoadBidFunction, TaskAllocator
from .models import ObstaclePathPlanner, Robot, Task
from .network import CommunicationNetwork


class RoundRobinAllocator(TaskAllocator):
    """Базовый алгоритм Round Robin: назначает задачи свободным роботам по очереди без оценки расстояний."""

    name = "round_robin"

    def __init__(self) -> None:
        self._cursor: int = 0

    def allocate(
        self,
        robots: List[Robot],
        open_tasks: List[Task],
        network: CommunicationNetwork,
        diagonal: float,
        round_idx: int,
        planner: Optional[ObstaclePathPlanner] = None,
        bid_fn: Optional[BidFunction] = None,
        record_history: bool = False,
    ) -> AllocationResult:
        assignments: Dict[int, Optional[int]] = {r.robot_id: None for r in robots}
        if not robots or not open_tasks:
            return AllocationResult(assignments=assignments, sync_steps=0, converged=True, algorithm_name=self.name)

        n = len(robots)
        used_robots: set[int] = set()
        for task in sorted(open_tasks, key=lambda t: t.task_id):
            if len(used_robots) == n:
                break
            for offset in range(n):
                idx = (self._cursor + offset) % n
                r = robots[idx]
                if r.robot_id in used_robots:
                    continue
                if r.can_execute(task, planner=planner):
                    assignments[r.robot_id] = task.task_id
                    used_robots.add(r.robot_id)
                    self._cursor = (idx + 1) % n
                    break

        return AllocationResult(
            assignments=assignments,
            sync_steps=1,
            converged=True,
            algorithm_name=self.name,
        )


class GreedyNearestAllocator(TaskAllocator):
    """Базовый алгоритм Greedy / Nearest Robot: для очередной задачи выбирает свободного робота с минимальной стоимостью."""

    name = "greedy"

    def allocate(
        self,
        robots: List[Robot],
        open_tasks: List[Task],
        network: CommunicationNetwork,
        diagonal: float,
        round_idx: int,
        planner: Optional[ObstaclePathPlanner] = None,
        bid_fn: Optional[BidFunction] = None,
        record_history: bool = False,
    ) -> AllocationResult:
        assignments: Dict[int, Optional[int]] = {r.robot_id: None for r in robots}
        used_robots: set[int] = set()

        for task in sorted(open_tasks, key=lambda t: t.task_id):
            best_robot: Optional[Robot] = None
            best_cost = float("inf")
            for r in robots:
                if r.robot_id in used_robots:
                    continue
                if not r.can_execute(task, planner=planner):
                    continue
                cost = r.task_energy(task, planner=planner)
                if cost < best_cost - 1e-9 or (abs(cost - best_cost) <= 1e-9 and (best_robot is None or r.robot_id < best_robot.robot_id)):
                    best_cost = cost
                    best_robot = r
            if best_robot is not None:
                assignments[best_robot.robot_id] = task.task_id
                used_robots.add(best_robot.robot_id)

        return AllocationResult(
            assignments=assignments,
            sync_steps=1,
            converged=True,
            algorithm_name=self.name,
        )


class OptimalAssignmentAllocator(TaskAllocator):
    """Базовый алгоритм оптимального назначения (венгерский алгоритм): минимизирует суммарную стоимость раунда."""

    name = "optimal"

    def allocate(
        self,
        robots: List[Robot],
        open_tasks: List[Task],
        network: CommunicationNetwork,
        diagonal: float,
        round_idx: int,
        planner: Optional[ObstaclePathPlanner] = None,
        bid_fn: Optional[BidFunction] = None,
        record_history: bool = False,
    ) -> AllocationResult:
        assignments: Dict[int, Optional[int]] = {r.robot_id: None for r in robots}
        if not robots or not open_tasks:
            return AllocationResult(assignments=assignments, sync_steps=0, converged=True, algorithm_name=self.name)

        n = len(robots)
        m = len(open_tasks)
        # Добавляем фиктивные столбцы «без назначения» с большим штрафом, чтобы максимизировать число назначений,
        # а при равном числе назначений минимизировать суммарные энергозатраты.
        big_m = 1e6
        unassigned_penalty = 1e4
        cost_matrix = np.full((n, m + n), big_m, dtype=float)
        feasible = [[False] * m for _ in range(n)]

        for i, r in enumerate(robots):
            cost_matrix[i, m + i] = unassigned_penalty
            for j, t in enumerate(open_tasks):
                if r.can_execute(t, planner=planner):
                    cost_matrix[i, j] = r.task_energy(t, planner=planner)
                    feasible[i][j] = True

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            if j < m and feasible[i][j]:
                assignments[robots[i].robot_id] = open_tasks[j].task_id

        return AllocationResult(
            assignments=assignments,
            sync_steps=1,
            converged=True,
            algorithm_name=self.name,
        )


class DecentralizedAuctionAllocator(TaskAllocator):
    """Децентрализованный аукционный алгоритм CBAA с подключаемой функцией ставки (Задачи 2 и 3)."""

    name = "cbaa"

    def __init__(self, alpha: float = 0.5, beta_load: float = 0.0) -> None:
        self.alpha = alpha
        self.beta_load = beta_load

    def allocate(
        self,
        robots: List[Robot],
        open_tasks: List[Task],
        network: CommunicationNetwork,
        diagonal: float,
        round_idx: int,
        planner: Optional[ObstaclePathPlanner] = None,
        bid_fn: Optional[BidFunction] = None,
        record_history: bool = False,
    ) -> AllocationResult:
        effective_bid = bid_fn or EnergyLoadBidFunction(alpha=self.alpha, beta_load=self.beta_load)
        auction = DecentralizedAuctionRound(
            robots=robots,
            open_tasks=open_tasks,
            network=network,
            diagonal=diagonal,
            alpha=self.alpha,
            round_idx=round_idx,
            planner=planner,
            record_history=record_history,
            bid_fn=effective_bid,
        )
        assignments = auction.run()
        return AllocationResult(
            assignments=assignments,
            sync_steps=auction.sync_steps,
            converged=auction.converged,
            step_history=auction.step_history,
            algorithm_name=self.name,
        )


def build_allocator(name: str, alpha: float = 0.5, beta_load: float = 0.0) -> TaskAllocator:
    """Фабрика создания алгоритма распределения задач по строковому имени."""
    key = name.lower().strip()
    if key == "cbaa":
        return DecentralizedAuctionAllocator(alpha=alpha, beta_load=beta_load)
    if key == "round_robin":
        return RoundRobinAllocator()
    if key == "greedy":
        return GreedyNearestAllocator()
    if key == "optimal":
        return OptimalAssignmentAllocator()
    raise ValueError(f"Неизвестный алгоритм распределения: {name}")
