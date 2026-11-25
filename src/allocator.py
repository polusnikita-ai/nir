"""Децентрализованный аукционный алгоритм согласования задач (Energy-Aware CBAA)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .models import ObstaclePathPlanner, Robot, Task
from .network import CommunicationNetwork


@dataclass
class BidEntry:
    """Запись о лучшей известной ставке на задачу."""

    score_code: int      # Целочисленный код ставки floor(10^6 * s + 0.5)
    winner_id: int       # Идентификатор робота-победителя (-1, если нет)


def is_better_bid(candidate: BidEntry, current: BidEntry) -> bool:
    """Единое детерминированное правило сравнения ставок (при равенстве — меньший ID)."""
    if candidate.winner_id == -1:
        return False
    if current.winner_id == -1:
        return True
    if candidate.score_code != current.score_code:
        return candidate.score_code > current.score_code
    return candidate.winner_id < current.winner_id


def compute_bid_code(
    robot: Robot,
    task: Task,
    diagonal: float,
    alpha: float,
    planner: Optional[ObstaclePathPlanner] = None,
) -> Optional[int]:
    """Вычисление целочисленного кода ставки с учётом расстояния и относительной энергонагрузки."""
    avail = robot.energy - robot.reserve_energy
    if avail <= 1e-9:
        return None
    d = robot.distance_to(task.x, task.y, planner=planner)
    e_task = robot.energy_per_meter * d + task.base_energy * robot.inspection_factor
    e_ret = robot.return_energy_from(task.x, task.y, planner=planner)
    total_need = e_task + e_ret
    if total_need > avail + 1e-9:
        return None
    rho = total_need / avail
    score = 1.0 / (1.0 + alpha * (d / diagonal) + (1.0 - alpha) * rho)
    return int(math.floor(1_000_000 * score + 0.5))


class DecentralizedAuctionRound:
    """Выполнение одного децентрализованного раунда согласования CBAA по графу связи."""

    def __init__(
        self,
        robots: List[Robot],
        open_tasks: List[Task],
        network: CommunicationNetwork,
        diagonal: float,
        alpha: float,
        round_idx: int,
        planner: Optional[ObstaclePathPlanner] = None,
        record_history: bool = False,
        bid_fn=None,
    ) -> None:
        self.robots = robots
        self.open_tasks = open_tasks
        self.task_map = {t.task_id: t for t in open_tasks}
        self.network = network
        self.diagonal = diagonal
        self.alpha = alpha
        self.round_idx = round_idx
        self.planner = planner
        self.record_history = record_history
        self.bid_fn = bid_fn

        # Локальные таблицы лучших ставок у каждого робота
        self.tables: Dict[int, Dict[int, BidEntry]] = {
            r.robot_id: {t.task_id: BidEntry(0, -1) for t in open_tasks}
            for r in robots
        }
        self.local_choice: Dict[int, Optional[int]] = {r.robot_id: None for r in robots}
        self.local_bids: Dict[int, Dict[int, int]] = {}
        for r in robots:
            bids_r: Dict[int, int] = {}
            for t in open_tasks:
                if self.bid_fn is not None:
                    code = self.bid_fn.compute_bid(r, t, diagonal, planner=self.planner)
                else:
                    code = compute_bid_code(r, t, diagonal, alpha, planner=self.planner)
                if code is not None:
                    bids_r[t.task_id] = code
            self.local_bids[r.robot_id] = bids_r

        self.sync_steps: int = 0
        self.converged: bool = False
        self.step_history: List[Dict] = []

    def _snapshot_bids(
        self,
        phase_detail: str,
        active_links: List[Tuple[int, int]],
        contested: Optional[List[int]] = None,
    ) -> None:
        if not self.record_history:
            return
        annotations: List[Dict] = []
        summary: List[str] = []
        for r in self.robots:
            rid = r.robot_id
            tid = self.local_choice[rid]
            if tid is not None:
                code = self.local_bids[rid].get(tid, 0)
                annotations.append(
                    {
                        "robot_id": rid,
                        "task_id": tid,
                        "score_code": code,
                        "score": code / 1_000_000.0,
                    }
                )
                summary.append(f"R{rid} → T{tid} (s={code / 1_000_000.0:.3f})")
            else:
                reason = "нет допуст." if not self.local_bids[rid] else "перебита"
                summary.append(f"R{rid} → — ({reason})")
        self.step_history.append(
            {
                "step": self.sync_steps,
                "phase_detail": phase_detail,
                "active_links": list(active_links),
                "choices": dict(self.local_choice),
                "bid_annotations": annotations,
                "contested_tasks": list(contested or []),
                "round_summary": summary,
            }
        )

    def _local_select(self) -> Dict[int, bool]:
        """Локальный выбор допустимой задачи роботами без текущего назначения."""
        agent_changed: Dict[int, bool] = {r.robot_id: False for r in self.robots}
        for r in self.robots:
            rid = r.robot_id
            if self.local_choice[rid] is not None:
                continue
            best_task_id: Optional[int] = None
            best_entry = BidEntry(0, -1)
            for tid, code in sorted(self.local_bids[rid].items()):
                cand = BidEntry(code, rid)
                cur = self.tables[rid][tid]
                if is_better_bid(cand, cur):
                    if (
                        best_task_id is None
                        or code > best_entry.score_code
                        or (code == best_entry.score_code and tid < best_task_id)
                    ):
                        best_task_id = tid
                        best_entry = cand
            if best_task_id is not None:
                self.local_choice[rid] = best_task_id
                self.tables[rid][best_task_id] = best_entry
                agent_changed[rid] = True

        if self.record_history and any(agent_changed.values()):
            counts = Counter(tid for tid in self.local_choice.values() if tid is not None)
            contested = sorted(tid for tid, cnt in counts.items() if cnt > 1)
            detail = (
                "ЛОКАЛЬНЫЙ ВЫБОР (ОБНАРУЖЕН КОНФЛИКТ СТАВОК)"
                if contested
                else "ЛОКАЛЬНЫЙ ВЫБОР ЗАДАЧ И РАСЧЁТ СТАВОК"
            )
            self._snapshot_bids(detail, active_links=[], contested=contested)
        return agent_changed

    def _exchange_tables_once(self, agent_changed: Dict[int, bool]) -> None:
        """Один синхронный шаг обмена таблицами ставок между соседями по рёбрам графа."""
        self.sync_steps += 1
        outgoing: Dict[int, Dict[int, BidEntry]] = {
            r.robot_id: dict(self.tables[r.robot_id]) for r in self.robots
        }
        active_edges: List[Tuple[int, int]] = []
        # Размер сериализованного сообщения: заголовок 16 байт + 12 байт на каждую запись
        payload_bytes = 16 + 12 * len(self.open_tasks)

        for r in self.robots:
            u = r.robot_id
            for v in self.network.neighbors(u):
                self.network.record_delivery(
                    u,
                    v,
                    payload_bytes,
                    round_idx=self.round_idx,
                    step=self.sync_steps,
                    msg_type="BID_TABLE",
                )
                edge = (min(u, v), max(u, v))
                if edge not in active_edges:
                    active_edges.append(edge)

        outbid_tasks: List[int] = []
        for r in self.robots:
            rid = r.robot_id
            for nbr in self.network.neighbors(rid):
                nbr_table = outgoing[nbr]
                for tid, incoming in nbr_table.items():
                    if is_better_bid(incoming, self.tables[rid][tid]):
                        self.tables[rid][tid] = incoming
                        agent_changed[rid] = True
            # Проверка, не перебита ли собственная выбранная задача
            chosen = self.local_choice[rid]
            if chosen is not None and self.tables[rid][chosen].winner_id != rid:
                outbid_tasks.append(chosen)
                self.local_choice[rid] = None
                agent_changed[rid] = True

        if self.record_history:
            counts = Counter(tid for tid in self.local_choice.values() if tid is not None)
            contested = sorted(set(outbid_tasks) | {tid for tid, cnt in counts.items() if cnt > 1})
            detail = (
                "ОБМЕН ПО СЕТИ: КОНФЛИКТ РАЗРЕШЁН, ПЕРЕСМОТР СТАВКИ"
                if outbid_tasks
                else "ОБМЕН ТАБЛИЦАМИ СТАВОК С СОСЕДЯМИ"
            )
            self._snapshot_bids(detail, active_links=active_edges, contested=contested)

    def run(self, max_outer_iters: int = 50) -> Dict[int, Optional[int]]:
        """Запуск полного раунда согласования до подтверждённой стабильности по сети."""
        if not self.open_tasks:
            self.converged = True
            return dict(self.local_choice)

        d_steps = max(1, self.network.diameter_upper_bound)
        for _ in range(max_outer_iters):
            agent_changed = self._local_select()

            # Фаза 1: распространение таблиц в течение D шагов
            for _ in range(d_steps):
                self._exchange_tables_once(agent_changed)

            # Фаза 2: децентрализованное распространение флагов стабильности (логическое ИЛИ за D шагов)
            flags: Dict[int, bool] = dict(agent_changed)
            flag_payload = 12  # round_idx + flag + sender_id
            for _ in range(d_steps):
                self.sync_steps += 1
                outgoing_flags = dict(flags)
                for r in self.robots:
                    u = r.robot_id
                    for v in self.network.neighbors(u):
                        self.network.record_delivery(
                            u,
                            v,
                            flag_payload,
                            round_idx=self.round_idx,
                            step=self.sync_steps,
                            msg_type="STABILITY_FLAG",
                        )
                for r in self.robots:
                    rid = r.robot_id
                    for nbr in self.network.neighbors(rid):
                        if outgoing_flags[nbr]:
                            flags[rid] = True

            # Каждый агент локально проверяет объединённый флаг стабильности
            if all(not flags[r.robot_id] for r in self.robots):
                self.converged = True
                if self.record_history:
                    self._snapshot_bids(
                        "КОНСЕНСУС ПОДТВЕРЖДЁН (БЕЗ ДИСПЕТЧЕРА)",
                        active_links=self.network.edges,
                        contested=[],
                    )
                break

        return dict(self.local_choice)


def brute_force_best_round(
    robots: List[Robot],
    open_tasks: List[Task],
    diagonal: float,
    alpha: float,
    planner: Optional[ObstaclePathPlanner] = None,
) -> Tuple[Tuple[int, int], Dict[int, Optional[int]]]:
    """Полный перебор допустимых назначений за один раунд для верификации небольших сценариев."""
    n = len(robots)
    feasible_bids: List[List[Tuple[int, int]]] = []
    for r in robots:
        opts: List[Tuple[int, int]] = []
        for t in open_tasks:
            code = compute_bid_code(r, t, diagonal, alpha, planner=planner)
            if code is not None:
                opts.append((t.task_id, code))
        feasible_bids.append(opts)

    best_key = (-1, -1)
    best_assign: Dict[int, Optional[int]] = {r.robot_id: None for r in robots}
    used_tasks: set[int] = set()
    cur_assign: Dict[int, Optional[int]] = {r.robot_id: None for r in robots}

    def dfs(idx: int, cnt: int, total_score: int) -> None:
        nonlocal best_key, best_assign
        # Отсечение по максимально возможному числу оставшихся назначений
        if cnt + (n - idx) < best_key[0]:
            return
        if idx == n:
            key = (cnt, total_score)
            if key > best_key:
                best_key = key
                best_assign = dict(cur_assign)
            return
        rid = robots[idx].robot_id
        for tid, code in feasible_bids[idx]:
            if tid in used_tasks:
                continue
            used_tasks.add(tid)
            cur_assign[rid] = tid
            dfs(idx + 1, cnt + 1, total_score + code)
            cur_assign[rid] = None
            used_tasks.remove(tid)
        # Вариант без назначения для данного робота
        dfs(idx + 1, cnt, total_score)

    dfs(0, 0, 0)
    return best_key, best_assign
