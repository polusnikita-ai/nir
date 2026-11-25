"""Открытые программные интерфейсы расширения имитационной модели МРС.

Обеспечивают совместимость ядра симулятора (Задача 1А) со смежными модулями:
- Задача 1Б: базовые алгоритмы MRTA (Round Robin, Greedy / Nearest, Оптимальное назначение);
- Задача 2: децентрализованные аукционные алгоритмы (CBAA / CBBA);
- Задача 3: функции формирования ставок с учётом энергии, расстояния и загрузки роботов;
- Задача 4: регистрация согласованных назначений в смарт-контракте / блокчейне через API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import urllib.request

from .models import ObstaclePathPlanner, Robot, Task
from .network import CommunicationNetwork


@dataclass
class AllocationResult:
    """Унифицированный результат одного раунда распределения задач."""

    assignments: Dict[int, Optional[int]]
    sync_steps: int = 0
    converged: bool = True
    step_history: List[Dict] = field(default_factory=list)
    algorithm_name: str = ""


@dataclass
class LedgerReceipt:
    """Квитанция регистрации согласованного назначения раунда в реестре / смарт-контракте."""

    round_idx: int
    tx_hash: str
    backend: str
    timestamp: float
    assignments_count: int
    payload_hash: str


class BidFunction(ABC):
    """Абстрактный интерфейс функции расчёта ставки (интеграция с Задачей 3)."""

    name: str = "base"

    @abstractmethod
    def compute_bid(
        self,
        robot: Robot,
        task: Task,
        diagonal: float,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> Optional[int]:
        """Возвращает целочисленный код ставки (>= 0) либо None, если пара недопустима."""


class EnergyLoadBidFunction(BidFunction):
    """Комбинированная ставка с учётом расстояния, относительной энергонагрузки и загрузки робота."""

    def __init__(self, alpha: float = 0.5, beta_load: float = 0.0) -> None:
        self.alpha = alpha
        self.beta_load = beta_load
        self.name = f"energy_load(alpha={alpha:.2f},beta={beta_load:.2f})"

    def compute_bid(
        self,
        robot: Robot,
        task: Task,
        diagonal: float,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> Optional[int]:
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
        load_penalty = self.beta_load * (robot.tasks_completed / (1.0 + robot.tasks_completed))
        score = 1.0 / (1.0 + self.alpha * (d / max(diagonal, 1e-9)) + (1.0 - self.alpha) * rho + load_penalty)
        return int(math.floor(1_000_000 * score + 0.5))


class TaskAllocator(ABC):
    """Абстрактный интерфейс алгоритма распределения задач (интеграция с Задачами 1Б и 2)."""

    name: str = "base_allocator"

    @abstractmethod
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
        """Выполняет распределение открытых задач между роботами на один раунд."""


class AssignmentLedger(ABC):
    """Абстрактный интерфейс распределённого реестра / смарт-контракта (интеграция с Задачей 4)."""

    @abstractmethod
    def register_round(
        self,
        round_idx: int,
        assignments: Dict[int, Optional[int]],
        metadata: Optional[Dict] = None,
    ) -> LedgerReceipt:
        """Сохраняет согласованное назначение раунда и возвращает квитанцию."""

    @abstractmethod
    def get_records(self) -> List[Dict]:
        """Возвращает журнал всех зарегистрированных раундов."""


class InMemoryLedger(AssignmentLedger):
    """Локальная имитация реестра смарт-контракта с вычислением SHA-256 хэша каждого раунда."""

    def __init__(self) -> None:
        self._records: List[Dict] = []

    def register_round(
        self,
        round_idx: int,
        assignments: Dict[int, Optional[int]],
        metadata: Optional[Dict] = None,
    ) -> LedgerReceipt:
        prev_hash = self._records[-1]["tx_hash"] if self._records else "0" * 16
        payload = {
            "round": round_idx,
            "assignments": {str(k): v for k, v in sorted(assignments.items()) if v is not None},
            "metadata": metadata or {},
            "prev_hash": prev_hash,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        tx_hash = "0x" + digest[:16]
        receipt = LedgerReceipt(
            round_idx=round_idx,
            tx_hash=tx_hash,
            backend="in_memory_smart_contract",
            timestamp=round(time.time(), 3),
            assignments_count=len(payload["assignments"]),
            payload_hash=digest,
        )
        self._records.append(
            {
                "round": round_idx,
                "tx_hash": tx_hash,
                "payload_hash": digest,
                "assignments": payload["assignments"],
                "metadata": payload["metadata"],
            }
        )
        return receipt

    def get_records(self) -> List[Dict]:
        return list(self._records)


class HttpSmartContractLedger(AssignmentLedger):
    """Адаптер HTTP API для подключения внешнего смарт-контракта (Задача 4).

    При отсутствии сети или в режиме dry_run автоматически фиксирует локальный хэш
    и сохраняет подготовленный JSON-пакет для отправки на endpoint смарт-контракта.
    """

    def __init__(self, endpoint_url: str = "", timeout_sec: float = 3.0, dry_run: bool = True) -> None:
        self.endpoint_url = endpoint_url
        self.timeout_sec = timeout_sec
        self.dry_run = dry_run
        self._fallback = InMemoryLedger()

    def register_round(
        self,
        round_idx: int,
        assignments: Dict[int, Optional[int]],
        metadata: Optional[Dict] = None,
    ) -> LedgerReceipt:
        receipt = self._fallback.register_round(round_idx, assignments, metadata=metadata)
        if not self.dry_run and self.endpoint_url:
            body = json.dumps(self._fallback.get_records()[-1], ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                resp.read()
            receipt.backend = f"http_smart_contract({self.endpoint_url})"
        else:
            receipt.backend = "http_smart_contract_stub"
        return receipt

    def get_records(self) -> List[Dict]:
        return self._fallback.get_records()
