"""Модель коммуникационной сети и учёта трафика многороботной системы."""

from __future__ import annotations

from typing import Dict, List, Tuple


class CommunicationNetwork:
    """Статический связный граф связи G = (R, E) с подсчётом сообщений, объёма данных и журналом."""

    def __init__(self, num_robots: int, topology: str = "ring", log_messages: bool = False) -> None:
        if num_robots < 1:
            raise ValueError("Число роботов должно быть не менее 1.")
        self.num_robots = num_robots
        self.topology = topology
        self.log_messages = log_messages
        self.adj: Dict[int, List[int]] = {i: [] for i in range(num_robots)}
        self.edges: List[Tuple[int, int]] = []
        self.messages_sent: int = 0
        self.bytes_sent: int = 0
        self.message_log: List[Dict] = []
        self._build_topology()

    def _build_topology(self) -> None:
        n = self.num_robots
        if n <= 1:
            return
        if self.topology == "full":
            for i in range(n):
                for j in range(i + 1, n):
                    self._add_edge(i, j)
        elif self.topology == "ring":
            for i in range(n - 1):
                self._add_edge(i, i + 1)
            if n > 2:
                self._add_edge(n - 1, 0)
        elif self.topology == "line":
            for i in range(n - 1):
                self._add_edge(i, i + 1)
        else:
            raise ValueError(f"Неизвестная топология сети: {self.topology}")

    def _add_edge(self, u: int, v: int) -> None:
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)
        edge = (min(u, v), max(u, v))
        if edge not in self.edges:
            self.edges.append(edge)

    @property
    def diameter_upper_bound(self) -> int:
        """Точный диаметр (верхняя оценка числа пересылок) связной топологии."""
        if self.num_robots <= 1:
            return 0
        if self.topology == "full":
            return 1
        if self.topology == "ring":
            return self.num_robots // 2
        return self.num_robots - 1

    def neighbors(self, robot_id: int) -> List[int]:
        return self.adj[robot_id]

    def record_delivery(
        self,
        sender: int,
        receiver: int,
        payload_bytes: int,
        round_idx: int = 0,
        step: int = 0,
        msg_type: str = "BID_TABLE",
    ) -> None:
        """Фиксация одной направленной доставки сообщения соседу."""
        self.messages_sent += 1
        self.bytes_sent += payload_bytes
        if self.log_messages:
            self.message_log.append(
                {
                    "round": round_idx,
                    "sync_step": step,
                    "msg_type": msg_type,
                    "sender": sender,
                    "receiver": receiver,
                    "bytes": payload_bytes,
                }
            )
