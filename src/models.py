"""Структуры данных, энергетическая модель и планирование пути с обходом препятствий."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


@dataclass
class Obstacle:
    """Статическое круговое препятствие на 2D-полигоне."""

    x: float
    y: float
    radius: float


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Минимальное расстояние от точки (px, py) до отрезка AB."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def segment_is_clear(
    p1: Point,
    p2: Point,
    obstacles: Sequence[Obstacle],
    safety_margin: float = 0.75,
) -> bool:
    """Проверка, что отрезок p1->p2 не пересекает препятствия с учётом запаса безопасности."""
    ax, ay = p1
    bx, by = p2
    for obs in obstacles:
        dist = _point_to_segment_distance(obs.x, obs.y, ax, ay, bx, by)
        if dist < obs.radius + safety_margin - 1e-6:
            return False
    return True


class ObstaclePathPlanner:
    """Предвычисленный граф видимости с матрицей кратчайших путей обхода препятствий."""

    def __init__(
        self,
        obstacles: Sequence[Obstacle],
        arena_size: float = 100.0,
        safety_margin: float = 0.75,
    ) -> None:
        self.obstacles = list(obstacles)
        self.arena_size = arena_size
        self.safety_margin = safety_margin
        self._nodes: List[Point] = []
        self._dist: List[List[float]] = []
        self._next: List[List[int]] = []
        self._vis_cache: Dict[Point, List[Tuple[int, float]]] = {}
        self._cache: Dict[Tuple[Point, Point], Tuple[List[Point], float]] = {}

        if not self.obstacles:
            return

        n_sides = 12
        scale = 1.0 / math.cos(math.pi / n_sides)
        for obs in self.obstacles:
            ring_r = (obs.radius + safety_margin + 0.15) * scale
            for k in range(n_sides):
                ang = 2.0 * math.pi * k / n_sides
                vx = max(1.0, min(arena_size - 1.0, obs.x + ring_r * math.cos(ang)))
                vy = max(1.0, min(arena_size - 1.0, obs.y + ring_r * math.sin(ang)))
                if all(math.hypot(vx - o.x, vy - o.y) >= o.radius + safety_margin for o in self.obstacles):
                    self._nodes.append((vx, vy))

        m = len(self._nodes)
        inf = float("inf")
        self._dist = [[inf] * m for _ in range(m)]
        self._next = [[-1] * m for _ in range(m)]
        for i in range(m):
            self._dist[i][i] = 0.0
            self._next[i][i] = i
            pi = self._nodes[i]
            for j in range(i + 1, m):
                pj = self._nodes[j]
                if segment_is_clear(pi, pj, self.obstacles, safety_margin=safety_margin):
                    w = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
                    self._dist[i][j] = w
                    self._dist[j][i] = w
                    self._next[i][j] = j
                    self._next[j][i] = i

        # Флойд–Уоршелл по 48 вершинам выполняется за < 1 мс при инициализации сцены
        for k in range(m):
            dk = self._dist[k]
            for i in range(m):
                dik = self._dist[i][k]
                if dik == inf:
                    continue
                di = self._dist[i]
                ni = self._next[i]
                nik = ni[k]
                for j in range(m):
                    cand = dik + dk[j]
                    if cand < di[j]:
                        di[j] = cand
                        ni[j] = nik

    def _visible_nodes(self, pt: Point) -> List[Tuple[int, float]]:
        vis = self._vis_cache.get(pt)
        if vis is not None:
            return vis
        vis = []
        for idx, node in enumerate(self._nodes):
            if segment_is_clear(pt, node, self.obstacles, safety_margin=self.safety_margin):
                vis.append((idx, math.hypot(node[0] - pt[0], node[1] - pt[1])))
        self._vis_cache[pt] = vis
        return vis

    def shortest_path(self, start: Point, goal: Point) -> Tuple[List[Point], float]:
        """Возвращает (список вершин пути, длина пути)."""
        direct_len = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if not self.obstacles or direct_len <= 1e-9:
            return [start, goal], direct_len

        key = (start, goal)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if segment_is_clear(start, goal, self.obstacles, safety_margin=self.safety_margin):
            res = ([start, goal], direct_len)
            self._cache[key] = res
            return res

        vis_s = self._visible_nodes(start)
        vis_g = self._visible_nodes(goal)
        best_len = float("inf")
        best_pair: Optional[Tuple[int, int]] = None
        for i, ds in vis_s:
            di = self._dist[i]
            for j, dg in vis_g:
                cand = ds + di[j] + dg
                if cand < best_len:
                    best_len = cand
                    best_pair = (i, j)

        if best_pair is None:
            res = ([start, goal], direct_len)
            self._cache[key] = res
            return res

        i, j = best_pair
        path: List[Point] = [start, self._nodes[i]]
        cur = i
        while cur != j:
            cur = self._next[cur][j]
            path.append(self._nodes[cur])
        path.append(goal)
        res = (path, best_len)
        self._cache[key] = res
        return res

    def distance(self, start: Point, goal: Point) -> float:
        return self.shortest_path(start, goal)[1]


def plan_obstacle_free_path(
    start: Point,
    goal: Point,
    obstacles: Sequence[Obstacle],
    arena_size: float = 100.0,
    safety_margin: float = 0.75,
) -> List[Point]:
    """Построение кратчайшего кусочно-линейного пути в обход круговых препятствий."""
    if not obstacles or math.hypot(goal[0] - start[0], goal[1] - start[1]) <= 1e-9:
        return [start, goal]
    planner = ObstaclePathPlanner(obstacles, arena_size=arena_size, safety_margin=safety_margin)
    return planner.shortest_path(start, goal)[0]


def polyline_length(points: Sequence[Point]) -> float:
    """Суммарная длина кусочно-линейной траектории."""
    total = 0.0
    for i in range(len(points) - 1):
        total += math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
    return total


@dataclass
class Task:
    """Контрольная точка обследования (задача)."""

    task_id: int
    x: float
    y: float
    inspection_time: float = 4.0       # Длительность обследования, с
    base_energy: float = 5.0           # Базовая энергия обследования h_j, усл. ед.
    completed: bool = False
    assigned_to: Optional[int] = None

    @property
    def pos(self) -> Point:
        return (self.x, self.y)


@dataclass
class Robot:
    """Автономный мобильный робот с энергетической моделью и траекторией."""

    robot_id: int
    x: float
    y: float
    base_x: float
    base_y: float
    energy: float
    initial_energy: float
    reserve_energy: float = 10.0       # Неприкосновенный резерв энергии R_i, усл. ед.
    energy_per_meter: float = 1.0      # Расход энергии на единицу пути (gamma_i)
    inspection_factor: float = 1.0     # Коэффициент энергозатрат на обследование (eta_i)
    speed: float = 6.0                 # Линейная скорость движения, м/с
    state: str = "IDLE"                # IDLE | NEGOTIATING | MOVING | INSPECTING | RETURNING | AT_BASE
    assigned_task_id: Optional[int] = None
    inspect_remaining: float = 0.0
    distance_traveled: float = 0.0
    tasks_completed: int = 0
    trajectory: List[Point] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trajectory:
            self.trajectory.append((self.x, self.y))

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    @property
    def base_pos(self) -> Point:
        return (self.base_x, self.base_y)

    def distance_to(
        self,
        tx: float,
        ty: float,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> float:
        """Евклидово расстояние или длина кратчайшего пути по графу видимости."""
        if planner is None:
            return math.hypot(self.x - tx, self.y - ty)
        return planner.distance((self.x, self.y), (tx, ty))

    def return_energy_from(
        self,
        tx: float,
        ty: float,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> float:
        """Энергия гарантированного возвращения из точки (tx, ty) на базу."""
        if planner is None:
            dist = math.hypot(tx - self.base_x, ty - self.base_y)
        else:
            dist = planner.distance((tx, ty), (self.base_x, self.base_y))
        return self.energy_per_meter * dist

    def task_energy(
        self,
        task: Task,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> float:
        """Суммарная энергия перехода к точке и её обследования."""
        d = self.distance_to(task.x, task.y, planner=planner)
        return self.energy_per_meter * d + task.base_energy * self.inspection_factor

    def can_execute(
        self,
        task: Task,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> bool:
        """Проверка энергетического фильтра допустимости пары «робот–точка»."""
        e_task = self.task_energy(task, planner=planner)
        e_ret = self.return_energy_from(task.x, task.y, planner=planner)
        return (e_task + e_ret + self.reserve_energy) <= self.energy + 1e-9

    def relative_load(
        self,
        task: Task,
        planner: Optional[ObstaclePathPlanner] = None,
    ) -> float:
        """Относительная энергетическая нагрузка rho_ij."""
        avail = self.energy - self.reserve_energy
        if avail <= 1e-9:
            return float("inf")
        e_task = self.task_energy(task, planner=planner)
        e_ret = self.return_energy_from(task.x, task.y, planner=planner)
        return (e_task + e_ret) / avail


@dataclass
class ScenarioConfig:
    """Конфигурация сценария моделирования."""

    arena_size: float = 100.0
    base_pos: Point = (50.0, 50.0)
    num_robots: int = 6
    num_tasks: int = 24
    num_obstacles: int = 4
    topology: str = "ring"             # full | ring | line
    heterogeneous: bool = True         # Однородный или неоднородный энергетический режим
    alpha: float = 0.5                 # Вес расстояния (1.0 = только расстояние, 0.5 = энергия + расстояние)
    seed: int = 42
    dt: float = 0.2                    # Шаг дискретизации времени, с
    allocator_name: str = "cbaa"       # cbaa | round_robin | greedy | optimal
    beta_load: float = 0.0             # Вес загрузки робота (Задача 3)


@dataclass
class FrameSnapshot:
    """Снимок состояния системы для покадровой визуализации и записи видео."""

    sim_time: float
    round_idx: int
    phase: str                         # AUCTION | MOVING | RETURN_BASE | FINISHED
    robots: List[Dict]
    tasks: List[Dict]
    active_links: List[Tuple[int, int]]
    bid_annotations: List[Dict]
    messages_total: int
    bytes_total: int
    completed_count: int
    total_tasks: int
    phase_detail: str = ""
    contested_tasks: List[int] = field(default_factory=list)
    round_summary: List[str] = field(default_factory=list)
    ledger_tx: str = ""
    allocator_name: str = "cbaa"
