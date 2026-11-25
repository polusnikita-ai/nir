"""Модульные тесты проверки инвариантов имитационной модели и протокола согласования."""

import math
import unittest

from src.allocator import (
    BidEntry,
    DecentralizedAuctionRound,
    brute_force_best_round,
    compute_bid_code,
    is_better_bid,
)
from src.models import Obstacle, ObstaclePathPlanner, Robot, ScenarioConfig, Task
from src.network import CommunicationNetwork
from src.interfaces import EnergyLoadBidFunction, HttpSmartContractLedger, InMemoryLedger
from src.simulator import MultiRobotSimulator


class TestMultiRobotSimulation(unittest.TestCase):
    def test_energy_filter_and_bid_change(self) -> None:
        """Проверка расчётного примера смены исполнителя при учёте энергетической нагрузки."""
        r1 = Robot(robot_id=0, x=8.0, y=0.0, base_x=0.0, base_y=0.0, energy=20.0, initial_energy=20.0, reserve_energy=2.0)
        r2 = Robot(robot_id=1, x=5.0, y=0.0, base_x=0.0, base_y=0.0, energy=40.0, initial_energy=40.0, reserve_energy=2.0)
        task = Task(task_id=0, x=10.0, y=0.0, base_energy=4.0)

        self.assertTrue(r1.can_execute(task))
        self.assertTrue(r2.can_execute(task))

        # При alpha = 1.0 (только расстояние) выигрывает ближайший робот R0
        b1_dist = compute_bid_code(r1, task, diagonal=20.0, alpha=1.0)
        b2_dist = compute_bid_code(r2, task, diagonal=20.0, alpha=1.0)
        self.assertIsNotNone(b1_dist)
        self.assertIsNotNone(b2_dist)
        self.assertGreater(b1_dist, b2_dist)

        # При alpha = 0.5 (расстояние + энергия) выигрывает R1 с большим запасом энергии
        b1_en = compute_bid_code(r1, task, diagonal=20.0, alpha=0.5)
        b2_en = compute_bid_code(r2, task, diagonal=20.0, alpha=0.5)
        self.assertGreater(b2_en, b1_en)

    def test_exact_energy_boundary_and_low_energy_block(self) -> None:
        """Проверка точного равенства запаса/потребности и блокировки при E <= R."""
        # d=2, h=4, ret=10, reserve=2 -> нужно ровно 18.0
        t = Task(task_id=0, x=10.0, y=0.0, base_energy=4.0)
        r_exact = Robot(robot_id=0, x=8.0, y=0.0, base_x=0.0, base_y=0.0, energy=18.0, initial_energy=18.0, reserve_energy=2.0)
        r_short = Robot(robot_id=1, x=8.0, y=0.0, base_x=0.0, base_y=0.0, energy=17.99, initial_energy=17.99, reserve_energy=2.0)
        r_below_res = Robot(robot_id=2, x=50.0, y=50.0, base_x=50.0, base_y=50.0, energy=9.0, initial_energy=9.0, reserve_energy=10.0)

        self.assertTrue(r_exact.can_execute(t))
        self.assertIsNotNone(compute_bid_code(r_exact, t, diagonal=20.0, alpha=0.5))
        self.assertFalse(r_short.can_execute(t))
        self.assertIsNone(compute_bid_code(r_short, t, diagonal=20.0, alpha=0.5))
        self.assertFalse(r_below_res.can_execute(t))
        self.assertIsNone(compute_bid_code(r_below_res, t, diagonal=100.0, alpha=0.5))

    def test_tie_breaking_and_rounding(self) -> None:
        """При равных кодах ставок выигрывает меньший robot_id, затем меньший task_id."""
        self.assertTrue(is_better_bid(BidEntry(500_000, 0), BidEntry(500_000, 2)))
        self.assertFalse(is_better_bid(BidEntry(500_000, 2), BidEntry(500_000, 0)))

        # Два идентичных робота (id=0 и id=1) на одной позиции, две симметричные задачи (id=0 и id=1)
        robots = [
            Robot(0, 50.0, 50.0, 50.0, 50.0, 100.0, 100.0),
            Robot(1, 50.0, 50.0, 50.0, 50.0, 100.0, 100.0),
        ]
        tasks = [
            Task(0, 60.0, 50.0, base_energy=4.0),
            Task(1, 40.0, 50.0, base_energy=4.0),
        ]
        net = CommunicationNetwork(2, "line")
        rnd = DecentralizedAuctionRound(robots, tasks, net, diagonal=100.0, alpha=0.5, round_idx=1)
        assign = rnd.run()
        self.assertTrue(rnd.converged)
        self.assertEqual(assign, {0: 0, 1: 1})

        # Граничный тест округления: d=0, rho=0.5, alpha=0 -> score=2/3=0.6666666... -> 666667
        r = Robot(0, 10.0, 0.0, 0.0, 0.0, 30.0, 30.0, reserve_energy=2.0)
        t_same = Task(0, 10.0, 0.0, base_energy=4.0)  # e_task=4, ret=10, avail=28 -> rho=0.5 -> s=2/3
        code = compute_bid_code(r, t_same, diagonal=20.0, alpha=0.0)
        self.assertEqual(code, 666667)

    def test_empty_and_unreachable_tasks(self) -> None:
        """При пустом списке задач или недостижимых целях раунд сходится с пустым назначением."""
        robots = [Robot(0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, reserve_energy=10.0)]
        net = CommunicationNetwork(1, "line")
        rnd_empty = DecentralizedAuctionRound(robots, [], net, diagonal=100.0, alpha=0.5, round_idx=1)
        self.assertEqual(rnd_empty.run(), {0: None})
        self.assertTrue(rnd_empty.converged)

        far_task = [Task(0, 95.0, 95.0, base_energy=5.0)]
        rnd_far = DecentralizedAuctionRound(robots, far_task, net, diagonal=100.0, alpha=0.5, round_idx=1)
        self.assertEqual(rnd_far.run(), {0: None})
        self.assertTrue(rnd_far.converged)

    def test_idle_relay_robot_in_line_topology(self) -> None:
        """Робот без доступной энергии участвует в ретрансляции ставок и согласовании флагов."""
        # Топология line: R0 — R1(без запаса) — R2. Оба R0 и R2 претендуют на одну задачу T0.
        robots = [
            Robot(0, 40.0, 50.0, 50.0, 50.0, 100.0, 100.0, reserve_energy=10.0),
            Robot(1, 50.0, 50.0, 50.0, 50.0, 10.0, 10.0, reserve_energy=10.0),  # Не может делать ставки!
            Robot(2, 58.0, 50.0, 50.0, 50.0, 120.0, 120.0, reserve_energy=10.0),
        ]
        tasks = [Task(0, 60.0, 50.0, base_energy=3.0)]
        net = CommunicationNetwork(3, "line", log_messages=True)
        rnd = DecentralizedAuctionRound(robots, tasks, net, diagonal=100.0, alpha=0.5, round_idx=1)
        assign = rnd.run()
        self.assertTrue(rnd.converged)
        # R2 ближе и богаче по энергии — должен выиграть через ретрансляцию через R1
        self.assertEqual(assign, {0: None, 1: None, 2: 0})
        self.assertTrue(any(m["sender"] == 1 for m in net.message_log))

    def test_consensus_same_across_topologies(self) -> None:
        """На одном раунде все связные топологии дают одинаковое бесконфликтное назначение."""
        results = {}
        for topo in ("full", "ring", "line"):
            robots = [
                Robot(i, 40.0 + 5.0 * i, 50.0, 50.0, 50.0, 150.0, 150.0)
                for i in range(4)
            ]
            tasks = [Task(j, 20.0 + 20.0 * j, 60.0) for j in range(4)]
            net = CommunicationNetwork(4, topo)
            rnd = DecentralizedAuctionRound(robots, tasks, net, diagonal=141.42, alpha=0.5, round_idx=1)
            results[topo] = rnd.run()
            self.assertTrue(rnd.converged)

        self.assertEqual(results["full"], results["ring"])
        self.assertEqual(results["ring"], results["line"])

    def test_compare_with_brute_force_small_round(self) -> None:
        """Сверка согласованного раунда с полным перебором допустимых назначений."""
        robots = [
            Robot(0, 42.0, 50.0, 50.0, 50.0, 120.0, 120.0),
            Robot(1, 55.0, 48.0, 50.0, 50.0, 140.0, 140.0),
            Robot(2, 50.0, 60.0, 50.0, 50.0, 110.0, 110.0),
        ]
        tasks = [
            Task(0, 30.0, 50.0, base_energy=3.0),
            Task(1, 70.0, 52.0, base_energy=4.0),
            Task(2, 52.0, 80.0, base_energy=5.0),
        ]
        net = CommunicationNetwork(3, "ring")
        rnd = DecentralizedAuctionRound(robots, tasks, net, diagonal=141.42, alpha=0.5, round_idx=1)
        assign = rnd.run()
        self.assertTrue(rnd.converged)

        bf_key, bf_assign = brute_force_best_round(robots, tasks, diagonal=141.42, alpha=0.5)
        self.assertEqual(sum(1 for v in assign.values() if v is not None), bf_key[0])
        self.assertEqual(assign, bf_assign)

    def test_obstacle_planner_avoids_circle(self) -> None:
        """Планировщик пути строит обход кругового препятствия без его пересечения."""
        obs = [Obstacle(50.0, 50.0, 8.0)]
        planner = ObstaclePathPlanner(obs, arena_size=100.0, safety_margin=0.75)
        path, length = planner.shortest_path((20.0, 50.0), (80.0, 50.0))
        self.assertGreater(len(path), 2)
        self.assertGreater(length, 60.0)
        for pt in path:
            self.assertGreaterEqual(math.hypot(pt[0] - 50.0, pt[1] - 50.0), 8.0)

    def test_full_mission_invariants_and_reproducibility(self) -> None:
        """Проверка отсутствия нарушений, коллизий, точного энергобаланса и воспроизводимости."""
        cfg = ScenarioConfig(num_robots=5, num_tasks=18, topology="ring", alpha=0.5, seed=77, dt=0.5)
        sim1 = MultiRobotSimulator(cfg, log_messages=True)
        m1 = sim1.run()
        m2 = MultiRobotSimulator(cfg).run()

        self.assertEqual(m1["violations"], 0)
        self.assertEqual(m1["obstacle_penetrations"], 0)
        self.assertEqual(m1["unconverged_rounds"], 0)
        self.assertGreaterEqual(m1["min_final_energy"], 9.99)
        self.assertEqual(m1["completed_tasks"], m2["completed_tasks"])
        self.assertEqual(m1["messages_sent"], m2["messages_sent"])
        self.assertEqual(m1["total_energy_spent"], m2["total_energy_spent"])
        self.assertEqual(len(sim1.network.message_log), m1["messages_sent"])



    def test_pluggable_allocators_compatibility(self) -> None:
        """Проверка совместимости ядра со всеми 4 алгоритмами (Round Robin, Greedy, Optimal, CBAA)."""
        coverages = {}
        for alloc in ("round_robin", "greedy", "optimal", "cbaa"):
            cfg = ScenarioConfig(
                num_robots=5,
                num_tasks=18,
                topology="ring",
                alpha=0.5,
                allocator_name=alloc,
                seed=77,
                dt=0.5,
            )
            m = MultiRobotSimulator(cfg).run()
            self.assertEqual(m["violations"], 0, msg=f"Violations in {alloc}")
            self.assertEqual(m["obstacle_penetrations"], 0)
            self.assertGreater(m["completed_tasks"], 0)
            coverages[alloc] = m["coverage"]
        # Оптимальное и жадное назначения не должны быть хуже пустого результата
        self.assertGreater(coverages["optimal"], 0.5)

    def test_pluggable_energy_load_bid_function(self) -> None:
        """Проверка подключаемой функции ставки с учётом загрузки робота (Задача 3)."""
        r_idle = Robot(0, 50.0, 50.0, 50.0, 50.0, 120.0, 120.0, tasks_completed=0)
        r_busy = Robot(1, 50.0, 50.0, 50.0, 50.0, 120.0, 120.0, tasks_completed=5)
        t = Task(0, 60.0, 50.0, base_energy=4.0)
        bid_fn = EnergyLoadBidFunction(alpha=0.5, beta_load=0.3)
        b_idle = bid_fn.compute_bid(r_idle, t, diagonal=141.42)
        b_busy = bid_fn.compute_bid(r_busy, t, diagonal=141.42)
        self.assertIsNotNone(b_idle)
        self.assertIsNotNone(b_busy)
        self.assertGreater(b_idle, b_busy)

    def test_smart_contract_ledger_registration(self) -> None:
        """Проверка регистрации раундов в реестре смарт-контракта с цепочкой SHA-256 (Задача 4)."""
        ledger = HttpSmartContractLedger(endpoint_url="http://localhost:8545/api/register", dry_run=True)
        cfg = ScenarioConfig(num_robots=4, num_tasks=10, topology="ring", alpha=0.5, seed=11, dt=0.5)
        sim = MultiRobotSimulator(cfg, ledger=ledger)
        m = sim.run()
        records = ledger.get_records()
        self.assertEqual(m["violations"], 0)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(r["tx_hash"].startswith("0x") for r in records))

if __name__ == "__main__":
    unittest.main()
