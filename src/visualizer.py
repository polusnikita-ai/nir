"""Визуализация симуляции (25-секундное видео MP4) и построение графиков экспериментов."""

from __future__ import annotations

import math
import os
if os.name == "nt":
    import ctypes
    from ctypes import POINTER, byref, c_int64, c_uint32, c_uint64, c_void_p, wintypes
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from .models import FrameSnapshot, ScenarioConfig
from .simulator import MultiRobotSimulator


ROBOT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#0891b2", "#e377c2", "#8c564b"]


if os.name == "nt":
    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        @classmethod
        def from_str(cls, s: str) -> "_GUID":
            s = s.strip("{}")
            parts = s.split("-")
            b4 = bytes.fromhex(parts[3] + parts[4])
            return cls(int(parts[0], 16), int(parts[1], 16), int(parts[2], 16), (ctypes.c_ubyte * 8)(*b4))


    def _mf_call(obj: c_void_p, vtable_idx: int, restype, *argtypes_and_args):
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        fn_addr = vtbl[vtable_idx]
        argtypes = [a[0] for a in argtypes_and_args]
        args = [a[1] for a in argtypes_and_args]
        proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
        return proto(fn_addr)(obj, *args)


    class _MediaFoundationMP4Writer:
        """Нативный кодер H.264 MP4 через Windows Media Foundation (без внешнего ffmpeg)."""

        def __init__(self, output_path: str, width: int, height: int, fps: int, bitrate: int = 2_200_000) -> None:
            self.output_path = os.path.abspath(output_path)
            self.width = width
            self.height = height
            self.fps = fps
            self.bitrate = bitrate
            self.frame_idx = 0
            self.dur_100ns = int(10_000_000 // fps)

            self.ole32 = ctypes.OleDLL("ole32")
            self.mfplat = ctypes.WinDLL("mfplat")
            self.mfreadwrite = ctypes.WinDLL("mfreadwrite")
            self.ole32.CoInitializeEx(None, 0x2)
            self.mfplat.MFStartup(0x00020070, 0)

            self.sink_writer = c_void_p()
            self.mfreadwrite.MFCreateSinkWriterFromURL.argtypes = [
                wintypes.LPCWSTR, c_void_p, c_void_p, POINTER(c_void_p)
            ]
            self.mfreadwrite.MFCreateSinkWriterFromURL.restype = ctypes.HRESULT
            self.mfreadwrite.MFCreateSinkWriterFromURL(self.output_path, None, None, byref(self.sink_writer))

            mt_video = _GUID.from_str("{73646976-0000-0010-8000-00AA00389B71}")
            mt_h264 = _GUID.from_str("{34363248-0000-0010-8000-00AA00389B71}")
            mt_rgb32 = _GUID.from_str("{00000016-0000-0010-8000-00AA00389B71}")
            k_major = _GUID.from_str("{48eba18e-f8c9-4687-bf11-0a74c9f96a8f}")
            k_sub = _GUID.from_str("{f7e34c9a-42e8-4714-b74b-cb29d72c35e5}")
            k_bitrate = _GUID.from_str("{20332624-fb0d-4d9e-bd0d-cbf6786c102e}")
            k_interlace = _GUID.from_str("{e2724bb8-e676-4806-b4b2-a8d6efb44ccd}")
            k_size = _GUID.from_str("{1652c33d-d6b2-4012-b834-72030849a37d}")
            k_rate = _GUID.from_str("{c459a2e8-3d2c-4e44-b132-fee5156c7bb0}")
            k_par = _GUID.from_str("{c6376a1e-8d0a-4027-be45-6d9a0ad39bb6}")
            k_stride = _GUID.from_str("{644b4e48-1e02-4516-b0eb-c01ca9d49ac6}")

            def pack64(hi: int, lo: int) -> int:
                return (hi << 32) | lo

            mt_out = c_void_p()
            self.mfplat.MFCreateMediaType(byref(mt_out))
            _mf_call(mt_out, 24, ctypes.HRESULT, (POINTER(_GUID), byref(k_major)), (POINTER(_GUID), byref(mt_video)))
            _mf_call(mt_out, 24, ctypes.HRESULT, (POINTER(_GUID), byref(k_sub)), (POINTER(_GUID), byref(mt_h264)))
            _mf_call(mt_out, 21, ctypes.HRESULT, (POINTER(_GUID), byref(k_bitrate)), (c_uint32, self.bitrate))
            _mf_call(mt_out, 21, ctypes.HRESULT, (POINTER(_GUID), byref(k_interlace)), (c_uint32, 2))
            _mf_call(mt_out, 22, ctypes.HRESULT, (POINTER(_GUID), byref(k_size)), (c_uint64, pack64(width, height)))
            _mf_call(mt_out, 22, ctypes.HRESULT, (POINTER(_GUID), byref(k_rate)), (c_uint64, pack64(fps, 1)))
            _mf_call(mt_out, 22, ctypes.HRESULT, (POINTER(_GUID), byref(k_par)), (c_uint64, pack64(1, 1)))

            stream_idx = wintypes.DWORD()
            _mf_call(self.sink_writer, 3, ctypes.HRESULT, (c_void_p, mt_out), (POINTER(wintypes.DWORD), byref(stream_idx)))
            self.stream_idx = stream_idx.value
            _mf_call(mt_out, 2, c_uint32)

            mt_in = c_void_p()
            self.mfplat.MFCreateMediaType(byref(mt_in))
            _mf_call(mt_in, 24, ctypes.HRESULT, (POINTER(_GUID), byref(k_major)), (POINTER(_GUID), byref(mt_video)))
            _mf_call(mt_in, 24, ctypes.HRESULT, (POINTER(_GUID), byref(k_sub)), (POINTER(_GUID), byref(mt_rgb32)))
            _mf_call(mt_in, 21, ctypes.HRESULT, (POINTER(_GUID), byref(k_interlace)), (c_uint32, 2))
            _mf_call(mt_in, 22, ctypes.HRESULT, (POINTER(_GUID), byref(k_size)), (c_uint64, pack64(width, height)))
            _mf_call(mt_in, 22, ctypes.HRESULT, (POINTER(_GUID), byref(k_rate)), (c_uint64, pack64(fps, 1)))
            _mf_call(mt_in, 22, ctypes.HRESULT, (POINTER(_GUID), byref(k_par)), (c_uint64, pack64(1, 1)))
            _mf_call(mt_in, 21, ctypes.HRESULT, (POINTER(_GUID), byref(k_stride)), (c_uint32, width * 4))

            _mf_call(self.sink_writer, 4, ctypes.HRESULT, (wintypes.DWORD, self.stream_idx), (c_void_p, mt_in), (c_void_p, None))
            _mf_call(mt_in, 2, c_uint32)
            _mf_call(self.sink_writer, 5, ctypes.HRESULT)

        def write_rgba_frame(self, rgba: np.ndarray) -> None:
            stride = self.width * 4
            cb_buf = stride * self.height
            buf = c_void_p()
            self.mfplat.MFCreateMemoryBuffer(c_uint32(cb_buf), byref(buf))
            p_data = POINTER(ctypes.c_ubyte)()
            _mf_call(buf, 3, ctypes.HRESULT, (POINTER(POINTER(ctypes.c_ubyte)), byref(p_data)), (c_void_p, None), (c_void_p, None))
            dst = np.ctypeslib.as_array(p_data, shape=(self.height, self.width, 4))
            # RGB32 использует порядок BGRA
            dst[:, :, 0] = rgba[:, :, 2]
            dst[:, :, 1] = rgba[:, :, 1]
            dst[:, :, 2] = rgba[:, :, 0]
            dst[:, :, 3] = 255
            _mf_call(buf, 4, ctypes.HRESULT)
            _mf_call(buf, 6, ctypes.HRESULT, (wintypes.DWORD, cb_buf))

            sample = c_void_p()
            self.mfplat.MFCreateSample(byref(sample))
            _mf_call(sample, 42, ctypes.HRESULT, (c_void_p, buf))
            _mf_call(sample, 36, ctypes.HRESULT, (c_int64, self.frame_idx * self.dur_100ns))
            _mf_call(sample, 38, ctypes.HRESULT, (c_int64, self.dur_100ns))
            _mf_call(self.sink_writer, 6, ctypes.HRESULT, (wintypes.DWORD, self.stream_idx), (c_void_p, sample))
            _mf_call(sample, 2, c_uint32)
            _mf_call(buf, 2, c_uint32)
            self.frame_idx += 1

        def close(self) -> None:
            _mf_call(self.sink_writer, 11, ctypes.HRESULT)
            _mf_call(self.sink_writer, 2, c_uint32)
            self.mfplat.MFShutdown()
            self.ole32.CoUninitialize()




def _fmt_ru(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")


def _build_demo_timeline(raw_frames: List[FrameSnapshot], target_frames: int) -> List[FrameSnapshot]:
    """Формирует сбалансированную раскадровку: все этапы аукциона наглядно видны без пропусков."""
    if not raw_frames:
        return []

    auction_indices = [i for i, f in enumerate(raw_frames) if f.phase == "AUCTION"]
    motion_indices = [i for i, f in enumerate(raw_frames) if f.phase in ("MOVING", "RETURN_BASE")]
    finish_indices = [i for i, f in enumerate(raw_frames) if f.phase == "FINISHED"]

    # Выделяем ~40% времени видео на фазы аукциона и обмена сообщениями, ~55% на движение, ~5% на финал
    n_finish = max(10, int(round(target_frames * 0.05))) if finish_indices else 0
    n_auction_budget = min(int(round(target_frames * 0.40)), max(len(auction_indices) * 6, 40)) if auction_indices else 0
    n_motion_budget = max(1, target_frames - n_auction_budget - n_finish)

    # Распределяем кадры по каждому сегменту с сохранением хронологического порядка
    timeline: List[FrameSnapshot] = []
    # Группируем raw_frames по непрерывным блокам фаз
    blocks: List[Tuple[str, List[int]]] = []
    for idx, snap in enumerate(raw_frames):
        grp = "AUCTION" if snap.phase == "AUCTION" else ("FINISHED" if snap.phase == "FINISHED" else "MOTION")
        if not blocks or blocks[-1][0] != grp:
            blocks.append((grp, [idx]))
        else:
            blocks[-1][1].append(idx)

    total_auct_raw = max(1, len(auction_indices))
    total_mot_raw = max(1, len(motion_indices))

    allocated: List[Tuple[str, List[int], int]] = []
    for grp, idxs in blocks:
        if grp == "AUCTION":
            k = max(len(idxs), int(round(n_auction_budget * len(idxs) / total_auct_raw)))
        elif grp == "MOTION":
            k = max(1, int(round(n_motion_budget * len(idxs) / total_mot_raw)))
        else:
            k = max(1, n_finish)
        allocated.append((grp, idxs, k))

    # Нормализуем точное количество кадров до target_frames
    cur_total = sum(x[2] for x in allocated)
    while cur_total != target_frames:
        diff = target_frames - cur_total
        # Корректируем за счёт самого длинного блока MOTION или AUCTION
        candidates = sorted(range(len(allocated)), key=lambda i: allocated[i][2], reverse=True)
        for ci in candidates:
            grp, idxs, k = allocated[ci]
            if diff < 0 and k > len(idxs):
                allocated[ci] = (grp, idxs, k - 1)
                cur_total -= 1
                break
            elif diff > 0:
                allocated[ci] = (grp, idxs, k + 1)
                cur_total += 1
                break
        else:
            # Крайний случай: корректируем любой блок > 1
            for ci in candidates:
                grp, idxs, k = allocated[ci]
                if diff < 0 and k > 1:
                    allocated[ci] = (grp, idxs, k - 1)
                    cur_total -= 1
                    break

    for grp, idxs, k in allocated:
        if grp == "AUCTION":
            # Показываем каждый шаг аукциона в равных долях без перескоков
            sub = np.linspace(0, len(idxs) - 1, k).astype(int)
            for s in sub:
                timeline.append(raw_frames[idxs[s]])
        elif grp == "MOTION":
            sub = np.linspace(0, len(idxs) - 1, k).astype(int)
            for s in sub:
                timeline.append(raw_frames[idxs[s]])
        else:
            for _ in range(k):
                timeline.append(raw_frames[idxs[-1]])

    return timeline[:target_frames]


def render_demo_video(
    output_mp4: str,
    cfg: ScenarioConfig | None = None,
    duration_sec: float = 25.0,
    fps: int = 24,
) -> Dict:
    """Генерация 25-секундного видеоролика в современном стиле Mission-Control HUD (1280×720, H.264)."""
    from matplotlib.patches import FancyBboxPatch

    if cfg is None:
        cfg = ScenarioConfig(
            arena_size=100.0,
            base_pos=(50.0, 50.0),
            num_robots=6,
            num_tasks=20,
            num_obstacles=4,
            topology="ring",
            heterogeneous=True,
            alpha=0.5,
            seed=42,
            dt=0.25,
        )

    sim = MultiRobotSimulator(cfg, record_frames=True, log_messages=True)
    metrics = sim.run()

    raw_frames = sim.frames
    target_frames = int(round(duration_sec * fps))
    sampled_frames = _build_demo_timeline(raw_frames, target_frames)

    # Палитра тёмного инженерного HUD
    BG_DARK = "#0b1120"
    PANEL_BG = "#111827"
    ARENA_BG = "#0f172a"
    BORDER_COL = "#1e293b"
    TEXT_MAIN = "#f1f5f9"
    TEXT_MUTED = "#94a3b8"
    NEON_COLORS = ["#38bdf8", "#fb7185", "#34d399", "#a78bfa", "#fbbf24", "#2dd4bf", "#f472b6", "#60a5fa"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG_DARK)
    fig.subplots_adjust(left=0.048, right=0.975, top=0.885, bottom=0.165, wspace=0.16, hspace=0.28)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.38, 1.0], height_ratios=[1.12, 0.56, 1.08])
    ax_arena = fig.add_subplot(gs[:, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_kpi = fig.add_subplot(gs[1, 1])
    ax_feed = fig.add_subplot(gs[2, 1])

    # Тонкая полоса прогресса миссии в самом низу кадра
    ax_prog = fig.add_axes([0.048, 0.018, 0.927, 0.012], facecolor="#1e293b")
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")

    phase_titles = {
        "AUCTION": "ДЕЦЕНТРАЛИЗОВАННЫЙ АУКЦИОН (ОБМЕН СТАВКАМИ)",
        "MOVING": "ДВИЖЕНИЕ С ОБХОДОМ ПРЕПЯТСТВИЙ И ОБСЛЕДОВАНИЕ",
        "RETURN_BASE": "ВОЗВРАЩЕНИЕ ГРУППЫ НА БАЗУ С РЕЗЕРВОМ",
        "FINISHED": "МИССИЯ ЗАВЕРШЕНА • ВСЕ РОБОТЫ НА БАЗЕ",
    }

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", label="База", markerfacecolor="#38bdf8", markeredgecolor="#f8fafc", markersize=7.5),
        Line2D([0], [0], marker="o", color="none", label="Препятствие", markerfacecolor="#1e293b", markeredgecolor="#64748b", markersize=8.5),
        Line2D([0], [0], marker="x", color="#94a3b8", linestyle="None", label="Открыта", markersize=6.5, markeredgewidth=1.7),
        Line2D([0], [0], marker="^", color="none", label="Назначена", markerfacecolor="#38bdf8", markeredgecolor="#f8fafc", markersize=7.5),
        Line2D([0], [0], marker="o", color="none", label="Выполнена", markerfacecolor="#10b981", markeredgecolor="#d1fae5", markersize=6.8),
        Line2D([0], [0], color="#f59e0b", linestyle=":", linewidth=2.0, label="Связь (пакеты)"),
        Line2D([0], [0], marker="o", color="none", label="Конфликт", markerfacecolor="none", markeredgecolor="#f43f5e", markeredgewidth=1.8, markersize=8.5),
    ]

    # Верхний заголовок всего HUD
    sup_title = fig.text(
        0.048, 0.952,
        "ИМИТАЦИОННАЯ МОДЕЛЬ МНОГОРОБОТНОЙ СИСТЕМЫ",
        fontsize=13.5, fontweight="bold", color=TEXT_MAIN, ha="left", va="center",
    )
    sup_meta = fig.text(
        0.975, 0.952,
        "",
        fontsize=9.5, fontweight="bold", color="#38bdf8", ha="right", va="center",
    )

    def draw_frame(idx: int) -> None:
        snap = sampled_frames[idx]
        progress = idx / max(1, len(sampled_frames) - 1)
        sup_meta.set_text(
            f"АЛГОРИТМ: {snap.allocator_name.upper()}  •  СЕТЬ: {cfg.topology.upper()}  •  "
            f"РАУНД {snap.round_idx}  •  t = {snap.sim_time:5.1f} с"
        )

        # --- Прогресс-бар внизу ---
        ax_prog.clear()
        ax_prog.set_xlim(0, 1)
        ax_prog.set_ylim(0, 1)
        ax_prog.axis("off")
        ax_prog.add_patch(FancyBboxPatch((0, 0.1), 1.0, 0.8, boxstyle="round,pad=0.0,rounding_size=0.008", facecolor="#1e293b", edgecolor="none"))
        bar_col = "#f59e0b" if snap.phase == "AUCTION" else ("#10b981" if snap.phase == "FINISHED" else "#38bdf8")
        if progress > 0.005:
            ax_prog.add_patch(FancyBboxPatch((0, 0.1), max(0.012, progress), 0.8, boxstyle="round,pad=0.0,rounding_size=0.008", facecolor=bar_col, edgecolor="none"))

        # --- Левая панель: 2D-полигон ---
        ax_arena.clear()
        ax_arena.set_facecolor(ARENA_BG)
        ax_arena.set_xlim(0, cfg.arena_size)
        ax_arena.set_ylim(0, cfg.arena_size)
        ax_arena.set_aspect("equal")
        ax_arena.grid(True, linestyle="--", alpha=0.18, color="#64748b", linewidth=0.7)
        for sp in ax_arena.spines.values():
            sp.set_color("#334155")
            sp.set_linewidth(1.1)
        ax_arena.tick_params(colors=TEXT_MUTED, labelsize=8.5)
        ax_arena.set_xlabel("Координата X, м", fontsize=9.2, color=TEXT_MUTED)
        ax_arena.set_ylabel("Координата Y, м", fontsize=9.2, color=TEXT_MUTED)

        sub_str = snap.phase_detail or phase_titles.get(snap.phase, snap.phase)
        if snap.phase == "AUCTION":
            badge_bg, badge_ec, badge_fg = "#451a03", "#f59e0b", "#fde68a"
        elif snap.phase == "FINISHED":
            badge_bg, badge_ec, badge_fg = "#064e3b", "#10b981", "#a7f3d0"
        else:
            badge_bg, badge_ec, badge_fg = "#0c4a6e", "#38bdf8", "#bae6fd"

        ax_arena.text(
            0.5, 1.022,
            sub_str,
            transform=ax_arena.transAxes,
            ha="center", va="bottom",
            fontsize=8.8, fontweight="bold", color=badge_fg,
            bbox=dict(boxstyle="round,pad=0.32", facecolor=badge_bg, edgecolor=badge_ec, lw=1.1),
            clip_on=False,
        )

        # Препятствия с зоной безопасности (+0.75 м)
        for obs in sim.obstacles:
            ax_arena.add_patch(
                Circle((obs.x, obs.y), obs.radius + 0.75, facecolor="none", edgecolor="#334155", lw=0.9, ls="--", zorder=2)
            )
            ax_arena.add_patch(
                Circle((obs.x, obs.y), obs.radius, facecolor="#1e293b", edgecolor="#64748b", lw=1.3, zorder=2)
            )

        # База со свечением
        bx, by = cfg.base_pos
        ax_arena.add_patch(Circle((bx, by), 4.6, facecolor="#38bdf8", alpha=0.12, edgecolor="none", zorder=3))
        ax_arena.scatter([bx], [by], marker="s", s=165, color="#0ea5e9", edgecolor="#f8fafc", linewidth=1.4, zorder=4)
        ax_arena.text(bx, by - 4.4, "БАЗА", ha="center", va="top", fontsize=7.8, fontweight="bold", color="#7dd3fc", zorder=5)

        rmap = {r["id"]: r for r in snap.robots}
        tmap = {t["id"]: t for t in snap.tasks}

        # Анимированные линии связи и движущиеся пакеты во время аукциона
        if snap.active_links:
            pulse = (idx % 16) / 16.0
            for u, v in snap.active_links:
                ru, rv = rmap[u], rmap[v]
                ax_arena.plot(
                    [ru["x"], rv["x"]], [ru["y"], rv["y"]],
                    linestyle=":", linewidth=2.2, color="#f59e0b", alpha=0.85, zorder=3,
                )
                # Две встречные точки-пакеты вдоль ребра связи
                px1 = ru["x"] + pulse * (rv["x"] - ru["x"])
                py1 = ru["y"] + pulse * (rv["y"] - ru["y"])
                px2 = rv["x"] + pulse * (ru["x"] - rv["x"])
                py2 = rv["y"] + pulse * (ru["y"] - rv["y"])
                ax_arena.scatter([px1, px2], [py1, py2], s=24, color="#fde047", edgecolor="#b45309", linewidth=0.6, zorder=5)

        # Кандидатные ставки в фазе аукциона
        if snap.phase == "AUCTION" and snap.bid_annotations:
            for ann in snap.bid_annotations:
                r = rmap.get(ann["robot_id"])
                t = tmap.get(ann["task_id"])
                if r is None or t is None:
                    continue
                c = NEON_COLORS[r["id"] % len(NEON_COLORS)]
                ax_arena.plot([r["x"], t["x"]], [r["y"], t["y"]], color=c, linestyle="--", linewidth=1.35, alpha=0.8, zorder=4)
                dist_rt = math.hypot(t["x"] - r["x"], t["y"] - r["y"])
                if dist_rt > 8.5:
                    lx = 0.30 * r["x"] + 0.70 * t["x"]
                    ly = 0.30 * r["y"] + 0.70 * t["y"]
                    ax_arena.text(
                        lx, ly, _fmt_ru(ann["score"], 2),
                        fontsize=7.3, fontweight="bold", color=TEXT_MAIN,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.16", facecolor="#0f172a", edgecolor=c, lw=0.9, alpha=0.95),
                        zorder=7,
                    )

        # Контрольные точки (задачи)
        contested_set = set(snap.contested_tasks)
        pulse_rad = 220 + 70 * math.sin(idx * 0.5)
        for t in snap.tasks:
            tid = t["id"]
            if t["completed"]:
                ax_arena.scatter([t["x"]], [t["y"]], marker="o", s=110, color="#10b981", alpha=0.22, zorder=3)
                ax_arena.scatter([t["x"]], [t["y"]], marker="o", s=50, color="#10b981", edgecolor="#ecfdf5", linewidth=0.9, zorder=4)
            elif t["assigned_to"] is not None:
                c = NEON_COLORS[t["assigned_to"] % len(NEON_COLORS)]
                ax_arena.scatter([t["x"]], [t["y"]], marker="^", s=78, color=c, edgecolor="#f8fafc", linewidth=0.9, zorder=5)
            else:
                ax_arena.scatter([t["x"]], [t["y"]], marker="x", s=50, color="#94a3b8", linewidths=1.7, zorder=4)

            if tid in contested_set:
                ax_arena.scatter(
                    [t["x"]], [t["y"]], s=pulse_rad, facecolors="none", edgecolors="#f43f5e", linewidths=2.2, zorder=6
                )

        # Роботы со свечением и плавным хвостом траектории
        for r in snap.robots:
            c = NEON_COLORS[r["id"] % len(NEON_COLORS)]
            traj = r["traj"]
            if len(traj) > 1:
                xs, ys = zip(*traj)
                ax_arena.plot(xs, ys, color=c, alpha=0.52, linewidth=1.6, zorder=3)
            if r["assigned"] is not None:
                t = tmap[r["assigned"]]
                ax_arena.plot([r["x"], t["x"]], [r["y"], t["y"]], color=c, linestyle="--", linewidth=1.1, alpha=0.55, zorder=3)
            ax_arena.scatter([r["x"]], [r["y"]], s=260, color=c, alpha=0.22, edgecolor="none", zorder=5)
            ax_arena.scatter([r["x"]], [r["y"]], s=135, color=c, edgecolor="#f8fafc", linewidth=1.3, zorder=6)
            ax_arena.text(r["x"], r["y"] + 2.9, f"R{r['id']}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=TEXT_MAIN, zorder=7)

        leg = ax_arena.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.115),
            ncol=4,
            fontsize=7.5,
            frameon=True,
            facecolor=PANEL_BG,
            edgecolor="#334155",
            labelcolor=TEXT_MAIN,
            columnspacing=0.9,
            handletextpad=0.35,
        )

        # --- Правая верхняя панель: Запас энергии ---
        ax_energy.clear()
        ax_energy.set_facecolor(PANEL_BG)
        for sp in ax_energy.spines.values():
            sp.set_color(BORDER_COL)
        ax_energy.spines["top"].set_visible(False)
        ax_energy.spines["right"].set_visible(False)
        rids = [f"R{r['id']}" for r in snap.robots]
        energies = [r["energy"] for r in snap.robots]
        max_e = max(r["initial_energy"] for r in snap.robots) * 1.16
        colors = [NEON_COLORS[r["id"] % len(NEON_COLORS)] for r in snap.robots]
        bars = ax_energy.bar(rids, energies, color=colors, edgecolor="#f8fafc", linewidth=0.7, width=0.58)
        ax_energy.axhline(10.0, color="#f43f5e", linestyle="--", linewidth=1.3, label="Резерв (10 ед.)")
        ax_energy.set_ylim(0, max_e)
        ax_energy.tick_params(colors=TEXT_MUTED, labelsize=8.5)
        ax_energy.set_ylabel("Энергия, ед.", fontsize=8.8, color=TEXT_MUTED)
        ax_energy.set_title("Текущий запас энергии роботов", fontsize=10.0, fontweight="bold", color=TEXT_MAIN, pad=6)
        ax_energy.legend(loc="upper right", fontsize=8.0, facecolor=ARENA_BG, edgecolor="#334155", labelcolor=TEXT_MAIN)
        ax_energy.grid(True, axis="y", linestyle="--", alpha=0.18, color="#64748b")
        for bar, val in zip(bars, energies):
            ax_energy.text(bar.get_x() + bar.get_width() / 2, val + 3.0, f"{val:.0f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=TEXT_MAIN)

        # --- Правая средняя панель: 4 KPI-плитки ---
        ax_kpi.clear()
        ax_kpi.set_xlim(0, 1)
        ax_kpi.set_ylim(0, 1)
        ax_kpi.axis("off")
        cov_pct = 100.0 * snap.completed_count / max(1, snap.total_tasks)
        kpis = [
            ("ПОКРЫТИЕ", f"{snap.completed_count}/{snap.total_tasks} ({_fmt_ru(cov_pct, 0)}%)", "#34d399"),
            ("СООБЩЕНИЯ", f"{snap.messages_total}", "#38bdf8"),
            ("ТРАФИК", f"{_fmt_ru(snap.bytes_total / 1024, 1)} КБ", "#fbbf24"),
            ("НАРУШЕНИЯ", f"{sim.violations}", "#34d399" if sim.violations == 0 else "#f43f5e"),
        ]
        card_w = 0.222
        gap = 0.036
        for ci, (k_lbl, k_val, k_col) in enumerate(kpis):
            x0 = ci * (card_w + gap)
            ax_kpi.add_patch(
                FancyBboxPatch((x0 + 0.005, 0.08), card_w - 0.005, 0.84, boxstyle="round,pad=0.01,rounding_size=0.05", facecolor=PANEL_BG, edgecolor="#334155", lw=1.0)
            )
            ax_kpi.text(x0 + card_w / 2, 0.70, k_lbl, ha="center", va="center", fontsize=7.0, fontweight="bold", color=TEXT_MUTED)
            ax_kpi.text(x0 + card_w / 2, 0.34, k_val, ha="center", va="center", fontsize=10.2, fontweight="bold", color=k_col)

        # --- Правая нижняя панель: Журнал аукциона и реестра смарт-контракта ---
        ax_feed.clear()
        ax_feed.set_xlim(0, 1)
        ax_feed.set_ylim(0, 1)
        ax_feed.axis("off")
        tx_str = snap.ledger_tx if snap.ledger_tx else "ожидание консенсуса..."

        if snap.phase == "AUCTION" and snap.round_summary:
            lines = "\n".join(snap.round_summary)
            feed_title = f"АУКЦИОН БЕЗ ДИСПЕТЧЕРА  •  РАУНД {snap.round_idx} (α = {_fmt_ru(cfg.alpha, 2)})"
            feed_body = f"{lines}\nРеестр (Задача 4): {tx_str}"
            f_bg, f_ec, f_tc = "#1c1917", "#f59e0b", "#fde68a"
        else:
            feed_title = "ТЕЛЕМЕТРИЯ ПЛАТФОРМЫ И РЕЕСТР НАЗНАЧЕНИЙ"
            feed_body = (
                f"Режим группы: {'неоднородный' if cfg.heterogeneous else 'однородный'}  •  α = {_fmt_ru(cfg.alpha, 2)}\n"
                f"Совместимость: MRTA (1Б) • CBAA (2) • Ставки (3) • API (4)\n"
                f"Последняя запись реестра: {tx_str}\n"
                f"Коллизий с препятствиями: {sim.obstacle_penetrations}  •  Нарушений: {sim.violations}\n"
                f"Пройдено шагов согласования: {sim.total_sync_steps}"
            )
            f_bg, f_ec, f_tc = PANEL_BG, "#334155", "#7dd3fc"

        ax_feed.add_patch(
            FancyBboxPatch((0.005, 0.03), 0.99, 0.94, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=f_bg, edgecolor=f_ec, lw=1.1)
        )
        ax_feed.text(0.03, 0.88, feed_title, ha="left", va="top", fontsize=8.3, fontweight="bold", color=f_tc)
        ax_feed.text(0.03, 0.72, feed_body, ha="left", va="top", fontsize=8.4, family="DejaVu Sans", color=TEXT_MAIN, linespacing=1.35)

    out_dir = os.path.dirname(output_mp4)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(output_mp4)[1].lower()
    if ext == ".gif":
        anim = FuncAnimation(fig, draw_frame, frames=len(sampled_frames), interval=1000 / fps)
        anim.save(output_mp4, writer=PillowWriter(fps=fps))
    elif FFMpegWriter.isAvailable():
        anim = FuncAnimation(fig, draw_frame, frames=len(sampled_frames), interval=1000 / fps)
        writer = FFMpegWriter(fps=fps, metadata={"title": "Multi-Robot System Simulation HUD"}, bitrate=2600)
        anim.save(output_mp4, writer=writer)
    elif os.name == "nt":
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        mf_writer = _MediaFoundationMP4Writer(output_mp4, w_px, h_px, fps=fps, bitrate=2_600_000)
        last_snap_id: Optional[int] = None
        cached_rgba: Optional[np.ndarray] = None
        try:
            for i in range(len(sampled_frames)):
                snap_id = id(sampled_frames[i])
                if snap_id != last_snap_id or cached_rgba is None:
                    draw_frame(i)
                    fig.canvas.draw()
                    cached_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
                    last_snap_id = snap_id
                mf_writer.write_rgba_frame(cached_rgba)
        finally:
            mf_writer.close()
    else:
        gif_fallback = os.path.splitext(output_mp4)[0] + ".gif"
        anim = FuncAnimation(fig, draw_frame, frames=len(sampled_frames), interval=1000 / fps)
        anim.save(gif_fallback, writer=PillowWriter(fps=fps))

    # Сохраняем кадр аукциона и кадр движения
    base_no_ext = os.path.splitext(output_mp4)[0]
    auction_shot_idx = next(
        (
            i
            for i, f in enumerate(sampled_frames)
            if f.phase == "AUCTION" and f.round_idx >= 1 and f.active_links and f.bid_annotations
        ),
        0,
    )
    draw_frame(auction_shot_idx)
    fig.savefig(base_no_ext + "_frame.png", dpi=180, facecolor=BG_DARK)

    moving_indices = [i for i, f in enumerate(sampled_frames) if f.phase == "MOVING" and f.completed_count >= 4]
    if moving_indices:
        draw_frame(moving_indices[len(moving_indices) // 3])
        fig.savefig(base_no_ext + "_moving_frame.png", dpi=180, facecolor=BG_DARK)

    plt.close(fig)
    return metrics


def plot_experiment_charts(summary_rows: List[Dict], out_dir: str) -> List[str]:
    """Построение графиков результатов сравнительных экспериментов для отчёта."""
    os.makedirs(out_dir, exist_ok=True)
    created: List[str] = []

    # График 1: Сравнение покрытия K по весу alpha (однородный vs неоднородный режимы, кольцо)
    alphas = sorted({row["alpha"] for row in summary_rows})
    hom_rows = [r for r in summary_rows if not r["heterogeneous"] and r["topology"] == "ring"]
    het_rows = [r for r in summary_rows if r["heterogeneous"] and r["topology"] == "ring"]
    if het_rows and hom_rows:
        fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=200)
        x = np.arange(len(alphas))
        width = 0.36
        hom_cov = [next(r["mean_coverage"] * 100.0 for r in hom_rows if r["alpha"] == a) for a in alphas]
        het_cov = [next(r["mean_coverage"] * 100.0 for r in het_rows if r["alpha"] == a) for a in alphas]

        b1 = ax.bar(x - width / 2, hom_cov, width, label="Однородная группа", color="#64748b", edgecolor="#0f172a")
        b2 = ax.bar(x + width / 2, het_cov, width, label="Неоднородная группа", color="#2563eb", edgecolor="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels([f"α = {_fmt_ru(a, 2)}" for a in alphas], fontsize=9.5)
        ax.set_ylabel("Среднее покрытие K, %", fontsize=10)
        ax.set_xlabel("Коэффициент веса расстояния α", fontsize=10)
        ax.set_title("Влияние веса α на покрытие K в однородной и неоднородной группах", fontsize=10.2, fontweight="bold")
        ax.set_ylim(75, max(hom_cov + het_cov) + 6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.8, frameon=True)
        for bar, val in list(zip(b1, hom_cov)) + list(zip(b2, het_cov)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{_fmt_ru(val, 1)}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )
        p1 = os.path.join(out_dir, "chart_coverage_alpha.png")
        fig.savefig(p1, bbox_inches="tight")
        plt.close(fig)
        created.append(p1)

    # График 2: Сравнение коммуникационных затрат по топологиям сети (full, ring, line)
    topo_order = ["full", "ring", "line"]
    topo_labels = {"full": "Полный граф\n(D = 1)", "ring": "Кольцо\n(D = 3)", "line": "Линейная цепь\n(D = 5)"}
    topo_rows = [r for r in summary_rows if r["heterogeneous"] and abs(r["alpha"] - 0.5) < 1e-6]
    if topo_rows:
        fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=200)
        msgs = [next(r["mean_messages"] for r in topo_rows if r["topology"] == t) for t in topo_order]
        steps = [next(r["mean_sync_steps"] for r in topo_rows if r["topology"] == t) for t in topo_order]
        bars = ax.bar(
            [topo_labels[t] for t in topo_order],
            msgs,
            color=["#1e3a8a", "#0284c7", "#0d9488"],
            width=0.46,
            edgecolor="#0f172a",
        )
        ax.set_ylabel("Среднее число сообщений за миссию", fontsize=10)
        ax.set_xlabel("Топология статической связной сети", fontsize=10)
        ax.set_title("Зависимость коммуникационных затрат от топологии сети (α = 0,5)", fontsize=10.2, fontweight="bold")
        ax.set_ylim(0, max(msgs) * 1.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        for bar, val, st in zip(bars, msgs, steps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(msgs) * 0.025,
                f"{val:.0f} сообщ.\n({_fmt_ru(st, 1)} шагов)",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold",
            )
        p2 = os.path.join(out_dir, "chart_topology_messages.png")
        fig.savefig(p2, bbox_inches="tight")
        plt.close(fig)
        created.append(p2)

    # График 3: Удельный расход энергии на одну обследованную точку в неоднородной группе
    if het_rows and "mean_energy_per_task" in het_rows[0]:
        fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=200)
        ept = [next(r["mean_energy_per_task"] for r in het_rows if r["alpha"] == a) for a in alphas]
        labels = [f"α = {_fmt_ru(a, 2)}" for a in alphas]
        bars = ax.bar(labels, ept, color=["#0f766e", "#1d4ed8", "#16a34a", "#64748b"], width=0.48, edgecolor="#0f172a")
        ax.set_ylabel("Энергия на 1 точку, усл. ед.", fontsize=10)
        ax.set_xlabel("Коэффициент веса расстояния α", fontsize=10)
        ax.set_title("Удельный расход энергии на обследование одной точки (неоднородная группа)", fontsize=10, fontweight="bold")
        ax.set_ylim(min(ept) * 0.85, max(ept) * 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, ept):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25, _fmt_ru(val, 2), ha="center", va="bottom", fontsize=9, fontweight="bold")
        p3 = os.path.join(out_dir, "chart_energy_per_task.png")
        fig.savefig(p3, bbox_inches="tight")
        plt.close(fig)
        created.append(p3)

    return created
