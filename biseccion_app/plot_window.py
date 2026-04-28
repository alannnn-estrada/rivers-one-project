from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget, QToolTip


class FunctionPlotWindow(QWidget):
    def __init__(
        self,
        function: Callable[[float], float],
        plot_span: float,
        root_markers: Sequence[tuple[float, str, str]],
        method_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Grafica de la funcion")
        self.resize(1100, 720)
        self.setMinimumSize(920, 600)
        self.setStyleSheet(
            """
            QWidget {
                background: #0b1220;
                color: #e5e7eb;
            }
            QLabel#plotHint {
                color: #cbd5e1;
                font-size: 12px;
                padding: 4px 2px;
            }
            QPushButton {
                background: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
                min-height: 18px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton#secondaryButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #334155;
            }
            QPushButton#secondaryButton:hover {
                background: #334155;
            }
            QToolTip {
                background: #111827;
                color: #f9fafb;
                border: 1px solid #475569;
                padding: 6px;
            }
            QToolBar {
                background: #111827;
                border: 1px solid #334155;
                spacing: 6px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hint_label = QLabel(
            "Pasa el mouse sobre las raíces para ver su valor. Usa la barra o los botones para acercar, alejar y mover la vista."
        )
        hint_label.setObjectName("plotHint")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.zoom_in_btn = QPushButton("Acercar")
        self.zoom_in_btn.setToolTip("Aumenta el zoom centrado en el área visible actual.")
        self.zoom_in_btn.setCursor(Qt.PointingHandCursor)
        self.zoom_in_btn.clicked.connect(lambda: self._zoom(0.8))
        self.zoom_out_btn = QPushButton("Alejar")
        self.zoom_out_btn.setToolTip("Reduce el zoom para ver un tramo mayor de la función.")
        self.zoom_out_btn.setObjectName("secondaryButton")
        self.zoom_out_btn.setCursor(Qt.PointingHandCursor)
        self.zoom_out_btn.clicked.connect(lambda: self._zoom(1.25))
        self.reset_btn = QPushButton("Restaurar vista")
        self.reset_btn.setToolTip("Vuelve al encuadre original calculado para la gráfica.")
        self.reset_btn.setObjectName("secondaryButton")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(self.zoom_in_btn)
        controls.addWidget(self.zoom_out_btn)
        controls.addWidget(self.reset_btn)
        controls.addStretch()
        layout.addLayout(controls)

        figure = Figure(figsize=(9.4, 5.8), dpi=100, facecolor="#0b1220")
        self._axis = figure.add_subplot(111)
        self._axis.set_facecolor("#111827")
        self._canvas = FigureCanvas(figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setToolTip("Herramientas: zoom rectangular, paneo, restaurar y guardar.")
        self._toolbar.setStyleSheet(
            """
            QToolBar {
                background: #111827;
                border: 1px solid #334155;
            }
            QToolButton {
                background: transparent;
                color: #e5e7eb;
                border: none;
                padding: 4px;
            }
            QToolButton:hover {
                background: #1f2937;
                border-radius: 4px;
            }
            """
        )
        for action in self._toolbar.actions():
            text = action.text().replace("&", "").strip()
            if text:
                action.setToolTip(text)
                action.setStatusTip(text)

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton("Cerrar")
        close_btn.setObjectName("secondaryButton")
        close_btn.setToolTip("Cierra solo la ventana de la gráfica.")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn, alignment=Qt.AlignRight)
        layout.addLayout(footer)

        self._root_markers: list[tuple[float, str, str]] = list(root_markers)
        self._method_name = method_name
        self._function = function
        self._base_xlim: tuple[float, float] = (-plot_span, plot_span)
        self._base_ylim: tuple[float, float] | None = None
        self._marker_points: list[tuple[float, float, str]] = []
        self._hover_annotation = self._axis.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#0f172a", "ec": "#475569", "alpha": 0.96},
            color="#f9fafb",
            fontsize=9,
        )
        self._hover_annotation.set_visible(False)

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._render(function, plot_span, root_markers, method_name)

    def _render(
        self,
        function: Callable[[float], float],
        plot_span: float,
        root_markers: Sequence[tuple[float, str, str]],
        method_name: str,
    ) -> None:
        x_values = np.linspace(-plot_span, plot_span, 1200)
        y_values = np.array([self._safe_eval(function, x) for x in x_values], dtype=float)
        self._marker_points = []

        finite_y = y_values[np.isfinite(y_values)]
        if finite_y.size:
            y_min = float(np.min(finite_y))
            y_max = float(np.max(finite_y))
            y_padding = max(1.0, (y_max - y_min) * 0.12)
            self._base_ylim = (y_min - y_padding, y_max + y_padding)
        else:
            self._base_ylim = (-1.0, 1.0)

        axis = self._axis
        axis.clear()
        axis.set_facecolor("#111827")
        axis.plot(x_values, y_values, color="#60a5fa", linewidth=2.2, label="f(x)")
        axis.axhline(0, color="#94a3b8", linewidth=1.1, alpha=0.8)
        axis.axvline(0, color="#94a3b8", linewidth=1.0, linestyle="--", alpha=0.65)

        for x_value, label, color in root_markers:
            try:
                y_value = function(x_value)
                if not np.isfinite(y_value):
                    continue
                axis.scatter([x_value], [y_value], s=76, zorder=5, color=color, label=label, edgecolors="#f8fafc", linewidths=1.0)
                self._marker_points.append((float(x_value), float(y_value), label))
            except Exception:
                continue

        axis.set_title(f"Funcion y raices aproximadas por {method_name}", fontsize=13, pad=14)
        axis.set_xlabel("x")
        axis.set_ylabel("f(x)")
        axis.tick_params(colors="#e5e7eb")
        axis.xaxis.label.set_color("#e5e7eb")
        axis.yaxis.label.set_color("#e5e7eb")
        axis.title.set_color("#f9fafb")
        for spine in axis.spines.values():
            spine.set_color("#475569")
        axis.grid(alpha=0.14, linestyle=":", color="#64748b")
        legend = axis.legend(loc="best", frameon=True, framealpha=0.96)
        if legend is not None:
            legend.get_frame().set_facecolor("#111827")
            legend.get_frame().set_edgecolor("#475569")
            for text in legend.get_texts():
                text.set_color("#f8fafc")
        axis.set_xlim(*self._base_xlim)
        if self._base_ylim is not None:
            axis.set_ylim(*self._base_ylim)
        self._axis.figure.tight_layout()
        self._canvas.draw_idle()

    def _zoom(self, factor: float) -> None:
        x_min, x_max = self._axis.get_xlim()
        y_min, y_max = self._axis.get_ylim()
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_half = (x_max - x_min) * factor / 2.0
        y_half = (y_max - y_min) * factor / 2.0
        self._axis.set_xlim(x_center - x_half, x_center + x_half)
        self._axis.set_ylim(y_center - y_half, y_center + y_half)
        self._canvas.draw_idle()

    def _reset_view(self) -> None:
        self._axis.set_xlim(*self._base_xlim)
        if self._base_ylim is not None:
            self._axis.set_ylim(*self._base_ylim)
        self._canvas.draw_idle()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes != self._axis or event.xdata is None or event.ydata is None:
            if self._hover_annotation.get_visible():
                self._hover_annotation.set_visible(False)
                self._canvas.draw_idle()
            QToolTip.hideText()
            return

        best_match: tuple[float, float, str] | None = None
        best_distance = 0.12 * max(self._axis.get_xlim()[1] - self._axis.get_xlim()[0], self._axis.get_ylim()[1] - self._axis.get_ylim()[0])

        for x_value, y_value, label in self._marker_points:
            distance = float(np.hypot(event.xdata - x_value, event.ydata - y_value))
            if distance <= best_distance:
                best_distance = distance
                best_match = (x_value, y_value, label)

        if best_match is None:
            if self._hover_annotation.get_visible():
                self._hover_annotation.set_visible(False)
                self._canvas.draw_idle()
            return

        x_value, y_value, label = best_match
        self._hover_annotation.xy = (x_value, y_value)
        self._hover_annotation.set_text(label)
        self._hover_annotation.set_visible(True)
        QToolTip.showText(QCursor.pos(), label, self._canvas)
        self._canvas.draw_idle()

    @staticmethod
    def _safe_eval(function: Callable[[float], float], x_value: float) -> float:
        try:
            y_value = float(function(x_value))
            return y_value if np.isfinite(y_value) else np.nan
        except Exception:
            return np.nan