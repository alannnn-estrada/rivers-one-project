"""Estilos y temas visuales compartidos para toda la aplicación."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from sympy import Symbol, latex as sympy_latex
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


_PARSING_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_X_SYMBOL = Symbol("x")


DARK_THEME_STYLESHEET = """
QMainWindow, QWidget {
    background: #0b1220;
    color: #e5e7eb;
}

QLabel {
    color: #e5e7eb;
    font-size: 11px;
}

QLineEdit, QTextEdit {
    background: #111827;
    color: #f9fafb;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #2563eb;
    font-size: 11px;
}

QLineEdit:focus, QTextEdit:focus {
    border: 2px solid #2563eb;
}

QGroupBox {
    color: #e5e7eb;
    border: 1px solid #334155;
    border-radius: 6px;
    padding-top: 10px;
    margin-top: 8px;
    font-size: 11px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}

QPushButton {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 11px;
    min-height: 18px;
}

QPushButton:hover {
    background: #1d4ed8;
}

QPushButton:pressed {
    background: #1a3fbe;
}

QPushButton#secondaryButton {
    background: #1f2937;
    color: #e5e7eb;
    border: 1px solid #334155;
}

QPushButton#secondaryButton:hover {
    background: #334155;
}

QComboBox {
    background: #111827;
    color: #f9fafb;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px;
    font-size: 11px;
}

QComboBox:focus {
    border: 2px solid #2563eb;
}

QComboBox::drop-down {
    border: none;
    background: transparent;
}

QComboBox::down-arrow {
    image: none;
    background: transparent;
}

QComboBox QAbstractItemView {
    background: #1f2937;
    color: #f9fafb;
    selection-background-color: #2563eb;
    border: 1px solid #334155;
}

QTableWidget, QTreeWidget {
    background: #111827;
    alternate-background-color: #0f172a;
    color: #f9fafb;
    border: 1px solid #334155;
    gridline-color: #334155;
    font-size: 10px;
}

QTableWidget::item:selected, QTreeWidget::item:selected {
    background: #2563eb;
    color: #ffffff;
}

QHeaderView::section {
    background: #1f2937;
    color: #e5e7eb;
    padding: 4px;
    border: none;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    font-size: 10px;
    font-weight: 600;
}

QScrollBar:vertical {
    background: #0b1220;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar:horizontal {
    background: #0b1220;
    height: 12px;
    border: none;
}

QScrollBar::handle:horizontal {
    background: #334155;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background: #475569;
}

QToolTip {
    background: #111827;
    color: #f9fafb;
    border: 1px solid #475569;
    padding: 6px;
    border-radius: 4px;
    font-size: 10px;
}

QMessageBox QLabel {
    color: #e5e7eb;
}

QMessageBox {
    background: #0b1220;
}
"""


def latex_to_pixmap(
    formula: str,
    fontsize: int = 14,
    dpi: int = 100,
    label: str = "f(x) =",
) -> QPixmap | None:
    """
    Convierte una fórmula LaTeX a un QPixmap para mostrar en la UI.
    Retorna None si la fórmula está vacía o hay error.
    """
    if not formula or not formula.strip():
        return None

    try:
        clean_formula = formula.strip()
        while '  ' in clean_formula:
            clean_formula = clean_formula.replace('  ', ' ')

        cleaned_for_parse = clean_formula.replace("f(x) =", "").replace("f'(x) =", "").strip()
        parsed = parse_expr(
            cleaned_for_parse,
            local_dict={"x": _X_SYMBOL},
            transformations=_PARSING_TRANSFORMATIONS,
            evaluate=True,
        )
        latex_expression = sympy_latex(parsed)

        fig = Figure(figsize=(8, 1.2), dpi=dpi, facecolor='#111827', edgecolor='none')
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.margins(0)

        ax.text(
            0.5, 0.5, f'${label} {latex_expression}$',
            fontsize=fontsize,
            ha='center', va='center',
            transform=ax.transAxes,
            color='#f9fafb',
            family='serif',
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': '#111827', 'edgecolor': 'none', 'alpha': 0}
        )
        fig.tight_layout(pad=0.05)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.08)
        qpixmap = QPixmap()
        qpixmap.loadFromData(buffer.getvalue(), 'PNG')
        return qpixmap if not qpixmap.isNull() else None
    except Exception:
        return None


def get_dark_stylesheet() -> str:
    """Retorna el stylesheet del tema oscuro completo."""
    return DARK_THEME_STYLESHEET
