from __future__ import annotations

"""Interfaz gráfica usando PySide6 para el método de la bisección.

Reimplementa la UI previamente basada en Tkinter. Mantiene la lógica central
en `biseccion_app.math_engine` y usa Qt widgets para la interacción y tablas.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import re

import matplotlib.pyplot as plt

# Intento de importación de Qt en un único bloque
try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QTreeWidget,
        QTreeWidgetItem,
        QGroupBox,
        QMessageBox,
        QComboBox,
    )
    from PySide6.QtCore import Qt
except Exception as exc:  # pragma: no cover - facilita mensaje claro si faltan bindings
    raise ImportError(
        "Falta un binding de Qt necesario para la interfaz gráfica.\n"
        "Instale PySide6 o PyQt5 en su entorno virtual y vuelva a ejecutar.\n\n"
        "Pasos rápidos (Windows PowerShell):\n"
        ".\\venv\\Scripts\\activate  # activar el venv\n"
        "pip install PySide6  # o: pip install PyQt5\n\n"
        f"Mensaje original: {exc}"
    )

from biseccion_app.math_engine import (
    BisectionResult,
    FormulaError,
    SignChangeSearchResult,
    MultipleSignChangeSearchResult,
    SuccessiveApproxResult,
    NewtonRaphsonResult,
    compile_function,
    compile_derivative,
    find_first_sign_change,
    find_all_sign_changes,
    run_bisection,
    run_successive_approximations,
    run_newton_raphson,
    get_derivative_expression,
)
from biseccion_app.plot_window import FunctionPlotWindow
from biseccion_app.methods import get_available_methods


def _get_runtime_signature() -> str:
    """Devuelve la versión de Python y metadatos Git del repositorio actual."""
    python_version = sys.version.split()[0]
    repo_root = Path(__file__).resolve().parents[1]

    try:
        commit_id = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit_count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_part = f"Git {commit_id} | commits: {commit_count}"
    except Exception:
        git_part = "Git no disponible"

    return f"Python {python_version} | {git_part}"


class BisectionApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Metodo de biseccion")
        self.resize(1080, 680)

        self.compiled_function = None
        self.compiled_derivative = None
        self.formula_text = ""
        self.sign_search_result: Optional[SignChangeSearchResult | MultipleSignChangeSearchResult] = None
        self.plot_span = 5.0
        self.result: Optional[BisectionResult] = None
        self.successive_result: Optional[SuccessiveApproxResult] = None
        self.newton_raphson_result: Optional[NewtonRaphsonResult] = None
        self.found_roots: list[Tuple[int, BisectionResult]] = []
        self._plot_window: Optional[QWidget] = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Entradas
        input_group = QGroupBox("Entradas")
        input_layout = QHBoxLayout()
        input_group.setLayout(input_layout)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.formula_edit = QLineEdit("x^3 + 2x^2 - 9")
        self.proposed_edit = QLineEdit("0")
        self.tab_max_edit = QLineEdit("100")
        self.error_edit = QLineEdit("0.001")
        self.max_iter_edit = QLineEdit("100")
        self.derivative_edit = QLineEdit("")
        self.derivative_edit.setReadOnly(True)

        left_layout.addWidget(QLabel("Formula f(x):"))
        left_layout.addWidget(self.formula_edit)
        left_layout.addWidget(QLabel("Numero propuesto:"))
        left_layout.addWidget(self.proposed_edit)

        right_layout.addWidget(QLabel("Tabulaciones maximas (busqueda):"))
        right_layout.addWidget(self.tab_max_edit)
        right_layout.addWidget(QLabel("Tolerancia de error:"))
        right_layout.addWidget(self.error_edit)
        right_layout.addWidget(QLabel("Max iteraciones:"))
        right_layout.addWidget(self.max_iter_edit)
        self.derivative_label = QLabel("Derivada f'(x):")
        right_layout.addWidget(self.derivative_label)
        right_layout.addWidget(self.derivative_edit)

        input_layout.addLayout(left_layout)
        input_layout.addLayout(right_layout)

        main_layout.addWidget(input_group)

        # Botones
        button_layout = QHBoxLayout()
        # calc_btn = QPushButton("Calcular")
        # calc_btn.clicked.connect(self.calculate)
        all_btn = QPushButton("Buscar todas")
        all_btn.clicked.connect(self.calculate_all)
        # Selector de método (menu escalable)
        self.method_combo = QComboBox()
        for mid, name, desc in get_available_methods():
            self.method_combo.addItem(name, mid)
        self.selected_method_id = self.method_combo.currentData()
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        execute_btn = QPushButton("Ejecutar método")
        execute_btn.clicked.connect(self.execute_selected_method)
        self.plot_btn = QPushButton("Mostrar grafica")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self.show_plot)
        clear_btn = QPushButton("Limpiar")
        clear_btn.clicked.connect(self.clear_table)

        # button_layout.addWidget(calc_btn)
        button_layout.addWidget(all_btn)
        button_layout.addWidget(self.method_combo)
        button_layout.addWidget(execute_btn)
        button_layout.addWidget(self.plot_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # Estado
        self.status_label = QLabel("Listo.")
        self.status_label.setStyleSheet("color: #1c3d5a;")
        main_layout.addWidget(self.status_label)


        # Tabulación (cambio de signo)
        self.sign_group = QGroupBox("Tabulacion para encontrar cambio de signo")
        sign_layout = QVBoxLayout()
        self.sign_group.setLayout(sign_layout)

        # filtro para mostrar solo cambios/no cambios
        self.sign_filter_combo = QComboBox()
        self.sign_filter_combo.addItem("Todos", "all")
        self.sign_filter_combo.addItem("Solo con cambio", "with")
        self.sign_filter_combo.addItem("Solo sin cambio", "without")
        self.sign_filter_combo.currentIndexChanged.connect(self._re_render_sign_table)
        sign_layout.addWidget(self.sign_filter_combo)

        tab_columns = ("Paso", "Lado", "Intervalo", "f(x_izq)", "f(x_der)", "Cambio", "Desarrollo")
        self.sign_table = QTreeWidget()
        self.sign_table.setColumnCount(len(tab_columns))
        self.sign_table.setHeaderLabels(list(tab_columns))
        self.sign_table.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.sign_table.header().setStretchLastSection(True)

        self.sign_table.itemClicked.connect(self._on_sign_item_clicked)
        sign_layout.addWidget(self.sign_table)
        main_layout.addWidget(self.sign_group)

        # Tabla del método
        table_group = QGroupBox("Tabla del metodo")
        table_layout = QVBoxLayout()
        table_group.setLayout(table_layout)

        columns = ("n", "a", "b", "xn", "f(a)", "f(b)", "f(xn)", "Error %")
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(list(columns))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        table_layout.addWidget(self.table)
        main_layout.addWidget(table_group)
        self.derivative_label.hide()
        self.derivative_edit.hide()

        # Footer técnico siempre visible en la esquina inferior derecha.
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        footer_label = QLabel(_get_runtime_signature())
        footer_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        footer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        footer_layout.addWidget(footer_label)
        main_layout.addLayout(footer_layout)

    def calculate(self) -> None:
        try:
            self.successive_result = None
            formula = self.formula_edit.text().strip()
            proposed_number = float(self.proposed_edit.text())
            max_tabulations = int(self.tab_max_edit.text())
            error_pct = float(self.error_edit.text())
            max_iter = int(self.max_iter_edit.text())

            if not formula:
                raise ValueError("Debe ingresar una formula en terminos de x.")
            if max_tabulations <= 0:
                raise ValueError("Las tabulaciones maximas deben ser mayores que 0.")
            if error_pct <= 0:
                raise ValueError("El error porcentual debe ser mayor que 0.")
            if max_iter <= 0:
                raise ValueError("Las iteraciones maximas deben ser mayores que 0.")

            self.compiled_function = compile_function(formula)
            self.formula_text = formula
            self.sign_search_result = find_first_sign_change(
                self.compiled_function, proposed_number, max_tabulations
            )
            self._render_sign_tabulation(self.sign_search_result)

            if self.sign_search_result.interval is None:
                raise ValueError(
                    "No se encontro cambio de signo en la tabulacion. Se evaluo desde el numero propuesto hacia +x y -x hasta el maximo indicado."
                )

            a0, b0 = self.sign_search_result.interval
            self.result = run_bisection(self.compiled_function, a0, b0, error_pct, max_iter)
            # guardar como lista de raices encontradas (comportamiento original: 1 raiz)
            self.found_roots = [(1, self.result)]
            self.plot_span = max(5.0, abs(a0), abs(b0)) * 1.5

            self._render_result(self.result)
            self.plot_btn.setEnabled(True)

        except (ValueError, FormulaError) as exc:
            self.plot_btn.setEnabled(False)
            self.status_label.setText("Error en el calculo.")
            QMessageBox.critical(self, "Entrada invalida", str(exc))

    def _render_result(self, result: BisectionResult) -> None:
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["n", "a", "b", "xn", "f(a)", "f(b)", "f(xn)", "Error %"])
        # limpiar tabla
        self.table.setRowCount(0)
        for rec in result.records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            error_text = "-" if rec.error_pct is None else f"{rec.error_pct:.6f}"
            values = [
                str(rec.iteration),
                f"{rec.a:.6f}",
                f"{rec.b:.6f}",
                f"{rec.xn:.6f}",
                f"{rec.fa:.6f}",
                f"{rec.fb:.6f}",
                f"{rec.fxn:.6f}",
                error_text,
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)

        start_a, start_b = result.sign_interval
        final_error = result.records[-1].error_pct

        if result.met_tolerance:
            if final_error is None:
                summary = f"Raiz aproximada: {result.root:.10f}. Cambio de signo en [{start_a:.6f}, {start_b:.6f}]."
            else:
                summary = (
                    f"Raiz aproximada: {result.root:.10f}. Error final: {final_error:.6f}%. Cambio de signo en [{start_a:.6f}, {start_b:.6f}]."
                )
        else:
            summary = (
                f"Se alcanzo el maximo de iteraciones. Ultima aproximacion: {result.root:.10f}. Cambio de signo en [{start_a:.6f}, {start_b:.6f}]."
            )

        self.status_label.setText(summary)

    def _render_successive_result(self, result: SuccessiveApproxResult) -> None:
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["n", "x0", "f(x0)", "g(x0)", "Error %"])
        self.table.setRowCount(0)

        for rec in result.records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            error_pct_text = "-" if rec.error_pct is None else f"{rec.error_pct:.6f}"
            values = [
                str(rec.iteration),
                f"{rec.xn:.6f}",
                f"{rec.fxn:.6f}",
                f"{rec.xn_next:.6f}",
                error_pct_text,
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)

        last_error = result.records[-1].error_abs if result.records else None
        if result.met_tolerance and last_error is not None:
            self.status_label.setText(
                f"Aproximaciones sucesivas: raiz aprox {result.root:.10f}, E_abs final {last_error:.6f}, m={result.m:.6f}."
            )
        else:
            self.status_label.setText(
                f"Aproximaciones sucesivas: maximo de iteraciones alcanzado, ultima aprox {result.root:.10f}, m={result.m:.6f}."
            )

    def _render_newton_raphson_result(self, result: NewtonRaphsonResult) -> None:
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["n", "x0", "f(x0)", "f'(x0)", "g(x0)", "Error abs"])
        self.table.setRowCount(0)

        for rec in result.records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                str(rec.iteration),
                f"{rec.xn:.6f}",
                f"{rec.fxn:.6f}",
                f"{rec.fpxn:.6f}",
                f"{rec.gxn:.6f}",
                f"{rec.error_abs:.6f}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)

        last_error = result.records[-1].error_abs if result.records else None
        if result.met_tolerance and last_error is not None:
            self.status_label.setText(
                f"Newton-Raphson: raiz aprox {result.root:.10f}, Error absoluto final {last_error:.6f}."
            )
        else:
            self.status_label.setText(
                f"Newton-Raphson: maximo de iteraciones alcanzado, ultima aprox {result.root:.10f}."
            )

    def _format_x(self, value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _replace_x_in_formula(self, x_value: float) -> str:
        replacement = f"({self._format_x(x_value)})"
        return re.sub(r"\bx\b", replacement, self.formula_text)

    def _render_sign_tabulation(self, search_result: SignChangeSearchResult) -> None:
        self.sign_table.clear()
        # Respect filter selection
        filter_mode = getattr(self, "sign_filter_combo", None)
        filter_value = filter_mode.currentData() if filter_mode is not None else "all"

        for rec in search_result.records:
            side = rec.direction
            interval_text = f"[{rec.x_left:.6f}, {rec.x_right:.6f}]"
            sign_text = "SI" if rec.has_sign_change else "NO"

            formula_left = self._replace_x_in_formula(rec.x_left)
            formula_right = self._replace_x_in_formula(rec.x_right)
            derivation = (
                f"f({self._format_x(rec.x_left)}) = {formula_left} = {rec.fx_left:.6f}\n"
                f"f({self._format_x(rec.x_right)}) = {formula_right} = {rec.fx_right:.6f}"
            )

            item = QTreeWidgetItem([
                str(rec.step),
                side,
                interval_text,
                f"{rec.fx_left:.6f}",
                f"{rec.fx_right:.6f}",
                sign_text,
                derivation,
            ])
            # almacenar datos del intervalo en el item (string) para uso al clic
            interval_data = f"{rec.x_left},{rec.x_right},{int(rec.has_sign_change)}"
            item.setData(0, Qt.UserRole, interval_data)
            # aplicar filtro
            if filter_value == "all" or (filter_value == "with" and rec.has_sign_change) or (
                filter_value == "without" and not rec.has_sign_change
            ):
                self.sign_table.addTopLevelItem(item)

    def _re_render_sign_table(self) -> None:
        # cuando se cambia el filtro, volver a renderizar usando el último search_result
        sr = self.sign_search_result
        if sr is None:
            return
        self._render_sign_tabulation(sr)

    def _on_method_changed(self) -> None:
        """Actualiza el método activo y oculta/muestra la tabulación de cambio de signo según el método."""
        self.selected_method_id = self.method_combo.currentData()
        # Ocultar tabulación si es Newton-Raphson
        if self.selected_method_id == "newton_raphson":
            self.sign_group.hide()
            self.derivative_label.show()
            self.derivative_edit.show()
        else:
            self.sign_group.show()
            self.derivative_label.hide()
            self.derivative_edit.hide()

    def _on_sign_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Maneja el clic en una fila de la tabulación. Si tiene cambio de signo,
        ejecuta el método activo en ese intervalo y muestra su desarrollo.
        """
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        try:
            parts = str(data).split(",")
            x_left = float(parts[0])
            x_right = float(parts[1])
            has_change = bool(int(parts[2]))
        except Exception:
            QMessageBox.warning(self, "Dato invalido", "No se pudo interpretar el intervalo asociado.")
            return

        if not has_change:
            QMessageBox.information(self, "Sin cambio", "El intervalo seleccionado no muestra cambio de signo.")
            return

        active_method = self.method_combo.currentData()

        if active_method == "aprox_sucesivas":
            try:
                error_abs = float(self.error_edit.text())
                max_iter = int(self.max_iter_edit.text())
                x0 = (x_left + x_right) / 2.0
                res = run_successive_approximations(
                    self.compiled_function,
                    x0,
                    x_left,
                    x_right,
                    error_abs,
                    max_iter,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Error en aproximaciones sucesivas", str(exc))
                return

            self.successive_result = res
            self.result = None
            self.found_roots = []
            self.plot_span = max(5.0, abs(x_left), abs(x_right)) * 1.5
            self._render_successive_result(res)
            self.plot_btn.setEnabled(True)

            fx1 = self.compiled_function(x_left)
            fx2 = self.compiled_function(x_right)
            fx0 = self.compiled_function(x0)
            gx0 = res.records[0].xn_next if res.records else x0

            dev_text = (
                "Desarrollo (aproximaciones sucesivas)\n\n"
                f"Intervalo con cambio: [{x_left:.6f}, {x_right:.6f}]\n"
                f"m = (y2 - y1)/(x2 - x1) = ({fx2:.6f} - {fx1:.6f})/({x_right:.6f} - {x_left:.6f}) = {res.m:.6f}\n"
                f"x0 = (x1 + x2)/2 = ({x_left:.6f} + {x_right:.6f})/2 = {x0:.6f}\n"
                f"f(x0) = {fx0:.6f}\n"
                f"g(x0) = (f(x0) - m*x0)/(-m) = ({fx0:.6f} - {res.m:.6f}*{x0:.6f})/({-res.m:.6f}) = {gx0:.6f}"
            )
            QMessageBox.information(self, "Desarrollo del metodo", dev_text)
            return

        # bisección y buscar_todas usan la misma ejecución al hacer clic en un intervalo
        try:
            res = run_bisection(self.compiled_function, x_left, x_right, float(self.error_edit.text()), int(self.max_iter_edit.text()))
        except Exception as exc:
            QMessageBox.critical(self, "Error en biseccion", str(exc))
            return

        self.result = res
        self.successive_result = None
        self.found_roots = [(1, res)]
        self.plot_span = max(5.0, abs(x_left), abs(x_right)) * 1.5
        self._render_result(res)
        self.plot_btn.setEnabled(True)

        first = res.records[0] if res.records else None
        if first is not None:
            dev_text = (
                "Desarrollo (biseccion)\n\n"
                f"Intervalo con cambio: [{x_left:.6f}, {x_right:.6f}]\n"
                f"x_n = (a + b)/2 = ({first.a:.6f} + {first.b:.6f})/2 = {first.xn:.6f}\n"
                f"f(a) = {first.fa:.6f}, f(b) = {first.fb:.6f}, f(x_n) = {first.fxn:.6f}"
            )
            QMessageBox.information(self, "Desarrollo del metodo", dev_text)

    def calculate_all(self) -> None:
        """Busca todos los intervalos con cambio de signo en la tabulacion y
        aplica biseccion a cada uno, mostrando las raices encontradas.
        """
        try:
            self.successive_result = None
            formula = self.formula_edit.text().strip()
            proposed_number = float(self.proposed_edit.text())
            max_tabulations = int(self.tab_max_edit.text())
            error_pct = float(self.error_edit.text())
            max_iter = int(self.max_iter_edit.text())

            if not formula:
                raise ValueError("Debe ingresar una formula en terminos de x.")

            self.compiled_function = compile_function(formula)
            self.formula_text = formula

            multi: MultipleSignChangeSearchResult = find_all_sign_changes(
                self.compiled_function, proposed_number, max_tabulations
            )
            # guardar para re-render y filtros
            self.sign_search_result = multi
            # _render_sign_tabulation acepta cualquier objeto con .records
            self._render_sign_tabulation(multi)

            if not multi.intervals:
                raise ValueError("No se encontraron intervalos con cambio de signo.")

            all_roots: list[Tuple[int, BisectionResult]] = []
            for idx, (a0, b0) in enumerate(multi.intervals, start=1):
                try:
                    res = run_bisection(self.compiled_function, a0, b0, error_pct, max_iter)
                except ValueError:
                    # si el intervalo no es válido para bisección, lo saltamos
                    continue
                all_roots.append((idx, res))

            if not all_roots:
                raise ValueError("No fue posible aplicar biseccion a los intervalos detectados.")

            # Mostrar todas las raices en un mensaje y poblar la tabla con registros concatenados
            self.table.setRowCount(0)
            roots_list: list[str] = []
            for idx, res in all_roots:
                roots_list.append(f"Raiz {idx}: {res.root:.10f} (met_tol={res.met_tolerance})")
                # insertar registros de la biseccion en la tabla, con prefijo del indice de raiz
                for rec in res.records:
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    error_text = "-" if rec.error_pct is None else f"{rec.error_pct:.6f}"
                    values = [
                        f"{idx}-{rec.iteration}",
                        f"{rec.a:.6f}",
                        f"{rec.b:.6f}",
                        f"{rec.xn:.6f}",
                        f"{rec.fa:.6f}",
                        f"{rec.fb:.6f}",
                        f"{rec.fxn:.6f}",
                        error_text,
                    ]
                    for col, val in enumerate(values):
                        item = QTableWidgetItem(val)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table.setItem(row, col, item)

            summary = "\n".join(roots_list)
            # guardar todas las raices encontradas para graficar y acceso posterior
            self.found_roots = all_roots
            # ajustar plot_span según extremos de todos los intervalos
            min_x = min((iv[0] for iv in multi.intervals), default=0.0)
            max_x = max((iv[1] for iv in multi.intervals), default=0.0)
            self.plot_span = max(5.0, abs(min_x), abs(max_x)) * 1.5

            self.status_label.setText(f"Se encontraron {len(all_roots)} raiz(es).")
            QMessageBox.information(self, "Raices encontradas", summary)
            # permitir graficar la primera raiz encontrada
            self.result = all_roots[0][1]
            self.plot_btn.setEnabled(True)

        except (ValueError, FormulaError) as exc:
            self.plot_btn.setEnabled(False)
            self.status_label.setText("Error en el calculo.")
            QMessageBox.critical(self, "Entrada invalida", str(exc))

    def execute_selected_method(self) -> None:
        """Ejecuta la acción asociada al método seleccionado en el combo.

        Para preservar la interfaz previa, si el método seleccionado es
        `biseccion` se delega a `calculate()` (comportamiento original).
        Para `buscar_todas` se delega a `calculate_all()`.
        Para `aprox_sucesivas` se delega a `calculate_successive_approximations()`.
        Para `newton_raphson` se delega a `calculate_newton_raphson()`.
        """
        current_id = self.method_combo.currentData()
        self.selected_method_id = current_id
        if current_id == "biseccion":
            self.calculate()
        elif current_id == "buscar_todas":
            self.calculate_all()
        elif current_id == "aprox_sucesivas":
            self.calculate_successive_approximations()
        elif current_id == "newton_raphson":
            self.calculate_newton_raphson()
        else:
            QMessageBox.warning(self, "Metodo desconocido", "El metodo seleccionado no está implementado.")

    def calculate_successive_approximations(self) -> None:
        """Ejecuta aproximaciones sucesivas de forma independiente a bisección."""
        try:
            # Preparar tabla de aproximaciones desde el inicio para evitar mostrar columnas previas.
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(["n", "x0", "f(x0)", "g(x0)", "Error %"])
            self.table.setRowCount(0)

            formula = self.formula_edit.text().strip()
            proposed_number = float(self.proposed_edit.text())
            max_tabulations = int(self.tab_max_edit.text())
            error_abs = float(self.error_edit.text())
            max_iter = int(self.max_iter_edit.text())

            if not formula:
                raise ValueError("Debe ingresar una formula en terminos de x.")
            if max_tabulations <= 0:
                raise ValueError("Las tabulaciones maximas deben ser mayores que 0.")
            if error_abs <= 0:
                raise ValueError("La tolerancia (E_abs) debe ser mayor que 0.")
            if max_iter <= 0:
                raise ValueError("Las iteraciones maximas deben ser mayores que 0.")

            self.compiled_function = compile_function(formula)
            self.formula_text = formula

            # Buscar todos los intervalos desde el número propuesto.
            multi = find_all_sign_changes(self.compiled_function, proposed_number, max_tabulations)
            self.sign_search_result = multi
            self._render_sign_tabulation(multi)

            if not multi.intervals:
                raise ValueError("No se encontraron intervalos con cambio de signo en la tabulacion.")

            all_results: list[Tuple[int, float, float, float, SuccessiveApproxResult]] = []
            for idx, (x1, x2) in enumerate(multi.intervals, start=1):
                x0 = (x1 + x2) / 2.0
                try:
                    res = run_successive_approximations(
                        self.compiled_function,
                        x0,
                        x1,
                        x2,
                        error_abs,
                        max_iter,
                    )
                except ValueError:
                    continue
                all_results.append((idx, x1, x2, x0, res))

            if not all_results:
                raise ValueError("No fue posible aplicar aproximaciones sucesivas en los intervalos detectados.")

            roots_summary: list[str] = []
            calc_steps: list[str] = []

            for idx, x1, x2, x0, res in all_results:
                fx1 = self.compiled_function(x1)
                fx2 = self.compiled_function(x2)
                gx0 = res.records[0].xn_next if res.records else x0

                roots_summary.append(
                    f"Raiz {idx}: {res.root:.10f}  [intervalo {x1:.6f}, {x2:.6f}]  x0={x0:.6f}  m={res.m:.6f}"
                )
                calc_steps.append(
                    (
                        f"Raiz {idx}\n"
                        f"m = (y2 - y1) / (x2 - x1) = ({fx2:.6f} - {fx1:.6f}) / ({x2:.6f} - {x1:.6f}) = {res.m:.6f}\n"
                        f"x0 = (x1 + x2) / 2 = ({x1:.6f} + {x2:.6f}) / 2 = {x0:.6f}\n"
                        f"g(x0) = (f(x0) - m*x0) / (-m) = ({res.records[0].fxn:.6f} - {res.m:.6f}*{x0:.6f}) / ({-res.m:.6f}) = {gx0:.6f}"
                    )
                )
                for rec in res.records:
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    error_pct_text = "-" if rec.error_pct is None else f"{rec.error_pct:.6f}"
                    values = [
                        str(rec.iteration),
                        f"{rec.xn:.6f}",
                        f"{rec.fxn:.6f}",
                        f"{rec.xn_next:.6f}",
                        error_pct_text,
                    ]
                    for col, val in enumerate(values):
                        item = QTableWidgetItem(val)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.table.setItem(row, col, item)

            self.successive_result = all_results[0][4]
            self.result = None
            self.found_roots = []
            # Ajustar plot_span según extremos de todos los intervalos
            min_x = min((all_results[i][1] for i in range(len(all_results))), default=0.0)
            max_x = max((all_results[i][2] for i in range(len(all_results))), default=0.0)
            self.plot_span = max(5.0, abs(min_x), abs(max_x)) * 1.5
            self.status_label.setText(f"Aproximaciones sucesivas: {len(all_results)} raiz(es) detectadas desde la tabulacion.")
            QMessageBox.information(
                self,
                "Raices encontradas",
                "\n".join(roots_summary) + "\n\nDesarrollo del metodo:\n\n" + "\n\n".join(calc_steps),
            )
            self.plot_btn.setEnabled(True)

        except (ValueError, FormulaError) as exc:
            self.plot_btn.setEnabled(False)
            self.status_label.setText("Error en el calculo.")
            QMessageBox.critical(self, "Entrada invalida", str(exc))

    def calculate_newton_raphson(self) -> None:
        """Ejecuta el método de Newton-Raphson."""
        try:
            # Preparar tabla de Newton-Raphson para evitar mostrar columnas previas.
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(["n", "x0", "f(x0)", "f'(x0)", "g(x0)", "Error abs"])
            self.table.setRowCount(0)

            formula = self.formula_edit.text().strip()
            x0 = float(self.proposed_edit.text())
            error_abs = float(self.error_edit.text())
            max_iter = int(self.max_iter_edit.text())

            if not formula:
                raise ValueError("Debe ingresar una formula en terminos de x.")
            if error_abs <= 0:
                raise ValueError("La tolerancia (E_abs) debe ser mayor que 0.")
            if max_iter <= 0:
                raise ValueError("Las iteraciones maximas deben ser mayores que 0.")

            self.compiled_function = compile_function(formula)
            self.compiled_derivative = compile_derivative(formula)
            self.derivative_edit.setText(get_derivative_expression(formula))
            self.formula_text = formula

            # Ejecutar Newton-Raphson
            res = run_newton_raphson(
                self.compiled_function,
                self.compiled_derivative,
                x0,
                error_abs,
                max_iter,
            )

            self.newton_raphson_result = res
            self.result = None
            self.successive_result = None
            self.found_roots = []
            self.plot_span = max(5.0, abs(x0)) * 1.5

            # Mostrar la tabla
            self._render_newton_raphson_result(res)
            self.plot_btn.setEnabled(True)

            # Mostrar desarrollo del primer paso
            if res.records:
                first = res.records[0]
                dev_text = (
                    "Desarrollo (Newton-Raphson)\n\n"
                    f"x0 = {first.xn:.6f}\n"
                    f"f(x0) = {first.fxn:.6f}\n"
                    f"f'(x0) = {first.fpxn:.6f}\n"
                    f"g(x0) = x0 - f(x0)/f'(x0) = {first.xn:.6f} - ({first.fxn:.6f})/({first.fpxn:.6f}) = {first.gxn:.6f}\n"
                    f"Error absoluto = |g(x0) - x0| = |{first.gxn:.6f} - {first.xn:.6f}| = {first.error_abs:.6f}"
                )
                QMessageBox.information(self, "Desarrollo del metodo", dev_text)

        except (ValueError, FormulaError) as exc:
            self.derivative_edit.clear()
            self.plot_btn.setEnabled(False)
            self.status_label.setText("Error en el calculo.")
            QMessageBox.critical(self, "Entrada invalida", str(exc))

    def clear_table(self, reset_status: bool = True) -> None:
        self.table.setRowCount(0)
        self.sign_table.clear()
        self.derivative_edit.clear()
        if reset_status:
            self.status_label.setText("Listo.")
            self.result = None
            self.successive_result = None
            self.newton_raphson_result = None
            self.sign_search_result = None
            self.plot_btn.setEnabled(False)

    def show_plot(self) -> None:
        if self.compiled_function is None or (self.result is None and self.successive_result is None and self.newton_raphson_result is None):
            QMessageBox.warning(self, "Sin datos", "Primero realice un calculo con algun metodo.")
            return
        method_name = "biseccion"
        root_markers: list[tuple[float, str, str]] = []

        # Mostrar raíz de Newton-Raphson si está disponible
        if self.newton_raphson_result is not None:
            try:
                rx = self.newton_raphson_result.root
                root_markers.append((rx, f"Raiz aprox: {rx:.6f}", "#d62828"))
                method_name = "Newton-Raphson"
            except Exception:
                pass
        # Mostrar raíz de aproximaciones sucesivas si está disponible
        elif self.successive_result is not None:
            try:
                rx = self.successive_result.root
                root_markers.append((rx, f"Raiz aprox: {rx:.6f}", "#d62828"))
                method_name = "aproximaciones sucesivas"
            except Exception:
                pass
        # Mostrar todas las raíces encontradas de bisección (si las hay)
        elif getattr(self, "found_roots", None):
            for idx, res in self.found_roots:
                try:
                    rx = res.root
                    root_markers.append((rx, f"Raiz {idx}: {rx:.6f}", "#2d6a9f"))
                except Exception:
                    continue
        else:
            root_x = self.result.root
            root_markers.append((root_x, f"Raiz aprox: {root_x:.6f}", "#d62828"))

        # Mantener referencia a la ventana para evitar que el GC la cierre
        self._plot_window = FunctionPlotWindow(
            self.compiled_function,
            self.plot_span,
            root_markers,
            method_name,
            None,
        )
        self._plot_window.show()
        self._plot_window.raise_()
        self._plot_window.activateWindow()


def run_app() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    app = QApplication([])
    window = BisectionApp()
    window.setMinimumSize(980, 620)
    window.show()
    app.exec()
