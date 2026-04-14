from __future__ import annotations

"""Interfaz gráfica para el método de la bisección.

Permite ingresar una fórmula en `x`, buscar un intervalo con cambio de signo,
ejecutar el método de la bisección, mostrar la tabla de iteraciones y la gráfica
de la función con la raíz aproximada marcada.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from biseccion_app.math_engine import (
    BisectionResult,
    FormulaError,
    SignChangeSearchResult,
    compile_function,
    find_first_sign_change,
    run_bisection,
)


class BisectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Metodo de biseccion")
        self.root.geometry("1080x680")

        self.compiled_function = None
        self.formula_text = ""
        self.sign_search_result: Optional[SignChangeSearchResult] = None
        self.plot_span = 5.0
        self.result: Optional[BisectionResult] = None

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Estilo dedicado para que las derivaciones multilínea no se recorten en la tabla.
        style = ttk.Style(self.root)
        style.configure("Sign.Treeview", rowheight=44)

        input_frame = ttk.LabelFrame(main, text="Entradas", padding=10)
        input_frame.pack(fill=tk.X)

        self.formula_var = tk.StringVar(value="x^3 + 2x^2 - 9")
        # Alternativa: dejar la fórmula en blanco al iniciar
        # self.formula_var = tk.StringVar(value="")
        self.proposed_var = tk.StringVar(value="0")
        self.tab_max_var = tk.StringVar(value="100")
        self.error_var = tk.StringVar(value="0.001")
        self.max_iter_var = tk.StringVar(value="100")
        self.status_var = tk.StringVar(value="Listo.")

        fields = [
            ("Formula f(x):", self.formula_var),
            ("Numero propuesto:", self.proposed_var),
            ("Tabulaciones maximas (busqueda):", self.tab_max_var),
            ("Error porcentual (%):", self.error_var),
            ("Max iteraciones:", self.max_iter_var),
        ]

        for row, (label_text, variable) in enumerate(fields):
            ttk.Label(input_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(input_frame, textvariable=variable, width=35).grid(row=row, column=1, sticky="we", pady=4)

        input_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(main, padding=(0, 10))
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Calcular", command=self.calculate).pack(side=tk.LEFT)
        self.plot_button = ttk.Button(button_frame, text="Mostrar grafica", command=self.show_plot, state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=8)
        ttk.Button(button_frame, text="Limpiar", command=self.clear_table).pack(side=tk.LEFT)

        status_label = ttk.Label(main, textvariable=self.status_var, foreground="#1c3d5a")
        status_label.pack(fill=tk.X, pady=(0, 8))

        tab_frame = ttk.LabelFrame(main, text="Tabulacion para encontrar cambio de signo", padding=8)
        tab_frame.pack(fill=tk.BOTH, expand=False)

        tab_columns = ("Paso", "Lado", "Intervalo", "f(x_izq)", "f(x_der)", "Cambio", "Desarrollo")
        self.sign_table = ttk.Treeview(tab_frame, columns=tab_columns, show="headings", height=8, style="Sign.Treeview")

        for col in tab_columns:
            self.sign_table.heading(col, text=col)
            if col == "Paso":
                width = 55
            elif col in ("Lado", "Cambio"):
                width = 90
            elif col == "Intervalo":
                width = 160
            elif col in ("f(x_izq)", "f(x_der)"):
                width = 90
            else:
                width = 470
            anchor = "w" if col == "Desarrollo" else "center"
            self.sign_table.column(col, width=width, anchor=anchor)

        tab_scroll_x = ttk.Scrollbar(tab_frame, orient=tk.HORIZONTAL, command=self.sign_table.xview)
        tab_scroll_y = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=self.sign_table.yview)
        self.sign_table.configure(xscrollcommand=tab_scroll_x.set, yscrollcommand=tab_scroll_y.set)

        self.sign_table.grid(row=0, column=0, sticky="nsew")
        tab_scroll_y.grid(row=0, column=1, sticky="ns")
        tab_scroll_x.grid(row=1, column=0, sticky="ew")
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)

        table_frame = ttk.LabelFrame(main, text="Tabla del metodo", padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("n", "a", "b", "xn", "f(a)", "f(b)", "f(xn)", "Error %")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=16)

        for col in columns:
            self.table.heading(col, text=col)
            width = 90 if col != "n" else 60
            self.table.column(col, width=width, anchor="center")

        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=y_scroll.set)

        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def calculate(self) -> None:
        # Valida entradas, compila la fórmula, busca cambio de signo y ejecuta bisección
        try:
            formula = self.formula_var.get().strip()
            proposed_number = float(self.proposed_var.get())
            max_tabulations = int(self.tab_max_var.get())
            error_pct = float(self.error_var.get())
            max_iter = int(self.max_iter_var.get())

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
                self.compiled_function,
                proposed_number,
                max_tabulations,
            )
            self._render_sign_tabulation(self.sign_search_result)

            if self.sign_search_result.interval is None:
                raise ValueError(
                    "No se encontro cambio de signo en la tabulacion. "
                    "Se evaluo desde el numero propuesto hacia +x y -x hasta el maximo indicado."
                )

            a0, b0 = self.sign_search_result.interval
            self.result = run_bisection(self.compiled_function, a0, b0, error_pct, max_iter)
            self.plot_span = max(5.0, abs(a0), abs(b0)) * 1.5

            self._render_result(self.result)
            self.plot_button.configure(state=tk.NORMAL)

        except (ValueError, FormulaError) as exc:
            self.plot_button.configure(state=tk.DISABLED)
            self.status_var.set("Error en el calculo.")
            messagebox.showerror("Entrada invalida", str(exc))

    def _render_result(self, result: BisectionResult) -> None:
        # Limpia la tabla y llena las filas con los registros de iteración
        self._clear_bisection_table()

        for rec in result.records:
            error_text = "-" if rec.error_pct is None else f"{rec.error_pct:.6f}"
            self.table.insert(
                "",
                tk.END,
                values=(
                    rec.iteration,
                    f"{rec.a:.6f}",
                    f"{rec.b:.6f}",
                    f"{rec.xn:.6f}",
                    f"{rec.fa:.6f}",
                    f"{rec.fb:.6f}",
                    f"{rec.fxn:.6f}",
                    error_text,
                ),
            )

        start_a, start_b = result.sign_interval
        final_error = result.records[-1].error_pct

        if result.met_tolerance:
            if final_error is None:
                summary = f"Raiz aproximada: {result.root:.10f}. Cambio de signo en [{start_a:.6f}, {start_b:.6f}]."
            else:
                summary = (
                    f"Raiz aproximada: {result.root:.10f}. "
                    f"Error final: {final_error:.6f}%. Cambio de signo en [{start_a:.6f}, {start_b:.6f}]."
                )
        else:
            summary = (
                f"Se alcanzo el maximo de iteraciones. Ultima aproximacion: {result.root:.10f}. "
                f"Cambio de signo en [{start_a:.6f}, {start_b:.6f}]."
            )

        self.status_var.set(summary)

    def _format_x(self, value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _replace_x_in_formula(self, x_value: float) -> str:
        replacement = f"({self._format_x(x_value)})"
        return re.sub(r"\bx\b", replacement, self.formula_text)

    def _render_sign_tabulation(self, search_result: SignChangeSearchResult) -> None:
        for item in self.sign_table.get_children():
            self.sign_table.delete(item)

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

            self.sign_table.insert(
                "",
                tk.END,
                values=(
                    rec.step,
                    side,
                    interval_text,
                    f"{rec.fx_left:.6f}",
                    f"{rec.fx_right:.6f}",
                    sign_text,
                    derivation,
                ),
            )

    def clear_table(self, reset_status: bool = True) -> None:
        self._clear_bisection_table()

        for item in self.sign_table.get_children():
            self.sign_table.delete(item)

        if reset_status:
            self.status_var.set("Listo.")
            self.result = None
            self.sign_search_result = None
            self.plot_button.configure(state=tk.DISABLED)

    def _clear_bisection_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def show_plot(self) -> None:
        # Muestra la gráfica de la función y marca la raíz aproximada
        if self.result is None or self.compiled_function is None:
            messagebox.showwarning("Sin datos", "Primero realice el calculo de biseccion.")
            return

        x_values = np.linspace(-self.plot_span, self.plot_span, 600)
        y_values = np.array([self.compiled_function(x) for x in x_values])

        plot_window = tk.Toplevel(self.root)
        plot_window.title("Grafica de la funcion")
        plot_window.geometry("900x580")

        figure = Figure(figsize=(8.5, 5.2), dpi=100)
        axis = figure.add_subplot(111)

        axis.plot(x_values, y_values, color="#2d6a9f", linewidth=2.0, label="f(x)")
        axis.axhline(0, color="#4f4f4f", linewidth=1.2)
        axis.axvline(0, color="#4f4f4f", linewidth=1.0, linestyle="--", alpha=0.75)

        root_x = self.result.root
        root_y = self.compiled_function(root_x)
        axis.scatter([root_x], [root_y], color="#d62828", s=55, zorder=5, label=f"Raiz aprox: {root_x:.6f}")

        axis.set_title("Funcion y raiz aproximada por biseccion")
        axis.set_xlabel("x")
        axis.set_ylabel("f(x)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

        canvas = FigureCanvasTkAgg(figure, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_window)
        toolbar_frame.pack(fill=tk.X)
        ttk.Button(toolbar_frame, text="Cerrar", command=plot_window.destroy).pack(side=tk.RIGHT, padx=8, pady=6)


def run_app() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    root = tk.Tk()
    app = BisectionApp(root)
    root.minsize(980, 620)
    root.mainloop()
