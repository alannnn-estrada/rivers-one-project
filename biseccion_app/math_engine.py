from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from sympy import Symbol, lambdify, diff, sstr
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class FormulaError(ValueError):
    """Se lanza cuando una expresión no puede ser analizada o evaluada."""


@dataclass
class IterationRecord:
    iteration: int
    a: float
    b: float
    xn: float
    fa: float
    fb: float
    fxn: float
    error_pct: Optional[float]


@dataclass
class BisectionResult:
    records: List[IterationRecord]
    root: float
    met_tolerance: bool
    sign_interval: Tuple[float, float]


@dataclass
class SignChangeRecord:
    step: int
    direction: str
    x_left: float
    x_right: float
    fx_left: float
    fx_right: float
    has_sign_change: bool


@dataclass
class SignChangeSearchResult:
    interval: Optional[Tuple[float, float]]
    records: List[SignChangeRecord]


@dataclass
class MultipleSignChangeSearchResult:
    intervals: List[Tuple[float, float]]
    records: List[SignChangeRecord]


@dataclass
class SuccessiveApproxRecord:
    iteration: int
    xn: float
    fxn: float
    xn_next: float
    error_abs: float
    error_pct: Optional[float]


@dataclass
class SuccessiveApproxResult:
    records: List[SuccessiveApproxRecord]
    root: float
    met_tolerance: bool
    m: float


@dataclass
class NewtonRaphsonRecord:
    iteration: int
    xn: float
    fxn: float
    fpxn: float
    gxn: float
    error_abs: float


@dataclass
class NewtonRaphsonResult:
    records: List[NewtonRaphsonRecord]
    root: float
    met_tolerance: bool


def compile_function(expression: str) -> Callable[[float], float]:
    """Compila una expresión de usuario en la variable x a una función numérica."""
    x = Symbol("x")

    # Preparar transformaciones para el parser (p. ej. multiplicación implícita y manejo de '^')
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        parsed = parse_expr(expression, transformations=transformations)
        numeric_fn = lambdify(x, parsed, modules=["numpy"])
    except Exception as exc:  # pragma: no cover - guardia defensiva al parsear
        raise FormulaError(f"No se pudo interpretar la formula: {exc}") from exc

    def safe_eval(value: float) -> float:
        try:
            result = numeric_fn(value)
            value_as_float = float(result)
            if not np.isfinite(value_as_float):
                raise FormulaError("La formula devuelve un valor no finito.")
            return value_as_float
        except FormulaError:
            raise
        except Exception as exc:  # pragma: no cover - guardia defensiva al evaluar
            raise FormulaError(f"Error evaluando la formula en x={value}: {exc}") from exc

    return safe_eval


def compile_derivative(expression: str) -> Callable[[float], float]:
    """Compila la derivada de una expresión en la variable x a una función numérica."""
    x = Symbol("x")

    # Preparar transformaciones para el parser (p. ej. multiplicación implícita y manejo de '^')
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        parsed = parse_expr(expression, transformations=transformations)
        derivative = diff(parsed, x)
        numeric_fn = lambdify(x, derivative, modules=["numpy"])
    except Exception as exc:  # pragma: no cover - guardia defensiva al parsear/derivar
        raise FormulaError(f"No se pudo calcular la derivada de la formula: {exc}") from exc

    def safe_eval(value: float) -> float:
        try:
            result = numeric_fn(value)
            value_as_float = float(result)
            if not np.isfinite(value_as_float):
                raise FormulaError("La derivada devuelve un valor no finito.")
            return value_as_float
        except FormulaError:
            raise
        except Exception as exc:  # pragma: no cover - guardia defensiva al evaluar
            raise FormulaError(f"Error evaluando la derivada en x={value}: {exc}") from exc

    return safe_eval


def get_derivative_expression(expression: str) -> str:
    """Devuelve la derivada simbólica como texto legible para la interfaz."""
    x = Symbol("x")
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        parsed = parse_expr(expression, transformations=transformations)
        derivative = diff(parsed, x)
        return sstr(derivative).replace("**", "^").replace("*", "")
    except Exception as exc:  # pragma: no cover - guardia defensiva al parsear/derivar
        raise FormulaError(f"No se pudo calcular la derivada de la formula: {exc}") from exc


def _has_sign_change(f_left: float, f_right: float) -> bool:
    return f_left == 0 or f_right == 0 or (f_left < 0 < f_right) or (f_right < 0 < f_left)


def find_first_sign_change(
    function: Callable[[float], float],
    proposed_number: float,
    max_tabulations: int,
) -> SignChangeSearchResult:
    """Encuentra el primer cambio de signo tabulando desde un x propuesto hacia ambos lados."""
    if max_tabulations <= 0:
        raise ValueError("Las tabulaciones maximas deben ser mayores que 0.")

    records: List[SignChangeRecord] = []
    center_x = float(proposed_number)
    f_center = function(center_x)

    if f_center == 0:
        records.append(
            SignChangeRecord(
                step=0,
                direction="centro",
                x_left=center_x,
                x_right=center_x,
                fx_left=f_center,
                fx_right=f_center,
                has_sign_change=True,
            )
        )
        return SignChangeSearchResult(interval=(center_x, center_x), records=records)

    prev_right_x = center_x
    prev_right_f = f_center
    prev_left_x = center_x
    prev_left_f = f_center

    # Iterativamente tabula en pasos crecientes: primero a la derecha, luego a la izquierda,
    # comprobando si existe un cambio de signo entre el punto previo y el punto actual.
    for step in range(1, max_tabulations + 1):
        right_x = center_x + float(step)
        right_f = function(right_x)
        right_has_change = _has_sign_change(prev_right_f, right_f)
        records.append(
            SignChangeRecord(
                step=step,
                direction="derecha",
                x_left=prev_right_x,
                x_right=right_x,
                fx_left=prev_right_f,
                fx_right=right_f,
                has_sign_change=right_has_change,
            )
        )
        if right_has_change:
            return SignChangeSearchResult(interval=(prev_right_x, right_x), records=records)
        prev_right_x, prev_right_f = right_x, right_f

        left_x = center_x - float(step)
        left_f = function(left_x)
        left_has_change = _has_sign_change(prev_left_f, left_f)
        records.append(
            SignChangeRecord(
                step=step,
                direction="izquierda",
                x_left=prev_left_x,
                x_right=left_x,
                fx_left=prev_left_f,
                fx_right=left_f,
                has_sign_change=left_has_change,
            )
        )
        if left_has_change:
            return SignChangeSearchResult(interval=(left_x, prev_left_x), records=records)
        prev_left_x, prev_left_f = left_x, left_f

    # No se encontró cambio de signo tras las tabulaciones solicitadas
    return SignChangeSearchResult(interval=None, records=records)


def find_all_sign_changes(
    function: Callable[[float], float],
    proposed_number: float,
    max_tabulations: int,
) -> MultipleSignChangeSearchResult:
    """Tabula desde `proposed_number` hacia ambos lados y devuelve
    todos los intervalos que muestran cambio de signo encontrados.

    Devuelve también los registros intermedios para ayudar en la trazabilidad.
    """
    if max_tabulations <= 0:
        raise ValueError("Las tabulaciones maximas deben ser mayores que 0.")

    records: List[SignChangeRecord] = []
    found_intervals: List[Tuple[float, float]] = []
    center_x = float(proposed_number)
    f_center = function(center_x)

    if f_center == 0:
        records.append(
            SignChangeRecord(
                step=0,
                direction="centro",
                x_left=center_x,
                x_right=center_x,
                fx_left=f_center,
                fx_right=f_center,
                has_sign_change=True,
            )
        )
        found_intervals.append((center_x, center_x))

    prev_right_x = center_x
    prev_right_f = f_center
    prev_left_x = center_x
    prev_left_f = f_center

    for step in range(1, max_tabulations + 1):
        right_x = center_x + float(step)
        right_f = function(right_x)
        right_has_change = _has_sign_change(prev_right_f, right_f)
        rec_r = SignChangeRecord(
            step=step,
            direction="derecha",
            x_left=prev_right_x,
            x_right=right_x,
            fx_left=prev_right_f,
            fx_right=right_f,
            has_sign_change=right_has_change,
        )
        records.append(rec_r)
        if right_has_change:
            found_intervals.append((prev_right_x, right_x))
        prev_right_x, prev_right_f = right_x, right_f

        left_x = center_x - float(step)
        left_f = function(left_x)
        left_has_change = _has_sign_change(prev_left_f, left_f)
        rec_l = SignChangeRecord(
            step=step,
            direction="izquierda",
            x_left=prev_left_x,
            x_right=left_x,
            fx_left=prev_left_f,
            fx_right=left_f,
            has_sign_change=left_has_change,
        )
        records.append(rec_l)
        if left_has_change:
            # mantener orden (x_izq, x_der)
            found_intervals.append((left_x, prev_left_x))
        prev_left_x, prev_left_f = left_x, left_f

    return MultipleSignChangeSearchResult(intervals=found_intervals, records=records)


def run_bisection(
    function: Callable[[float], float],
    a: float,
    b: float,
    error_tolerance_pct: float,
    max_iterations: int = 200,
) -> BisectionResult:
    """Ejecuta el método de la bisección hasta alcanzar el error porcentual objetivo aproximado."""
    if error_tolerance_pct <= 0:
        raise ValueError("El error porcentual debe ser mayor que 0.")
    if a > b:
        a, b = b, a

    fa = function(a)
    fb = function(b)
    
    if not _has_sign_change(fa, fb):
        raise ValueError("El intervalo inicial no tiene cambio de signo.")

    if fa == 0:
        return BisectionResult(
            records=[IterationRecord(1, a, b, a, fa, fb, fa, 0.0)],
            root=a,
            met_tolerance=True,
            sign_interval=(a, b),
        )

    if fb == 0:
        return BisectionResult(
            records=[IterationRecord(1, a, b, b, fa, fb, fb, 0.0)],
            root=b,
            met_tolerance=True,
            sign_interval=(a, b),
        )

    records: List[IterationRecord] = []
    previous_xn: Optional[float] = None

    # Bucle principal del método de la bisección. En cada iteración se calcula xn,
    # se evalúa f(xn) y se actualiza el intervalo [a, b] conservando el cambio de signo.
    for iteration in range(1, max_iterations + 1):
        xn = (a + b) / 2.0
        fxn = function(xn)

        if previous_xn is None:
            error_pct = None
        elif xn == 0:
            error_pct = abs(xn - previous_xn) * 100.0
        else:
            error_pct = abs((xn - previous_xn) / xn) * 100.0

        record = IterationRecord(
            iteration=iteration,
            a=a,
            b=b,
            xn=xn,
            fa=fa,
            fb=fb,
            fxn=fxn,
            error_pct=error_pct,
        )
        records.append(record)

        if fxn == 0 or (error_pct is not None and error_pct <= error_tolerance_pct):
            return BisectionResult(
                records=records,
                root=xn,
                met_tolerance=True,
                sign_interval=(records[0].a, records[0].b),
            )

        if fa * fxn < 0:
            b = xn
            fb = fxn
        else:
            a = xn
            fa = fxn

        previous_xn = xn

    return BisectionResult(
        records=records,
        root=records[-1].xn,
        met_tolerance=False,
        sign_interval=(records[0].a, records[0].b),
    )


def run_successive_approximations(
    function: Callable[[float], float],
    x0: float,
    x1: float,
    x2: float,
    error_tolerance_abs: float,
    max_iterations: int = 200,
) -> SuccessiveApproxResult:
    """Ejecuta aproximaciones sucesivas usando:

    x_(n+1) = (f(x_n) - m*x_n)/(-m),
    m = (f(x2)-f(x1))/(x2-x1)
    """
    if error_tolerance_abs <= 0:
        raise ValueError("La tolerancia absoluta debe ser mayor que 0.")
    if max_iterations <= 0:
        raise ValueError("El maximo de iteraciones debe ser mayor que 0.")
    if x2 == x1:
        raise ValueError("x1 y x2 no pueden ser iguales para calcular m.")

    fx1 = function(x1)
    fx2 = function(x2)
    m = (fx2 - fx1) / (x2 - x1)
    if m == 0:
        raise ValueError("m = 0. No se puede aplicar la formula de aproximaciones sucesivas.")

    records: List[SuccessiveApproxRecord] = []
    xn = float(x0)

    for iteration in range(1, max_iterations + 1):
        fxn = function(xn)
        xn_next = (fxn - m * xn) / (-m)
        error_abs = abs(xn_next - xn)
        if xn_next == 0:
            error_pct = None
        else:
            error_pct = abs((xn_next - xn) / xn_next) * 100.0

        records.append(
            SuccessiveApproxRecord(
                iteration=iteration,
                xn=xn,
                fxn=fxn,
                xn_next=xn_next,
                error_abs=error_abs,
                error_pct=error_pct,
            )
        )

        if fxn == 0 or error_abs <= error_tolerance_abs:
            return SuccessiveApproxResult(
                records=records,
                root=xn_next,
                met_tolerance=True,
                m=m,
            )

        xn = xn_next

    return SuccessiveApproxResult(
        records=records,
        root=records[-1].xn_next,
        met_tolerance=False,
        m=m,
    )


def run_newton_raphson(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    x0: float,
    error_tolerance_abs: float,
    max_iterations: int = 200,
) -> NewtonRaphsonResult:
    """Ejecuta el método de Newton-Raphson usando:

    x_(n+1) = x_n - f(x_n) / f'(x_n)
    
    Columnas a mostrar: n, x0, f(x0), f'(x0), g(x0), error_abs
    """
    if error_tolerance_abs <= 0:
        raise ValueError("La tolerancia absoluta debe ser mayor que 0.")
    if max_iterations <= 0:
        raise ValueError("El maximo de iteraciones debe ser mayor que 0.")

    records: List[NewtonRaphsonRecord] = []
    xn = float(x0)

    for iteration in range(1, max_iterations + 1):
        fxn = function(xn)
        fpxn = derivative(xn)
        
        if fpxn == 0:
            raise ValueError(f"La derivada es cero en x={xn}. No se puede aplicar Newton-Raphson.")

        gxn = xn - (fxn / fpxn)
        error_abs = abs(gxn - xn)

        records.append(
            NewtonRaphsonRecord(
                iteration=iteration,
                xn=xn,
                fxn=fxn,
                fpxn=fpxn,
                gxn=gxn,
                error_abs=error_abs,
            )
        )

        if fxn == 0 or error_abs <= error_tolerance_abs:
            return NewtonRaphsonResult(
                records=records,
                root=gxn,
                met_tolerance=True,
            )

        xn = gxn

    return NewtonRaphsonResult(
        records=records,
        root=records[-1].gxn,
        met_tolerance=False,
    )
