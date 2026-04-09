from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from sympy import Symbol, lambdify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class FormulaError(ValueError):
    """Raised when an expression cannot be parsed or evaluated."""


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


def compile_function(expression: str) -> Callable[[float], float]:
    """Compile a user expression in variable x into a numeric callable."""
    x = Symbol("x")

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        parsed = parse_expr(expression, transformations=transformations)
        numeric_fn = lambdify(x, parsed, modules=["numpy"])
    except Exception as exc:  # pragma: no cover - defensive parse guard
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
        except Exception as exc:  # pragma: no cover - defensive eval guard
            raise FormulaError(f"Error evaluando la formula en x={value}: {exc}") from exc

    return safe_eval


def _has_sign_change(f_left: float, f_right: float) -> bool:
    return f_left == 0 or f_right == 0 or (f_left < 0 < f_right) or (f_right < 0 < f_left)


def find_first_sign_change(
    function: Callable[[float], float],
    proposed_number: float,
    max_tabulations: int,
) -> SignChangeSearchResult:
    """Find the first sign change by tabulating from a proposed x to both sides."""
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

    return SignChangeSearchResult(interval=None, records=records)


def run_bisection(
    function: Callable[[float], float],
    a: float,
    b: float,
    error_tolerance_pct: float,
    max_iterations: int = 200,
) -> BisectionResult:
    """Run bisection until the target approximate percent error is reached."""
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
