"""Registro ligero de métodos numéricos disponibles.

Este módulo expone `get_available_methods()` que devuelve una lista de
tuplas `(method_id, display_name, description)` usadas por la UI para
llenar el selector de métodos. La idea es permitir agregar módulos
por método en el paquete `biseccion_app.methods` en el futuro.
"""
from typing import List, Tuple


def get_available_methods() -> List[Tuple[str, str, str]]:
    """Devuelve la lista de métodos disponibles (id, nombre, descripción).

    Actualmente incluye:
    - 'biseccion': comportamiento clásico (primera raíz encontrada)
    - 'buscar_todas': tabulación que devuelve todas las raíces encontradas
    - 'aprox_sucesivas': método de aproximaciones sucesivas
    - 'newton_raphson': método de Newton-Raphson
    """
    return [
        ("biseccion", "Bisección (primera raíz)", "Encuentra la primera raíz con bisección"),
        ("buscar_todas", "Buscar todas (todas las raíces)", "Tabula y aplica bisección a todos los intervalos con cambio de signo"),
        (
            "aprox_sucesivas",
            "Aproximaciones sucesivas",
            "Tabula desde el numero propuesto y aplica aproximaciones sucesivas por intervalo",
        ),
        (
            "newton_raphson",
            "Newton-Raphson",
            "Método de Newton-Raphson usando derivada",
        ),
    ]
