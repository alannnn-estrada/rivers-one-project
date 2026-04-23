# Metodo de Biseccion y Aproximaciones Sucesivas

Aplicacion de escritorio en Python para calcular raices con metodos numericos separados.

## Caracteristicas

- Entrada dinamica de formula `f(x)`.
- Entrada de numero propuesto para iniciar la tabulacion.
- Entrada de tabulaciones maximas para buscar el primer cambio de signo.
- Deteccion automatica del primer cambio de signo evaluando desde el numero propuesto hacia `+x` y `-x`.
- Tabla de tabulacion para cambio de signo con reemplazo visual de la formula y resultados evaluados.
- Tabla iterativa: `n`, `a`, `b`, `xn`, `f(a)`, `f(b)`, `f(xn)`, `Error %`.
- Metodo de aproximaciones sucesivas separado de biseccion con su propia tabla:
	`n`, `x_n`, `f(x_n)`, `x_(n+1)`, `E_abs`, `E_%`.
- Detencion automatica cuando el error porcentual aproximado es menor o igual al objetivo.
- Boton para mostrar grafica de la funcion y la raiz aproximada.
- Arquitectura modular para extender a mas metodos numericos.

## Instalacion

Se asume que existe un entorno virtual en `venv`.

```powershell
rtk .\venv\Scripts\python -m pip install -r requirements.txt
```

## Ejecucion

```powershell
rtk .\venv\Scripts\python main.py
```

## Sintaxis de formulas

- Variable permitida: `x`
- Potencias: `x^3` o `x**3`
- Multiplicacion implicita: `2x`, `3x^2`
- Ejemplo: `x^3 + 2x^2 - 9`

## Estructura

- `main.py`: punto de entrada.
- `biseccion_app/math_engine.py`: parser seguro de formulas y algoritmos numericos.
- `biseccion_app/ui.py`: interfaz PySide6, tablas de resultados y ventana de grafica.

## Notas de escalabilidad

- El motor numerico esta separado de la UI.
- Puedes agregar nuevos metodos (falsa posicion, newton, secante) en modulos nuevos dentro de `biseccion_app` sin acoplarlos a la ventana principal.
