import copy
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy import stats

SEMILLA = 42
np.random.seed(SEMILLA)

sigmoid = (
    lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    lambda a: a * (1 - a),
)

relu = (
    lambda x: x * (x > 0),
    lambda a: (a > 0).astype(float),
)


class capa:
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.round(
            stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas).reshape(1, n_neuronas),
            3,
        )
        self.W = np.round(
            stats.truncnorm.rvs(
                -1, 1, loc=0, scale=1, size=n_neuronas * n_neuronas_capa_anterior
            ).reshape(n_neuronas_capa_anterior, n_neuronas),
            3,
        )


def crear_red(neuronas, funciones_activacion):
    return [
        capa(neuronas[i], neuronas[i + 1], funciones_activacion[i])
        for i in range(len(neuronas) - 1)
    ]


def mse(Ypredich, Yreal):
    diferencia = np.array(Ypredich) - np.array(Yreal)
    error = np.mean(diferencia**2)
    derivada = diferencia
    return error, derivada


def forward(X, red_neuronal):
    output = [X]
    for c in red_neuronal:
        z = output[-1] @ c.W + c.b
        a = c.funcion_act[0](z)
        output.append(a)
    return output


def predecir(X, red_neuronal):
    return forward(X, red_neuronal)[-1]


def entrenamiento(X, Y, red_neuronal, lr=0.01):
    output = forward(X, red_neuronal)

    back = list(range(len(output) - 1))
    back.reverse()

    delta = []

    for i in back:
        a = output[i + 1]

        if i == back[0]:
            x = mse(a, Y)[1] * red_neuronal[i].funcion_act[1](a)
        else:
            x = delta[-1] @ W_temp * red_neuronal[i].funcion_act[1](a)
        delta.append(x)

        W_temp = red_neuronal[i].W.transpose()

        red_neuronal[i].b = red_neuronal[i].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        red_neuronal[i].W = red_neuronal[i].W - output[i].transpose() @ delta[-1] * lr

    return output[-1]


def circulo(num_datos=100, R=1, minimo=0, maximo=1):
    pi = math.pi
    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    x = np.round(x.reshape((num_datos, 1)), 3)
    y = np.round(y.reshape((num_datos, 1)), 3)

    return np.column_stack([x, y])


FUENTE = "Arial"


def _fuente(**kwargs):
    return Font(name=FUENTE, **kwargs)


def _hoja_pesos(ws, red, titulo, formato_num, nombres_act):
    lado = Side(style="thin", color="999999")
    caja = Border(left=lado, right=lado, top=lado, bottom=lado)
    relleno = PatternFill("solid", start_color="D9E1F2")

    ws["A1"] = titulo
    ws["A1"].font = _fuente(bold=True, size=14)

    fila = 3
    for k, c in enumerate(red, start=1):
        n_ent, n_sal = c.W.shape
        salida = " (salida)" if k == len(red) else ""
        ws.cell(
            fila, 1, f"Capa {k}{salida}: {n_ent} -> {n_sal} neuronas, activacion {nombres_act[k - 1]}"
        ).font = _fuente(bold=True)
        fila += 1

        ws.cell(fila, 1, "W")
        for j in range(n_sal):
            ws.cell(fila, 2 + j, f"Neurona {j + 1}")
        for col in range(1, n_sal + 2):
            celda = ws.cell(fila, col)
            celda.font = _fuente(bold=True)
            celda.fill = relleno
            celda.border = caja
            celda.alignment = Alignment(horizontal="center")
        fila += 1

        for i in range(n_ent):
            ws.cell(fila, 1, f"Entrada x{i + 1}" if k == 1 else f"Neurona {i + 1} (capa {k - 1})")
            for j in range(n_sal):
                ws.cell(fila, 2 + j, float(c.W[i, j])).number_format = formato_num
            for col in range(1, n_sal + 2):
                celda = ws.cell(fila, col)
                celda.font = _fuente(bold=(col == 1))
                celda.border = caja
            fila += 1

        ws.cell(fila, 1, "b (bias)")
        for j in range(n_sal):
            ws.cell(fila, 2 + j, float(c.b[0, j])).number_format = formato_num
        for col in range(1, n_sal + 2):
            celda = ws.cell(fila, col)
            celda.font = _fuente(bold=(col == 1))
            celda.border = caja
        fila += 2

    ws.column_dimensions["A"].width = 28
    for col in range(2, 10):
        ws.column_dimensions[get_column_letter(col)].width = 14


def exportar_pesos_excel(ruta, red_inicial, red_final, X, Y, pred_final, config):
    wb = Workbook()
    nombres_act = config["nombres_act"]
    n = len(X)
    ultima = n + 1

    ws = wb.active
    ws.title = "Resumen"
    ws["A1"] = "Red neuronal desde cero en Python - pesos generados"
    ws["A1"].font = _fuente(bold=True, size=14)

    filas = [
        ("Arquitectura (neuronas por capa)", " -> ".join(str(v) for v in config["neuronas"]), None),
        ("Funciones de activacion (por capa)", ", ".join(nombres_act), None),
        ("Funcion de coste", "Error cuadratico medio (MSE)", None),
        ("Optimizador", "Gradient descent (backpropagation)", None),
        ("Learning rate", config["lr"], "0.000"),
        ("Epochs", config["epochs"], "0"),
        ("Semilla aleatoria", config["semilla"], "0"),
        ("Datos", f"{n} puntos: 150 de clase 0 (radio 2) y 150 de clase 1 (radio 0.5)", None),
        ("MSE final (calculado en Python)", config["mse"], "0.0000"),
        ("Precision final (calculada en Python)", config["precision"], "0.00%"),
        (
            "MSE final (recalculado en Excel con formula)",
            f"=SUMPRODUCT((Datos_predicciones!D2:D{ultima}-Datos_predicciones!C2:C{ultima})^2)"
            f"/COUNT(Datos_predicciones!C2:C{ultima})",
            "0.0000",
        ),
        (
            "Precision final (recalculada en Excel con formula)",
            f"=AVERAGE(Datos_predicciones!F2:F{ultima})",
            "0.00%",
        ),
    ]

    ws["A3"], ws["B3"] = "Parametro", "Valor"
    for celda in (ws["A3"], ws["B3"]):
        celda.font = _fuente(bold=True)
        celda.fill = PatternFill("solid", start_color="D9E1F2")
    fila = 4
    for etiqueta, valor, formato in filas:
        ws.cell(fila, 1, etiqueta).font = _fuente()
        celda = ws.cell(fila, 2, valor)
        celda.font = _fuente()
        celda.alignment = Alignment(horizontal="left")
        if formato:
            celda.number_format = formato
        fila += 1

    fila += 1
    ws.cell(fila, 1, "Hojas del archivo").font = _fuente(bold=True)
    fila += 1
    notas = [
        "Pesos_iniciales: W y b de cada capa antes de entrenar (valores aleatorios, redondeados a 3 decimales).",
        "Pesos_finales: W y b de cada capa despues de los entrenamientos (precision completa).",
        "Datos_predicciones: los 300 puntos, la clase real, la salida de la red y si acierta.",
        "",
        "Como leer los pesos",
        "W: cada fila es una neurona de la capa anterior (o una entrada) y cada columna una neurona de la capa.",
        "b: un bias por cada neurona de la capa.",
        "Calculo de cada capa: z = entrada @ W + b ; salida = activacion(z).",
        "Las columnas de acierto, MSE y precision de Excel son formulas: se recalculan solas.",
        "Fuente del codigo: https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/",
    ]
    for nota in notas:
        celda = ws.cell(fila, 1, nota)
        celda.font = _fuente(bold=(nota == "Como leer los pesos"))
        fila += 1
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 70

    _hoja_pesos(
        wb.create_sheet("Pesos_iniciales"), red_inicial,
        "Pesos iniciales (antes de entrenar)", "0.000", nombres_act,
    )
    _hoja_pesos(
        wb.create_sheet("Pesos_finales"), red_final,
        "Pesos finales (despues de entrenar)", "0.000000", nombres_act,
    )

    wd = wb.create_sheet("Datos_predicciones")
    encabezados = ["x1", "x2", "Clase real", "Salida de la red (sigmoide)", "Clase predicha", "Acierto (1 = si)"]
    for col, texto in enumerate(encabezados, start=1):
        celda = wd.cell(1, col, texto)
        celda.font = _fuente(bold=True)
        celda.fill = PatternFill("solid", start_color="D9E1F2")
        celda.alignment = Alignment(horizontal="center", wrap_text=True)
    for i in range(n):
        fila = i + 2
        wd.cell(fila, 1, float(X[i, 0]))
        wd.cell(fila, 2, float(X[i, 1]))
        wd.cell(fila, 3, int(Y[i, 0]))
        wd.cell(fila, 4, float(pred_final[i, 0])).number_format = "0.000000"
        wd.cell(fila, 5, int(np.round(pred_final[i, 0])))
        wd.cell(fila, 6, f"=IF(E{fila}=C{fila},1,0)")
        for col in range(1, 7):
            wd.cell(fila, col).font = _fuente()
    for col, ancho in zip("ABCDEF", (12, 12, 12, 18, 14, 14)):
        wd.column_dimensions[col].width = ancho
    wd.freeze_panes = "A2"

    wb.save(ruta)


def main():
    datos_1 = circulo(num_datos=150, R=2)
    datos_2 = circulo(num_datos=150, R=0.5)
    X = np.round(np.concatenate([datos_1, datos_2]), 3)
    Y = np.array([0] * 150 + [1] * 150).reshape(-1, 1)

    neuronas = [2, 4, 8, 1]
    funciones_activacion = [relu, relu, sigmoid]
    red_neuronal = crear_red(neuronas, funciones_activacion)
    red_inicial = copy.deepcopy(red_neuronal)

    epochs = 1000
    lr = 0.001
    error = []
    for epoch in range(epochs):
        prediccion = entrenamiento(X, Y, red_neuronal, lr=lr)
        error.append(mse(np.round(prediccion), Y)[0])

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | error: {error[-1]:.4f}")

    pred_final = predecir(X, red_neuronal)
    precision = np.mean(np.round(pred_final) == Y)
    print(f"\nMSE final: {mse(pred_final, Y)[0]:.4f}")
    print(f"Precision final: {precision:.2%}")

    carpeta = os.environ.get("OUTPUT_DIR", ".")
    os.makedirs(carpeta, exist_ok=True)

    ruta_excel = os.path.join(carpeta, "pesos.xlsx")
    exportar_pesos_excel(
        ruta_excel,
        red_inicial,
        red_neuronal,
        X,
        Y,
        pred_final,
        config={
            "neuronas": neuronas,
            "nombres_act": ["ReLU", "ReLU", "Sigmoide"],
            "lr": lr,
            "epochs": epochs,
            "semilla": SEMILLA,
            "mse": float(mse(pred_final, Y)[0]),
            "precision": float(precision),
        },
    )
    print(f"Pesos guardados en: {ruta_excel}")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    axes[0].scatter(X[:150, 0], X[:150, 1], c="b", label="Clase 0")
    axes[0].scatter(X[150:, 0], X[150:, 1], c="r", label="Clase 1")
    axes[0].set_title("Problema de clasificacion")
    axes[0].legend()

    axes[1].plot(range(epochs), error)
    axes[1].set_title("Evolucion del error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")

    fig.tight_layout()

    ruta = os.path.join(carpeta, "resultados.png")
    fig.savefig(ruta, dpi=120)
    print(f"Graficas guardadas en: {ruta}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
