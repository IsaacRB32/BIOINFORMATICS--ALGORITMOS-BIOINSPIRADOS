import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# IMPORTACIONES PARA EL DATASET ALON
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.datasets import load_svmlight_file

TAM_POBLACION = 20       # Debe ser par: los padres se cruzan por parejas
NUM_GENERACIONES = 20
PROB_CRUCE = 0.90        # Probabilidad de cruzar cada pareja
PROB_MUTACION = 0.30     # Probabilidad de invertir UN bit por hijo
MAX_GENES = 30           # Restricción provisional para Alon Colon Cancer
SEMILLA = 42
SEMILLAS_PRUEBA = [42, 52, 62]  # Tres ejecuciones para estudiar la estabilidad


## 1. GENERACIÓN DE LA POBLACIÓN INICIAL
## Cada posición del cromosoma se genera como 0 o 1
def generar_individuo(num_genes, max_genes):
    ## Generar directamente un cromosoma con pocos genes seleccionados
    individuo = np.zeros(num_genes, dtype=int)
    cantidad = np.random.randint(1, min(max_genes, num_genes) + 1)
    posiciones = np.random.choice(num_genes, cantidad, replace=False)
    individuo[posiciones] = 1
    return aplicar_restricciones(individuo, max_genes)


def generar_poblacion(num_genes, max_genes):
    poblacion = []
    intentos = 0

    while len(poblacion) < TAM_POBLACION:
        individuo = generar_individuo(num_genes, max_genes)

        ## Necesitamos que los individuos iniciales sean distintos
        repetido = any(np.array_equal(individuo, otro) for otro in poblacion)

        ## Límite de mil intentos 
        if not repetido or intentos >= 1000:
            poblacion.append(individuo)

        intentos += 1

    return poblacion


## 2. DECODIFICACIÓN
## Obtener las posiciones que tengan un 1
def decodificar(individuo):
    return np.where(individuo == 1)[0]


## RESTRICCIÓN ADICIONAL DE LA PRÁCTICA
## Debe haber al menos un gen seleccionado y nunca más de max_genes.
def aplicar_restricciones(individuo, max_genes):
    posiciones = decodificar(individuo)

    if len(posiciones) > max_genes:
        cuantos_quitar = len(posiciones) - max_genes
        quitar = np.random.choice(posiciones, cuantos_quitar, replace=False)
        individuo[quitar] = 0
    
    ## Si ningún gen está seleccionado, activar uno aleatoriamente
    if np.sum(individuo) == 0:
        posicion = np.random.randint(0, len(individuo))
        individuo[posicion] = 1

    return individuo


## 3. EVALUACIÓN EN LA FUNCIÓN OBJETIVO
## Un individuo es mejor si su selección de genes obtiene mayor balanced accuracy.
def calcular_aptitud(individuo, X, y, validacion, cache):
    ## Queremos utilizar el cromosoma como una clave dentro de un diccionario de Py
    clave = tuple(individuo)

    if clave in cache:
        return cache[clave]

    columnas = decodificar(individuo)

    clasificador = make_pipeline(
        ## puntuación estándar (z-score)
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    ## Evalúa el clasificador utilizando validación cruzada
    resultados = cross_val_score(
        clasificador, X[:, columnas], y,
        cv=validacion, scoring="balanced_accuracy"
    )

    aptitud = float(np.mean(resultados))
    cache[clave] = aptitud

    return aptitud


def evaluar_poblacion(poblacion, X, y, validacion, cache):
    aptitudes = []

    for individuo in poblacion:
        aptitudes.append(
            calcular_aptitud(individuo, X, y, validacion, cache)
        )

    return aptitudes


## 4. SELECCIÓN DE PADRES POR TORNEO (CON DOS PERMUTACIONES)
## Cada fila compara dos competidores
def seleccionar_padres(poblacion, aptitudes):
    n = len(poblacion)

    permutacion_1 = np.random.permutation(n)
    permutacion_2 = np.random.permutation(n)

    padres = []

    for i in range(n):
        a = permutacion_1[i]
        b = permutacion_2[i]

        ## MAXIMIZAMOS la FO: gana el individuo con mayor aptitud
        if aptitudes[a] >= aptitudes[b]:
            padres.append(poblacion[a].copy())
        else:
            padres.append(poblacion[b].copy())

    return padres


## 5. CRUZAMIENTO DE DOS PUNTOS
## Una pareja de padres puede producir dos hijos.
def cruzar(padre1, padre2):
    hijo1 = padre1.copy()
    hijo2 = padre2.copy()

    if len(padre1) >= 3 and np.random.random() < PROB_CRUCE:
        puntos = np.random.choice(
            range(1, len(padre1)), 2, replace=False
        )

        punto1, punto2 = sorted(puntos)

        hijo1[punto1:punto2] = padre2[punto1:punto2]
        hijo2[punto1:punto2] = padre1[punto1:punto2]

    return hijo1, hijo2


## 6. MUTACIÓN SIMPLE (BIT FLIP)
## Con probabilidad Pm, se selecciona un bit y se invierte.
def mutar(individuo):
    if np.random.random() < PROB_MUTACION:
        posicion = np.random.randint(0, len(individuo))
        individuo[posicion] = 1 - individuo[posicion]

    return individuo


## 7-9. SUSTITUCIÓN CON ELITISMO, DECODIFICACIÓN Y EVALUACIÓN
## El individuo más apto de la generación anterior pasa intacto a la nueva.
def ejecutar_ags(X, y, historial=None):
    num_genes = X.shape[1]
    max_genes = min(MAX_GENES, num_genes)

    validacion = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEMILLA
    )

    cache = {}

    ## Etapas 1, 2 y 3
    poblacion = generar_poblacion(num_genes, max_genes)
    aptitudes = evaluar_poblacion(poblacion, X, y, validacion, cache)

    for generacion in range(1, NUM_GENERACIONES + 1):

        ## Guardamos el mejor de la generación anterior (elitismo)
        indice_mejor = int(np.argmax(aptitudes))
        elite = poblacion[indice_mejor].copy()

        ## Etapa 4: torneos por dos permutaciones
        padres = seleccionar_padres(poblacion, aptitudes)

        ## Etapas 5 y 6: cruzamiento y mutación
        hijos = []

        for i in range(0, TAM_POBLACION, 2):
            hijo1, hijo2 = cruzar(padres[i], padres[i + 1])

            hijos.append(aplicar_restricciones(mutar(hijo1), max_genes))
            hijos.append(aplicar_restricciones(mutar(hijo2), max_genes))

        ## Etapa 7: sustitución generacional con un individuo élite
        poblacion = [elite] + hijos[:TAM_POBLACION - 1]

        ## Etapas 8 y 9: decodificación y evaluación de la nueva población
        aptitudes = evaluar_poblacion(
            poblacion, X, y, validacion, cache
        )

        indice_mejor = int(np.argmax(aptitudes))
        cantidad = len(decodificar(poblacion[indice_mejor]))

        print(
            f"Generación {generacion:02d} | "
            f"Aptitud: {aptitudes[indice_mejor]:.3f} | "
            f"Genes: {cantidad}"
        )

        if historial is not None:
            historial.append({
                "generacion": generacion,
                "aptitud": aptitudes[indice_mejor],
                "cantidad_genes": cantidad
            })

    indice_mejor = int(np.argmax(aptitudes))

    return poblacion[indice_mejor], aptitudes[indice_mejor]


## CLASIFICADOR Y MÉTRICAS 
def entrenar_y_predecir(X_entrena, y_entrena, X_prueba):
    clasificador = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    clasificador.fit(X_entrena, y_entrena)

    return clasificador.predict(X_prueba)


def mostrar_metricas(nombre, reales, predichas):
    print("\n" + nombre)

    print(f"Accuracy:          {accuracy_score(reales, predichas):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(reales, predichas):.3f}")
    print(f"Precision:         {precision_score(reales, predichas, zero_division=0):.3f}")
    print(f"Recall:            {recall_score(reales, predichas, zero_division=0):.3f}")
    print(f"F1-score:          {f1_score(reales, predichas, zero_division=0):.3f}")

    print("Matriz de confusión:")
    print(confusion_matrix(reales, predichas))


## CARGA DEL DATASET REAL ALON COLON CANCER
def cargar_alon():
    archivo = Path(__file__).resolve().parent / "colon-cancer.bz2"
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/colon-cancer.bz2"

    if not archivo.exists():
        print("Descargando Alon Colon Cancer desde LIBSVM...")
        try:
            urlretrieve(url, archivo)
        except Exception as error:
            if archivo.exists():
                archivo.unlink()
            raise RuntimeError(
                "No se pudo descargar colon-cancer.bz2. "
                "Descárgalo manualmente desde:\n" + url +
                "\ny colócalo junto a este archivo .py."
            ) from error

    X, etiquetas = load_svmlight_file(str(archivo), n_features=2000)
    X = X.toarray()
    etiquetas = np.asarray(etiquetas)

    if X.shape != (62, 2000):
        raise ValueError(f"Dimensiones inesperadas: {X.shape}. Se esperaban (62, 2000).")

    if not np.all(np.isfinite(X)):
        raise ValueError("El dataset contiene valores no finitos.")

    if set(np.unique(etiquetas)) != {-1.0, 1.0}:
        raise ValueError(f"Etiquetas inesperadas: {np.unique(etiquetas)}")

    # En la versión LIBSVM utilizada: -1 = tumoral; +1 = normal.
    y = (etiquetas == -1).astype(int)

    if np.sum(y == 1) != 40 or np.sum(y == 0) != 22:
        raise ValueError("Las clases no coinciden: se esperaban 40 tumorales y 22 normales.")

    return X, y


## MÉTRICAS PARA GUARDAR LOS RESULTADOS DE CADA PARTICIÓN
def obtener_metricas(reales, predichas):
    vn, fp, fn, vp = confusion_matrix(reales, predichas, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(reales, predichas),
        "balanced_accuracy": balanced_accuracy_score(reales, predichas),
        "precision": precision_score(reales, predichas, zero_division=0),
        "recall": recall_score(reales, predichas, zero_division=0),
        "f1_score": f1_score(reales, predichas, zero_division=0),
        "vn": int(vn),
        "fp": int(fp),
        "fn": int(fn),
        "vp": int(vp)
    }


## GUARDAR RESULTADOS EN ARCHIVOS CSV
def guardar_csv(ruta, columnas, filas):
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


## GRÁFICAS A PARTIR DE LOS ARCHIVOS CSV GENERADOS
def generar_graficas(carpeta):
    carpeta_graficas = carpeta / "graficas"
    carpeta_graficas.mkdir(exist_ok=True)

    ## Leer resultados globales
    with open(carpeta / "resumen.csv", encoding="utf-8-sig", newline="") as archivo:
        resumen = list(csv.DictReader(archivo))

    nombres = []
    exactitud = []
    exactitud_balanceada = []

    for fila in resumen:
        if fila["metodo"] == "Sin AGS":
            nombres.append("Sin AGS")
        else:
            nombres.append("AGS " + fila["semilla"])

        exactitud.append(float(fila["accuracy"]) * 100)
        exactitud_balanceada.append(float(fila["balanced_accuracy"]) * 100)

    ## Gráfica 1: comparación de métricas externas
    posiciones = np.arange(len(nombres))

    plt.figure(figsize=(9, 5))
    plt.bar(posiciones - 0.18, exactitud, width=0.36, label="Accuracy")
    plt.bar(posiciones + 0.18, exactitud_balanceada, width=0.36,
            label="Balanced accuracy")
    plt.xticks(posiciones, nombres)
    plt.ylim(0, 105)
    plt.ylabel("Porcentaje (%)")
    plt.title("Alon: clasificación externa con y sin AGS")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(carpeta_graficas / "01_comparacion_metricas.png", dpi=160)
    plt.close()

    ## Leer las aptitudes de todas las generaciones
    with open(carpeta / "evolucion_aptitud.csv", encoding="utf-8-sig", newline="") as archivo:
        registros = list(csv.DictReader(archivo))

    generaciones = sorted(set(int(fila["generacion"]) for fila in registros))
    promedios = []
    desviaciones = []

    for generacion in generaciones:
        valores = [float(fila["aptitud"]) for fila in registros
                   if int(fila["generacion"]) == generacion]

        promedios.append(float(np.mean(valores)))
        desviaciones.append(float(np.std(valores)))

    ## Gráfica 2: aptitud interna promedio y variabilidad entre ejecuciones
    promedios = np.array(promedios)
    desviaciones = np.array(desviaciones)

    plt.figure(figsize=(9, 5))
    linea, = plt.plot(generaciones, promedios, marker="o", markersize=3,
                      label="Aptitud promedio")

    plt.fill_between(generaciones, promedios - desviaciones,
                     promedios + desviaciones, alpha=0.18,
                     color=linea.get_color(), label="± 1 desviación estándar")

    plt.xlabel("Generación")
    plt.ylabel("Balanced accuracy interna")
    plt.title("Alon: evolución de la mejor aptitud")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta_graficas / "02_evolucion_aptitud.png", dpi=160)
    plt.close()

    ## Leer las frecuencias de los genes elegidos
    with open(carpeta / "frecuencia_genes.csv", encoding="utf-8-sig", newline="") as archivo:
        frecuencias = list(csv.DictReader(archivo))

    ## Gráfica 3: quince posiciones de genes seleccionadas con más frecuencia
    mas_frecuentes = frecuencias[:15][::-1]

    plt.figure(figsize=(9, 6))
    plt.barh([fila["posicion_gen"] for fila in mas_frecuentes],
             [int(fila["veces_seleccionado"]) for fila in mas_frecuentes])
    plt.xlabel("Cantidad de ejecuciones en las que apareció")
    plt.ylabel("Posición del gen (desde 1)")
    plt.title("Alon: genes seleccionados más frecuentemente")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(carpeta_graficas / "03_frecuencia_genes.png", dpi=160)
    plt.close()

    ## Leer cantidad de genes usados en cada partición
    with open(carpeta / "resultados_folds.csv", encoding="utf-8-sig", newline="") as archivo:
        folds = list(csv.DictReader(archivo))

    ## Gráfica 4: número de genes seleccionados en cada ejecución
    plt.figure(figsize=(9, 5))

    for semilla in SEMILLAS_PRUEBA:
        datos = [fila for fila in folds
                 if fila["metodo"] == "Con AGS" and fila["semilla"] == str(semilla)]

        datos.sort(key=lambda fila: int(fila["fold"]))

        plt.plot([int(fila["fold"]) for fila in datos],
                 [int(fila["genes"]) for fila in datos],
                 marker="o", label="Semilla " + str(semilla))

    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Partición externa")
    plt.ylabel("Genes seleccionados")
    plt.title("Alon: cantidad de genes por partición y semilla")
    plt.ylim(0, MAX_GENES + 2)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta_graficas / "04_cantidad_genes.png", dpi=160)
    plt.close()

    ## Gráficas 5-8: matriz de confusión para cada método
    for i, fila in enumerate(resumen, start=1):
        matriz = np.array([
            [int(fila["vn"]), int(fila["fp"])],
            [int(fila["fn"]), int(fila["vp"])]
        ])

        plt.figure(figsize=(5, 4.5))
        plt.imshow(matriz, cmap="Blues")
        plt.xticks([0, 1], ["Normal", "Tumoral"])
        plt.yticks([0, 1], ["Normal", "Tumoral"])
        plt.xlabel("Clase predicha")
        plt.ylabel("Clase real")

        nombre = nombres[i - 1]
        plt.title("Matriz de confusión: " + nombre)

        for r in range(2):
            for c in range(2):
                plt.text(c, r, str(matriz[r, c]), ha="center", va="center",
                         color="white" if matriz[r, c] > matriz.max() / 2 else "black")

        plt.tight_layout()
        plt.savefig(
            carpeta_graficas / f"0{i + 4}_matriz_{nombre.replace(' ', '_')}.png",
            dpi=160
        )
        plt.close()

    print("Gráficas guardadas en:", carpeta_graficas)


## EJECUCIÓN DEL AGS CON ALON COLON CANCER
def main():
    if TAM_POBLACION < 2 or TAM_POBLACION % 2 != 0:
        raise ValueError("TAM_POBLACION debe ser par y mayor o igual a 2")

    if MAX_GENES < 1:
        raise ValueError("MAX_GENES debe ser al menos 1")

    if not SEMILLAS_PRUEBA:
        raise ValueError("Debes indicar al menos una semilla")

    np.random.seed(SEMILLA)
    X, y = cargar_alon()

    print("\nDATASET REAL: ALON COLON CANCER")
    print("Muestras:", X.shape[0], "| Genes:", X.shape[1])
    print("Normales (0):", np.sum(y == 0), "| Tumorales (1):", np.sum(y == 1))
    print("Máximo de genes seleccionados:", min(MAX_GENES, X.shape[1]))
    print("Semillas del AGS:", SEMILLAS_PRUEBA)

    ## Validación cruzada externa: el AGS nunca ve las muestras de prueba.
    validacion_externa = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEMILLA
    )

    resultados_folds = []
    evolucion = []
    genes_elegidos = []
    predicciones = []

    pred_sin_ags = np.zeros(len(y), dtype=int)
    pred_con_ags = {}

    for semilla in SEMILLAS_PRUEBA:
        pred_con_ags[semilla] = np.zeros(len(y), dtype=int)

    for numero_fold, (indices_entrena, indices_prueba) in enumerate(
        validacion_externa.split(X, y), start=1
    ):
        X_entrena = X[indices_entrena]
        X_prueba = X[indices_prueba]
        y_entrena = y[indices_entrena]
        y_prueba = y[indices_prueba]

        print("\n" + "=" * 55)
        print(f"PARTICIÓN EXTERNA {numero_fold} DE 5")
        print("Entrenamiento:", len(y_entrena), "| Prueba:", len(y_prueba))

        ## Clasificación sin AGS: todos los genes, una vez por partición.
        pred_base = entrenar_y_predecir(X_entrena, y_entrena, X_prueba)
        pred_sin_ags[indices_prueba] = pred_base
        metricas_base = obtener_metricas(y_prueba, pred_base)

        resultados_folds.append({
            "semilla": "base",
            "fold": numero_fold,
            "metodo": "Sin AGS",
            "genes": X.shape[1],
            "aptitud_interna": "",
            **metricas_base
        })

        for indice, real, predicho in zip(indices_prueba, y_prueba, pred_base):
            predicciones.append({
                "semilla": "base", "fold": numero_fold,
                "metodo": "Sin AGS", "muestra": int(indice) + 1,
                "real": int(real), "prediccion": int(predicho)
            })

        ## Se repite el AGS con distintas semillas sobre el mismo entrenamiento.
        for semilla in SEMILLAS_PRUEBA:
            np.random.seed(semilla + numero_fold - 1)
            historial = []

            print(f"\nAGS | Partición {numero_fold} | Semilla {semilla}")
            mejor, aptitud = ejecutar_ags(X_entrena, y_entrena, historial)
            genes = decodificar(mejor)

            pred_ags = entrenar_y_predecir(
                X_entrena[:, genes], y_entrena, X_prueba[:, genes]
            )

            pred_con_ags[semilla][indices_prueba] = pred_ags
            metricas_ags = obtener_metricas(y_prueba, pred_ags)

            resultados_folds.append({
                "semilla": semilla,
                "fold": numero_fold,
                "metodo": "Con AGS",
                "genes": len(genes),
                "aptitud_interna": round(aptitud, 6),
                **metricas_ags
            })

            for registro in historial:
                evolucion.append({
                    "semilla": semilla,
                    "fold": numero_fold,
                    **registro
                })

            for gen in genes:
                genes_elegidos.append({
                    "semilla": semilla,
                    "fold": numero_fold,
                    "posicion_gen": int(gen) + 1,
                    "aptitud_interna": round(aptitud, 6)
                })

            for indice, real, predicho in zip(indices_prueba, y_prueba, pred_ags):
                predicciones.append({
                    "semilla": semilla, "fold": numero_fold,
                    "metodo": "Con AGS", "muestra": int(indice) + 1,
                    "real": int(real), "prediccion": int(predicho)
                })

            print("Genes seleccionados:", len(genes))
            print("Posiciones (desde 1):", (genes + 1).tolist())
            print("Aptitud interna:", round(aptitud, 3))
            print("Balanced accuracy externa:", round(metricas_ags["balanced_accuracy"], 3))

    ## Cada muestra fue evaluada una vez por semilla con un modelo que no la vio.
    resumen = []

    mostrar_metricas(
        "SIN AGS: predicciones externas de las 62 muestras",
        y, pred_sin_ags
    )

    resumen.append({
        "metodo": "Sin AGS",
        "semilla": "base",
        "genes_promedio": X.shape[1],
        **obtener_metricas(y, pred_sin_ags)
    })

    for semilla in SEMILLAS_PRUEBA:
        mostrar_metricas(
            f"CON AGS: predicciones externas con semilla {semilla}",
            y, pred_con_ags[semilla]
        )

        cantidades = [
            fila["genes"] for fila in resultados_folds
            if fila["metodo"] == "Con AGS" and fila["semilla"] == semilla
        ]

        resumen.append({
            "metodo": "Con AGS",
            "semilla": semilla,
            "genes_promedio": round(float(np.mean(cantidades)), 2),
            **obtener_metricas(y, pred_con_ags[semilla])
        })

    ## Los promedios entre semillas miden variación de la búsqueda genética.
    print("\nPROMEDIOS ENTRE SEMILLAS (AGS)")

    for metrica in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]:
        valores = [fila[metrica] for fila in resumen if fila["metodo"] == "Con AGS"]
        promedio = np.mean(valores)
        desviacion = np.std(valores, ddof=1) if len(valores) > 1 else 0.0
        print(f"{metrica}: {promedio:.3f} ± {desviacion:.3f}")

    ## Frecuencia de selección de cada posición entre las 15 ejecuciones.
    frecuencia = {}

    for fila in genes_elegidos:
        gen = fila["posicion_gen"]
        frecuencia[gen] = frecuencia.get(gen, 0) + 1

    frecuencia_genes = [
        {"posicion_gen": gen, "veces_seleccionado": veces}
        for gen, veces in sorted(
            frecuencia.items(),
            key=lambda dato: (-dato[1], dato[0])
        )
    ]

    carpeta = Path(__file__).resolve().parent / f"resultados_alon_{MAX_GENES}_genes"
    carpeta.mkdir(exist_ok=True)

    columnas_metricas = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1_score", "vn", "fp", "fn", "vp"
    ]

    guardar_csv(
        carpeta / "resultados_folds.csv",
        ["semilla", "fold", "metodo", "genes", "aptitud_interna"] + columnas_metricas,
        resultados_folds
    )

    guardar_csv(
        carpeta / "evolucion_aptitud.csv",
        ["semilla", "fold", "generacion", "aptitud", "cantidad_genes"],
        evolucion
    )

    guardar_csv(
        carpeta / "genes_seleccionados.csv",
        ["semilla", "fold", "posicion_gen", "aptitud_interna"],
        genes_elegidos
    )

    guardar_csv(
        carpeta / "frecuencia_genes.csv",
        ["posicion_gen", "veces_seleccionado"],
        frecuencia_genes
    )

    guardar_csv(
        carpeta / "predicciones_externas.csv",
        ["semilla", "fold", "metodo", "muestra", "real", "prediccion"],
        predicciones
    )

    guardar_csv(
        carpeta / "resumen.csv",
        ["metodo", "semilla", "genes_promedio"] + columnas_metricas,
        resumen
    )

    generar_graficas(carpeta)

    print("\nResultados guardados en:", carpeta)
    print("IMPORTANTE: la versión LIBSVM fue normalizada previamente con todas las muestras.")
    print("La validación externa evita usar las etiquetas de prueba para seleccionar genes,")
    print("pero NO elimina el posible sesgo del preprocesamiento previo del proveedor.")


if __name__ == "__main__":
    main()
