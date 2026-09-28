import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

## PARÁMETROS GENERALES DEL EXPERIMENTO
NUM_PARTICULAS = 100
NUM_ITERACIONES = 100
NUM_EJECUCIONES = 30

W = 0.7      ## Inercia
C1 = 1.5     ## Aprendizaje SOCIAL
C2 = 1.5     ## Aprendizaje COGNITIVO

PORCENTAJE_VMAX = 0.2
TOLERANCIA = 1e-12

CARPETA_GRAFICAS = "graficas_practica3"
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)


## PROBLEMA 1 - HEAT EXCHANGER DESIGN
LIMITES_INFERIORES_HEAT = np.array([
    100, 1000, 1000, 10, 10, 10, 10, 10
], dtype=float)

LIMITES_SUPERIORES_HEAT = np.array([
    10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000
], dtype=float)


def objetivo_heat_exchanger(x):
    return x[0] + x[1] + x[2]


def restricciones_heat_exchanger(x):

    x1, x2, x3, x4, x5, x6, x7, x8 = x

    ## Todas las restricciones se expresan como g(x) <= 0.
    ## Se parte de la formulación original normalizada <= 1.

    g1 = (
        833.33252 * x4 / (x1 * x6)
        + 100.0 / x6
        - 83333.333 / (x1 * x6)
        - 1.0
    )

    g2 = (
        1250.0 * x5 / (x2 * x7)
        + x4 / x7
        - 1250.0 * x4 / (x2 * x7)
        - 1.0
    )

    g3 = (
        1250000.0 / (x3 * x8)
        + x5 / x8
        - 2500.0 * x5 / (x3 * x8)
        - 1.0
    )

    g4 = 0.0025 * x4 + 0.0025 * x6 - 1.0

    g5 = -0.0025 * x4 + 0.0025 * x5 + 0.0025 * x7 - 1.0

    g6 = 0.01 * x8 - 0.01 * x5 - 1.0

    return np.array([g1, g2, g3, g4, g5, g6])


## PROBLEMA 2 - OPTIMAL REACTOR DESIGN

## Arreglos de [0.1,...,0.1]
LIMITES_INFERIORES_REACTOR = np.full(8, 0.1, dtype=float)
## Arreglos de [10.0,...,10.0]
LIMITES_SUPERIORES_REACTOR = np.full(8, 10.0, dtype=float)


def objetivo_reactor(x):

    x1, x2, x3, x4, x5, x6, x7, x8 = x

    return (
        0.4 * (x1 / x7) ** 0.67
        + 0.4 * (x2 / x8) ** 0.67
        + 10.0
        - x1
        - x2
    )


def restricciones_reactor(x):

    x1, x2, x3, x4, x5, x6, x7, x8 = x

    ## Todas las restricciones se expresan como g(x) <= 0.
    ## La formulación original está dada como expresión <= 1.

    g1 = 0.0588 * x5 * x7 + 0.1 * x1 - 1.0

    g2 = 0.0588 * x6 * x8 + 0.1 * x1 + 0.1 * x2 - 1.0

    g3 = (
        4.0 * x3 / x5
        + 2.0 / (x3 ** 0.71 * x5)
        + 0.0588 * x7 / (x3 ** 1.3)
        - 1.0
    )

    g4 = (
        4.0 * x4 / x6
        + 2.0 / (x4 ** 0.71 * x6)
        + 0.0588 * x8 / (x4 ** 1.3)
        - 1.0
    )

    return np.array([g1, g2, g3, g4])


## SVR - SUMA DE VIOLACIÓN DE RESTRICCIONES
def calcular_svr(x, funcion_restricciones):

    restricciones = funcion_restricciones(x)

    ## Si g(x) <= 0 se cumple la restricción.
    ## Solo se suman las partes positivas de las violaciones.
    violaciones = np.maximum(0.0, restricciones)

    return np.sum(violaciones)


## REGLAS DE DEB
def es_mejor(aptitud_a, svr_a, aptitud_b, svr_b):
    ## En el apunte SVR = 0 es viable 
    viable_a = svr_a <= TOLERANCIA
    viable_b = svr_b <= TOLERANCIA

    ## Regla 1: Si ambas son viables, se toma la de mejor aptitud.
    if viable_a and viable_b:
        return aptitud_a < aptitud_b

    ## Regla 2: Si una es viable y la otra no, se toma la viable.
    if viable_a and not viable_b:
        return True

    if not viable_a and viable_b:
        return False

    ## Regla 3: Si ambas son inviables, se toma la de menor SVR.
    if svr_a < svr_b:
        return True

    ## En caso de empate en SVR, se usa la aptitud como desempate.
    if np.isclose(svr_a, svr_b):
        return aptitud_a < aptitud_b

    return False


## MÉTRICAS DE DIVERSIDAD BAJO NORMA L1
def diversidad_l1(matriz):

    ## Media dimensión por dimensión.
    media = np.mean(matriz, axis=0)

    ## Diversidad de cada dimensión.
    diversidad_por_dimension = np.mean(
        np.abs(matriz - media),
        axis=0
    )

    ## Diversidad total del enjambre.
    return np.mean(diversidad_por_dimension)



def diversidad_posicion(posiciones):
    return diversidad_l1(posiciones)



def diversidad_velocidad(velocidades):
    return diversidad_l1(velocidades)



def diversidad_cognitiva(pbest):
    return diversidad_l1(pbest)


## FUNCIONES AUXILIARES PARA PBEST / GBEST / LBEST
def obtener_indice_mejor(aptitudes, svrs, indices=None):

    if indices is None:
        indices = range(len(aptitudes))

    indices = list(indices)
    mejor = indices[0]

    for indice in indices[1:]:

        if es_mejor(
            aptitudes[indice],
            svrs[indice],
            aptitudes[mejor],
            svrs[mejor]
        ):
            mejor = indice

    return mejor


def obtener_lbest_double_linked(pbest, pbest_aptitud, pbest_svr):

    numero_particulas = len(pbest)
    lbest = np.zeros_like(pbest)

    for i in range(numero_particulas):

        ## Vecindario double-linked: k-1, k, k+1.
        anterior = (i - 1) % numero_particulas
        actual = i
        siguiente = (i + 1) % numero_particulas

        vecinos = [anterior, actual, siguiente]

        indice_mejor = obtener_indice_mejor(
            pbest_aptitud,
            pbest_svr,
            vecinos
        )

        lbest[i] = pbest[indice_mejor].copy()

    return lbest


## PSO GLOBAL CLÁSICO
def pso_global(
    funcion_objetivo,
    funcion_restricciones,
    limites_inferiores,
    limites_superiores,
    semilla=None
):

    generadorRandom = np.random.default_rng(semilla)

    dimensiones = len(limites_inferiores)

    velocidad_maxima = PORCENTAJE_VMAX * (
        limites_superiores - limites_inferiores
    )

    ## 1- CREAR POBLACIÓN ALEATORIA
    posiciones = generadorRandom.uniform(
        limites_inferiores,
        limites_superiores,
        size=(NUM_PARTICULAS, dimensiones)
    )

    ## 2- CREAR VELOCIDADES INICIALES
    velocidades = generadorRandom.uniform(
        -velocidad_maxima,
        velocidad_maxima,
        size=(NUM_PARTICULAS, dimensiones)
    )

    ## 3- EVALUAR SOLUCIONES INICIALES
    aptitudes = np.array([
        funcion_objetivo(posicion)
        for posicion in posiciones
    ])

    svrs = np.array([
        calcular_svr(posicion, funcion_restricciones)
        for posicion in posiciones
    ])

    ## 4- INICIALIZAR PBEST
    pbest = posiciones.copy()
    pbest_aptitud = aptitudes.copy()
    pbest_svr = svrs.copy()

    ## 5- OBTENER GBEST CON REGLAS DE DEB
    indice_mejor = obtener_indice_mejor(
        pbest_aptitud,
        pbest_svr
    )

    gbest = pbest[indice_mejor].copy()
    gbest_aptitud = pbest_aptitud[indice_mejor]
    gbest_svr = pbest_svr[indice_mejor]

    ## HISTORIALES
    historial_aptitud = [gbest_aptitud]
    historial_svr = [gbest_svr]
    historial_dp = [diversidad_posicion(posiciones)]
    historial_dv = [diversidad_velocidad(velocidades)]
    historial_dc = [diversidad_cognitiva(pbest)]

    ## CICLO PRINCIPAL
    for iteracion in range(NUM_ITERACIONES):

        ## El GBEST se mantiene como guía social común para todas
        ## las partículas durante esta iteración.
        guia_global = gbest.copy()

        for i in range(NUM_PARTICULAS):

            ## 6- GENERAR FACTORES ALEATORIOS ENTRE [0,1)
            r1 = generadorRandom.random(dimensiones)
            r2 = generadorRandom.random(dimensiones)

            ## 7- COMPONENTE DE INERCIA
            inercia = W * velocidades[i]

            ## 8- COMPONENTE SOCIAL - GBEST GLOBAL
            social = r1 * C1 * (guia_global - posiciones[i])

            ## 9- COMPONENTE COGNITIVO - PBEST
            cognitivo = r2 * C2 * (pbest[i] - posiciones[i])

            ## 10- ACTUALIZAR VELOCIDAD
            velocidades[i] = inercia + social + cognitivo

            velocidades[i] = np.clip(
                velocidades[i],
                -velocidad_maxima,
                velocidad_maxima
            )

            ## 11- ACTUALIZAR POSICIÓN
            posiciones[i] = posiciones[i] + velocidades[i]

            posiciones[i] = np.clip(
                posiciones[i],
                limites_inferiores,
                limites_superiores
            )

            ## 12- EVALUAR NUEVA POSICIÓN
            aptitud_actual = funcion_objetivo(posiciones[i])
            svr_actual = calcular_svr(
                posiciones[i],
                funcion_restricciones
            )

            ## 13- ACTUALIZAR PBEST CON REGLAS DE DEB
            if es_mejor(
                aptitud_actual,
                svr_actual,
                pbest_aptitud[i],
                pbest_svr[i]
            ):

                pbest[i] = posiciones[i].copy()
                pbest_aptitud[i] = aptitud_actual
                pbest_svr[i] = svr_actual

        ## 14- ACTUALIZAR GBEST CON REGLAS DE DEB
        indice_mejor = obtener_indice_mejor(
            pbest_aptitud,
            pbest_svr
        )

        if es_mejor(
            pbest_aptitud[indice_mejor],
            pbest_svr[indice_mejor],
            gbest_aptitud,
            gbest_svr
        ):

            gbest = pbest[indice_mejor].copy()
            gbest_aptitud = pbest_aptitud[indice_mejor]
            gbest_svr = pbest_svr[indice_mejor]

        ## GUARDAR MÉTRICAS DE LA ITERACIÓN
        historial_aptitud.append(gbest_aptitud)
        historial_svr.append(gbest_svr)
        historial_dp.append(diversidad_posicion(posiciones))
        historial_dv.append(diversidad_velocidad(velocidades))
        historial_dc.append(diversidad_cognitiva(pbest))

    return {
        "mejor_posicion": gbest,
        "mejor_aptitud": gbest_aptitud,
        "mejor_svr": gbest_svr,
        "historial_aptitud": np.array(historial_aptitud),
        "historial_svr": np.array(historial_svr),
        "historial_dp": np.array(historial_dp),
        "historial_dv": np.array(historial_dv),
        "historial_dc": np.array(historial_dc)
    }


## PSO LOCAL - TOPOLOGÍA DOUBLE-LINKED
def pso_local_double_linked(
    funcion_objetivo,
    funcion_restricciones,
    limites_inferiores,
    limites_superiores,
    semilla=None
):

    generadorRandom = np.random.default_rng(semilla)

    dimensiones = len(limites_inferiores)

    velocidad_maxima = PORCENTAJE_VMAX * (
        limites_superiores - limites_inferiores
    )

    ## 1- CREAR POBLACIÓN ALEATORIA
    posiciones = generadorRandom.uniform(
        limites_inferiores,
        limites_superiores,
        size=(NUM_PARTICULAS, dimensiones)
    )

    ## 2- CREAR VELOCIDADES INICIALES
    velocidades = generadorRandom.uniform(
        -velocidad_maxima,
        velocidad_maxima,
        size=(NUM_PARTICULAS, dimensiones)
    )

    ## 3- EVALUAR SOLUCIONES INICIALES
    aptitudes = np.array([
        funcion_objetivo(posicion)
        for posicion in posiciones
    ])

    svrs = np.array([
        calcular_svr(posicion, funcion_restricciones)
        for posicion in posiciones
    ])

    ## 4- INICIALIZAR PBEST
    pbest = posiciones.copy()
    pbest_aptitud = aptitudes.copy()
    pbest_svr = svrs.copy()

    ## Para reportar el mejor resultado de todo el enjambre
    ## seguimos guardando un mejor global histórico, pero NO se usa
    ## como guía social en la ecuación de velocidad.
    indice_mejor = obtener_indice_mejor(
        pbest_aptitud,
        pbest_svr
    )

    mejor_global = pbest[indice_mejor].copy()
    mejor_global_aptitud = pbest_aptitud[indice_mejor]
    mejor_global_svr = pbest_svr[indice_mejor]

    ## HISTORIALES
    historial_aptitud = [mejor_global_aptitud]
    historial_svr = [mejor_global_svr]
    historial_dp = [diversidad_posicion(posiciones)]
    historial_dv = [diversidad_velocidad(velocidades)]
    historial_dc = [diversidad_cognitiva(pbest)]

    ## CICLO PRINCIPAL
    for iteracion in range(NUM_ITERACIONES):

        ## Al inicio de cada iteración se obtiene el mejor vecino
        ## de cada partícula usando k-1, k y k+1.
        lbest = obtener_lbest_double_linked(
            pbest,
            pbest_aptitud,
            pbest_svr
        )

        for i in range(NUM_PARTICULAS):

            ## 6- GENERAR FACTORES ALEATORIOS ENTRE [0,1)
            r1 = generadorRandom.random(dimensiones)
            r2 = generadorRandom.random(dimensiones)

            ## 7- COMPONENTE DE INERCIA
            inercia = W * velocidades[i]

            ## 8- COMPONENTE SOCIAL - LBEST LOCAL
            social = r1 * C1 * (lbest[i] - posiciones[i])

            ## 9- COMPONENTE COGNITIVO - PBEST
            cognitivo = r2 * C2 * (pbest[i] - posiciones[i])

            ## 10- ACTUALIZAR VELOCIDAD
            velocidades[i] = inercia + social + cognitivo

            velocidades[i] = np.clip(
                velocidades[i],
                -velocidad_maxima,
                velocidad_maxima
            )

            ## 11- ACTUALIZAR POSICIÓN
            posiciones[i] = posiciones[i] + velocidades[i]

            posiciones[i] = np.clip(
                posiciones[i],
                limites_inferiores,
                limites_superiores
            )

            ## 12- EVALUAR NUEVA POSICIÓN
            aptitud_actual = funcion_objetivo(posiciones[i])
            svr_actual = calcular_svr(
                posiciones[i],
                funcion_restricciones
            )

            ## 13- ACTUALIZAR PBEST CON REGLAS DE DEB
            if es_mejor(
                aptitud_actual,
                svr_actual,
                pbest_aptitud[i],
                pbest_svr[i]
            ):

                pbest[i] = posiciones[i].copy()
                pbest_aptitud[i] = aptitud_actual
                pbest_svr[i] = svr_actual

        ## 14- OBTENER EL MEJOR RESULTADO GLOBAL SOLO PARA REPORTAR
        indice_mejor = obtener_indice_mejor(
            pbest_aptitud,
            pbest_svr
        )

        if es_mejor(
            pbest_aptitud[indice_mejor],
            pbest_svr[indice_mejor],
            mejor_global_aptitud,
            mejor_global_svr
        ):

            mejor_global = pbest[indice_mejor].copy()
            mejor_global_aptitud = pbest_aptitud[indice_mejor]
            mejor_global_svr = pbest_svr[indice_mejor]

        ## GUARDAR MÉTRICAS DE LA ITERACIÓN
        historial_aptitud.append(mejor_global_aptitud)
        historial_svr.append(mejor_global_svr)
        historial_dp.append(diversidad_posicion(posiciones))
        historial_dv.append(diversidad_velocidad(velocidades))
        historial_dc.append(diversidad_cognitiva(pbest))

    return {
        "mejor_posicion": mejor_global,
        "mejor_aptitud": mejor_global_aptitud,
        "mejor_svr": mejor_global_svr,
        "historial_aptitud": np.array(historial_aptitud),
        "historial_svr": np.array(historial_svr),
        "historial_dp": np.array(historial_dp),
        "historial_dv": np.array(historial_dv),
        "historial_dc": np.array(historial_dc)
    }


## ESTADÍSTICAS
def resumen_estadistico(valores):

    valores = np.array(valores, dtype=float)

    return {
        "mejor": np.min(valores),
        "peor": np.max(valores),
        "promedio": np.mean(valores),
        "mediana": np.median(valores),
        "desviacion_estandar": np.std(valores, ddof=1)
    }


## GRÁFICAS
def graficar_convergencia(
    nombre_problema,
    historiales_global,
    historiales_local
):

    promedio_global = np.mean(historiales_global, axis=0)
    promedio_local = np.mean(historiales_local, axis=0)

    iteraciones = np.arange(len(promedio_global))

    plt.figure(figsize=(9, 5))
    plt.plot(iteraciones, promedio_global, label="PSO Global")
    plt.plot(iteraciones, promedio_local, label="PSO Local Double-Linked")
    plt.xlabel("Número de iteraciones")
    plt.ylabel("Aptitud")
    plt.title(f"Convergencia promedio - {nombre_problema}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CARPETA_GRAFICAS,
            f"convergencia_{nombre_problema}.png"
        ),
        dpi=150
    )
    plt.close()



def graficar_svr(
    nombre_problema,
    historiales_svr_global,
    historiales_svr_local
):

    promedio_global = np.mean(historiales_svr_global, axis=0)
    promedio_local = np.mean(historiales_svr_local, axis=0)

    iteraciones = np.arange(len(promedio_global))

    plt.figure(figsize=(9, 5))
    plt.plot(iteraciones, promedio_global, label="PSO Global")
    plt.plot(iteraciones, promedio_local, label="PSO Local Double-Linked")
    plt.xlabel("Número de iteraciones")
    plt.ylabel("SVR promedio")
    plt.title(f"Violación de restricciones - {nombre_problema}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CARPETA_GRAFICAS,
            f"svr_{nombre_problema}.png"
        ),
        dpi=150
    )
    plt.close()



def graficar_diversidad_mejor_ejecucion(
    nombre_problema,
    nombre_algoritmo,
    resultado
):

    iteraciones = np.arange(len(resultado["historial_dp"]))

    plt.figure(figsize=(9, 5))
    plt.plot(iteraciones, resultado["historial_dp"], label="Dp - Posición")
    plt.plot(iteraciones, resultado["historial_dv"], label="Dv - Velocidad")
    plt.plot(iteraciones, resultado["historial_dc"], label="Dc - Cognitiva")
    plt.xlabel("Número de iteraciones")
    plt.ylabel("Diversidad L1")
    plt.title(
        f"Diversidad de la mejor ejecución - {nombre_algoritmo} - {nombre_problema}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CARPETA_GRAFICAS,
            f"diversidad_{nombre_problema}_{nombre_algoritmo}.png"
        ),
        dpi=150
    )
    plt.close()



def graficar_boxplot(nombre_problema, aptitudes_global, aptitudes_local):

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [aptitudes_global, aptitudes_local],
        tick_labels=["PSO Global", "PSO Local"]
    )
    plt.ylabel("Aptitud final")
    plt.title(f"Resultados finales - {nombre_problema}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CARPETA_GRAFICAS,
            f"boxplot_{nombre_problema}.png"
        ),
        dpi=150
    )
    plt.close()


## EXPERIMENTO DE 30 EJECUCIONES
def ejecutar_experimento(
    nombre_problema,
    funcion_objetivo,
    funcion_restricciones,
    limites_inferiores,
    limites_superiores
):

    resultados_global = []
    resultados_local = []

    for ejecucion in range(1, NUM_EJECUCIONES + 1):

        ## Se usa la misma semilla para ambos algoritmos.
        semilla = ejecucion

        resultado_global = pso_global(
            funcion_objetivo,
            funcion_restricciones,
            limites_inferiores,
            limites_superiores,
            semilla=semilla
        )

        resultado_local = pso_local_double_linked(
            funcion_objetivo,
            funcion_restricciones,
            limites_inferiores,
            limites_superiores,
            semilla=semilla
        )

        resultados_global.append(resultado_global)
        resultados_local.append(resultado_local)

        print(
            f"Ejecución {ejecucion:2d} | "
            f"Global: f={resultado_global['mejor_aptitud']:.6f}, "
            f"SVR={resultado_global['mejor_svr']:.6e} | "
            f"Local: f={resultado_local['mejor_aptitud']:.6f}, "
            f"SVR={resultado_local['mejor_svr']:.6e}"
        )

    aptitudes_global = np.array([
        resultado["mejor_aptitud"]
        for resultado in resultados_global
    ])

    aptitudes_local = np.array([
        resultado["mejor_aptitud"]
        for resultado in resultados_local
    ])

    svrs_global = np.array([
        resultado["mejor_svr"]
        for resultado in resultados_global
    ])

    svrs_local = np.array([
        resultado["mejor_svr"]
        for resultado in resultados_local
    ])

    resumen_global = resumen_estadistico(aptitudes_global)
    resumen_local = resumen_estadistico(aptitudes_local)

    print("\n" + "=" * 80)
    print(f"RESUMEN - {nombre_problema}")
    print("=" * 80)

    print("\nPSO GLOBAL")
    for nombre, valor in resumen_global.items():
        print(f"{nombre}: {valor}")
    print(f"Ejecuciones viables: {np.sum(svrs_global <= TOLERANCIA)}/{NUM_EJECUCIONES}")

    print("\nPSO LOCAL DOUBLE-LINKED")
    for nombre, valor in resumen_local.items():
        print(f"{nombre}: {valor}")
    print(f"Ejecuciones viables: {np.sum(svrs_local <= TOLERANCIA)}/{NUM_EJECUCIONES}")

    ## PRUEBA DE WILCOXON
    ## Se realiza sobre las aptitudes finales pareadas si todas las
    ## ejecuciones finales de ambos algoritmos son viables.
    if np.all(svrs_global <= TOLERANCIA) and np.all(svrs_local <= TOLERANCIA):

        estadistico_w, p_valor = wilcoxon(
            aptitudes_global,
            aptitudes_local
        )

        print("\nPRUEBA DE WILCOXON")
        print(f"W = {estadistico_w}")
        print(f"p-valor = {p_valor}")

        if p_valor < 0.05:
            print("Se rechaza H0: existen diferencias significativas.")
        else:
            print("No se rechaza H0: no se detectan diferencias significativas.")

    else:
        print("\nPRUEBA DE WILCOXON")
        print(
            "No se aplica automáticamente sobre la aptitud porque "
            "existen ejecuciones finales inviables. Revisar los SVR."
        )

    ## HISTORIALES PARA GRÁFICAS PROMEDIO
    historiales_global = np.array([
        resultado["historial_aptitud"]
        for resultado in resultados_global
    ])

    historiales_local = np.array([
        resultado["historial_aptitud"]
        for resultado in resultados_local
    ])

    historiales_svr_global = np.array([
        resultado["historial_svr"]
        for resultado in resultados_global
    ])

    historiales_svr_local = np.array([
        resultado["historial_svr"]
        for resultado in resultados_local
    ])

    graficar_convergencia(
        nombre_problema,
        historiales_global,
        historiales_local
    )

    graficar_svr(
        nombre_problema,
        historiales_svr_global,
        historiales_svr_local
    )

    graficar_boxplot(
        nombre_problema,
        aptitudes_global,
        aptitudes_local
    )

    ## Para estudiar la diversidad se toma la mejor ejecución viable
    ## de cada algoritmo.
    indices_viables_global = np.where(svrs_global <= TOLERANCIA)[0]
    indices_viables_local = np.where(svrs_local <= TOLERANCIA)[0]

    if len(indices_viables_global) > 0:
        indice_mejor_global = indices_viables_global[
            np.argmin(aptitudes_global[indices_viables_global])
        ]

        graficar_diversidad_mejor_ejecucion(
            nombre_problema,
            "Global",
            resultados_global[indice_mejor_global]
        )

        print(
            f"\nMejor ejecución Global para diversidad: "
            f"{indice_mejor_global + 1}"
        )
        print("Posición:")
        print(resultados_global[indice_mejor_global]["mejor_posicion"])

    if len(indices_viables_local) > 0:
        indice_mejor_local = indices_viables_local[
            np.argmin(aptitudes_local[indices_viables_local])
        ]

        graficar_diversidad_mejor_ejecucion(
            nombre_problema,
            "Local",
            resultados_local[indice_mejor_local]
        )

        print(
            f"\nMejor ejecución Local para diversidad: "
            f"{indice_mejor_local + 1}"
        )
        print("Posición:")
        print(resultados_local[indice_mejor_local]["mejor_posicion"])

    return resultados_global, resultados_local


def main():

    print("=" * 80)
    print("PRÁCTICA 3 - PSO GLOBAL VS PSO LOCAL DOUBLE-LINKED")
    print("=" * 80)

    print(f"Partículas: {NUM_PARTICULAS}")
    print(f"Iteraciones: {NUM_ITERACIONES}")
    print(f"Ejecuciones: {NUM_EJECUCIONES}")
    print(f"W: {W}")
    print(f"C1 Social: {C1}")
    print(f"C2 Cognitivo: {C2}")
    print(f"Vmax: {PORCENTAJE_VMAX * 100:.0f}% del rango de cada dimensión")

    print("\n" + "#" * 80)
    print("PROBLEMA 1 - HEAT EXCHANGER DESIGN")
    print("#" * 80)

    ejecutar_experimento(
        "heat_exchanger",
        objetivo_heat_exchanger,
        restricciones_heat_exchanger,
        LIMITES_INFERIORES_HEAT,
        LIMITES_SUPERIORES_HEAT
    )

    print("\n" + "#" * 80)
    print("PROBLEMA 2 - OPTIMAL REACTOR DESIGN")
    print("#" * 80)

    ejecutar_experimento(
        "reactor_design",
        objetivo_reactor,
        restricciones_reactor,
        LIMITES_INFERIORES_REACTOR,
        LIMITES_SUPERIORES_REACTOR
    )


if __name__ == "__main__":
    main()
