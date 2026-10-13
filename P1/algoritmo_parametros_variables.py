import numpy as np
import matplotlib.pyplot as plt

## PARÁMETROS GLOBALES
LIMITE_INFERIOR = -5.12
LIMITE_SUPERIOR = 5.12

NUM_PARTICULAS = 100
NUM_ITERACIONES = 100

DIMENSIONES_PRUEBA = [10, 20, 30]

## PARÁMETROS VARIABLES DE PSO
C_MAX = 1.5  
C_MIN = 0.5  

W_MAX = 0.7
W_MIN = 0.4

## VELOCIDAD MÁXIMA
VELOCIDAD_MAXIMA = 0.2 * (LIMITE_SUPERIOR - LIMITE_INFERIOR)

## FUNCIÓN DE RASTRIGIN
def rastrigin(x):
    n = len(x)
    return (
        10 * n
        + np.sum(
            x**2
            - 10 * np.cos(2 * np.pi * x)
        )
    )

## ALGORITMO
def pso_variable(dimensiones, semilla=None):
    generadorRandom = np.random.default_rng(semilla)

    ## 1- CREAR POBLACIÓN Y VELOCIDADES INICIALES
    posiciones = generadorRandom.uniform(LIMITE_INFERIOR, LIMITE_SUPERIOR, size=(NUM_PARTICULAS, dimensiones))
    velocidades = generadorRandom.uniform(-VELOCIDAD_MAXIMA, VELOCIDAD_MAXIMA, size=(NUM_PARTICULAS, dimensiones))

    ## 3- EVALUAR SOLUCIONES INICIALES
    aptitudes = np.array([rastrigin(pos) for pos in posiciones])

    ## 4. INICIALIZAR PBEST Y GBEST
    pbest = posiciones.copy()
    pbest_aptitud = aptitudes.copy()

    indice_mejor = np.argmin(pbest_aptitud)
    gbest = pbest[indice_mejor].copy()
    gbest_aptitud = pbest_aptitud[indice_mejor]

    ## Guarda la evolución del mejor resultado
    historial = [gbest_aptitud]
    historial_parametros = []

    ## CICLO PRINCIPAL
    for iteracion in range(NUM_ITERACIONES):
        t = iteracion + 1
        progreso = iteracion / (NUM_ITERACIONES - 1)

        ## ACTUALIZAR PARÁMETROS DE FORMA LINEAL
        # C1 (Social) va de C_MIN a C_MAX
        C1 = C_MIN + (C_MAX - C_MIN) * progreso
        # C2 (Cognitivo) va de C_MAX a C_MIN
        C2 = C_MAX - (C_MAX - C_MIN) * progreso
        # W va de W_MAX a W_MIN
        W = W_MAX - (W_MAX - W_MIN) * progreso

        if t == 1 or t % 10 == 0:
            historial_parametros.append((t, W, C1, C2))

        ## ACTUALIZAR CADA PARTÍCULA
        for i in range(NUM_PARTICULAS):
            r1 = generadorRandom.random(dimensiones)
            r2 = generadorRandom.random(dimensiones)

            inercia = (W * velocidades[i])
            social = (r1 * C1 * (gbest - posiciones[i]))
            cognitivo = (r2 * C2 * (pbest[i] - posiciones[i]))

            velocidades[i] = (inercia + social + cognitivo)
            velocidades[i] = np.clip(velocidades[i], -VELOCIDAD_MAXIMA, VELOCIDAD_MAXIMA)

            posiciones[i] = (posiciones[i] + velocidades[i])
            posiciones[i] = np.clip(posiciones[i], LIMITE_INFERIOR, LIMITE_SUPERIOR)

            aptitud_actual = rastrigin(posiciones[i])

            if aptitud_actual < pbest_aptitud[i]:
                pbest[i] = posiciones[i].copy()
                pbest_aptitud[i] = aptitud_actual

        ## ACTUALIZAR GBEST
        indice_mejor = np.argmin(pbest_aptitud)
        if pbest_aptitud[indice_mejor] < gbest_aptitud:
            gbest = pbest[indice_mejor].copy()
            gbest_aptitud = pbest_aptitud[indice_mejor]

        historial.append(gbest_aptitud)

    return gbest, gbest_aptitud, historial, historial_parametros


## EXPERIMENTO Y GRÁFICAS
def ejecutar_experimento():
    NUM_EJECUCIONES = 10
    
    plt.figure(figsize=(10, 6))

    for dimensiones in DIMENSIONES_PRUEBA:
        resultados_finales = []
        historiales_completos = []

        print("\n" + "=" * 70)
        print(f"EXPERIMENTO con {dimensiones} DIMENSIONES")
        print("=" * 70)

        for ejecucion in range(1, NUM_EJECUCIONES + 1):

            mejor_posicion, mejor_aptitud, historial, parametros_registrados = pso_variable(dimensiones, ejecucion)

            print("\n--- EVOLUCIÓN DE PARÁMETROS ---")
            print(f"{'Iteración':<10} | {'Inercia (W)':<12} | {'C1 (Social)':<12} | {'C2 (Cognitivo)':<15}")
            print("-" * 55)
            for t, w_val, c1_val, c2_val in parametros_registrados:
                print(f"{t:<10} | {w_val:<12.4f} | {c1_val:<12.4f} | {c2_val:<15.4f}")
            
            resultados_finales.append(mejor_aptitud)
            historiales_completos.append(historial)
            
            print(f"Ejecución {ejecucion}: {mejor_aptitud:.4f}")

        resultados_finales = np.array(resultados_finales)
        historial_promedio = np.mean(historiales_completos, axis=0)

        print("\nRESUMEN")
        print(f"Mejor: {np.min(resultados_finales):.4f}")
        print(f"Peor: {np.max(resultados_finales):.4f}")
        print(f"Promedio: {np.mean(resultados_finales):.4f}")
        print(f"Mediana: {np.median(resultados_finales):.4f}")
        print(f"Desviación estándar: {np.std(resultados_finales, ddof=1):.4f}")

        plt.plot(historial_promedio, label=f'{dimensiones} Dimensiones')

    plt.title('Convergencia Promedio de PSO (Parámetros Variables Lineales)', fontsize=14)
    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('Aptitud (Función de Rastrigin)', fontsize=12)
    plt.yscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    print("\nGenerando gráfica de convergencia...")
    plt.show()

if __name__ == "__main__":
    print("=" * 70)
    print("PSO CON PARÁMETROS VARIABLES")
    print("=" * 70)
    print(f"W  = {W_MAX} -> {W_MIN}")
    print(f"C1 (Social) = {C_MIN} -> {C_MAX}")
    print(f"C2 (Cognitivo) = {C_MAX} -> {C_MIN}")
    print(f"Vmax = {VELOCIDAD_MAXIMA:.4f}")

    ejecutar_experimento()