import numpy as np

## PARÁMETROS GLOBALES
LIMITE_INFERIOR = -5.12
LIMITE_SUPERIOR = 5.12

NUM_PARTICULAS = 100
NUM_ITERACIONES = 100

DIMENSIONES_PRUEBA = [10, 20, 30]


## PARÁMETROS VARIABLES DE PSO
C_MAX = 2.5
C_MIN = 1.5

W_MAX = 0.9
W_MIN = 0.4


## VELOCIDAD MÁXIMA
## Se toma el 20 % de la amplitud total del dominio de búsqueda
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


    ## 1- CREAR POBLACIÓN ALEATORIA
    posiciones = generadorRandom.uniform(
        LIMITE_INFERIOR,
        LIMITE_SUPERIOR,
        size=(NUM_PARTICULAS, dimensiones)
    )

    ## 2- CREAR VELOCIDADES INICIALES
    velocidades = generadorRandom.uniform(
        -VELOCIDAD_MAXIMA,
        VELOCIDAD_MAXIMA,
        size=(NUM_PARTICULAS, dimensiones)
    )

    ## 3- EVALUAR SOLUCIONES INICIALES
    lista_aptitudes = []
    for posicion in posiciones:
        resultado = rastrigin(posicion)
        lista_aptitudes.append(resultado)

    aptitudes = np.array(lista_aptitudes)

    ## 4. INICIALIZAR PBEST
    ## Se realiza una copia independiente para que
    ## pbest no cambie automáticamente al modificar posiciones
    pbest = posiciones.copy()
    pbest_aptitud = aptitudes.copy()

    ## 5- OBTENER GBEST
    indice_mejor = np.argmin(pbest_aptitud)

    gbest = pbest[indice_mejor].copy()
    gbest_aptitud = pbest_aptitud[indice_mejor]

    ## Guarda la evolución del mejor resultado
    historial = [gbest_aptitud]


    ## CICLO PRINCIPAL
    for iteracion in range(NUM_ITERACIONES):

        ## ACTUALIZAR PARÁMETROS DE FORMA LINEAL
        t = iteracion + 1

        C1 = C_MIN + ((C_MAX - C_MIN) * t / NUM_ITERACIONES)
        C2 = C_MAX - ((C_MAX - C_MIN) * t / NUM_ITERACIONES)
        W = W_MAX + ((W_MIN - W_MAX) * t / NUM_ITERACIONES)

        ## ACTUALIZAR CADA PARTÍCULA
        for i in range(NUM_PARTICULAS):
            ##Partícula actual

            ## 6- GENERAR FACTORES ALEATORIOS, ENTRE [0,1)
            r1 = generadorRandom.random(dimensiones)
            r2 = generadorRandom.random(dimensiones)

            ## 7- COMPONENTE DE INERCIA
            inercia = (W * velocidades[i])

            ## 8- COMPONENTE SOCIAL
            social = (r1 * C1 * (gbest - posiciones[i]))

            ## 9- COMPONENTE COGNITIVO
            cognitivo = (r2 * C2 * (pbest[i] - posiciones[i]))

            ## 10- ACTUALIZAR VELOCIDAD
            velocidades[i] = (inercia + social + cognitivo)

            ## Limitar velocidad
            velocidades[i] = np.clip(
                velocidades[i],
                -VELOCIDAD_MAXIMA,
                VELOCIDAD_MAXIMA
            )

            ## 11- ACTUALIZAR POSICIÓN
            posiciones[i] = (posiciones[i] + velocidades[i])

            ## Mantener la partícula dentro del rango permitido
            posiciones[i] = np.clip(
                posiciones[i],
                LIMITE_INFERIOR,
                LIMITE_SUPERIOR
            )


            ## 12- EVALUAR NUEVA POSICIÓN
            aptitud_actual = rastrigin(posiciones[i])

            ## 13- ACTUALIZAR PBEST
            if aptitud_actual < pbest_aptitud[i]:
                ## La posición actual se convierte en la nueva mejor posición histórica
                pbest[i] = posiciones[i].copy()
                pbest_aptitud[i] = aptitud_actual

        ## 14- ACTUALIZAR GBEST
        ## Buscar cuál partícula tiene el mejor PBEST
        indice_mejor = np.argmin(pbest_aptitud)

        ## Si ese PBEST supera al mejor resultado global conocido,
        ## se convierte en el nuevo GBEST
        if pbest_aptitud[indice_mejor] < gbest_aptitud:

            gbest = pbest[indice_mejor].copy()
            gbest_aptitud = pbest_aptitud[indice_mejor]

        ## Guardar el mejor resultado global de esta iteración
        historial.append(gbest_aptitud)


    return gbest, gbest_aptitud, historial


## EXPERIMENTO DE 10 EJECUCIONES
def ejecutar_experimento():

    NUM_EJECUCIONES = 10

    for dimensiones in DIMENSIONES_PRUEBA:

        resultados = []

        print("\n" + "=" * 70)
        print(f"EXPERIMENTO con {dimensiones} DIMENSIONES")
        print("=" * 70)

        for ejecucion in range(1, NUM_EJECUCIONES + 1):

            mejor_posicion, mejor_aptitud, historial = pso_variable(
                dimensiones,
                ejecucion
            )

            resultados.append(mejor_aptitud)

            print(f"Ejecución {ejecucion}: {mejor_aptitud}")

        resultados = np.array(resultados)

        print("\nRESUMEN")
        print(f"Mejor: {np.min(resultados)}")
        print(f"Peor: {np.max(resultados)}")
        print(f"Promedio: {np.mean(resultados)}")
        print(f"Mediana: {np.median(resultados)}")
        print(f"Desviación estándar: {np.std(resultados, ddof=1)}")


print("=" * 70)
print("PSO CON PARÁMETROS VARIABLES")
print("=" * 70)

print(f"\nw  = {W_MAX} -> {W_MIN}")
print(f"C1 = {C_MIN} -> {C_MAX} (Social)")
print(f"C2 = {C_MAX} -> {C_MIN} (Cognitivo)")
print(f"Vmax = {VELOCIDAD_MAXIMA}")

ejecutar_experimento()
