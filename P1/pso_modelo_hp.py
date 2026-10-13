import numpy as np
import matplotlib.pyplot as plt

## PARÁMETROS GLOBALES
LIMITE_INFERIOR = -5.12
LIMITE_SUPERIOR = 5.12

NUM_PARTICULAS = 100
NUM_ITERACIONES = 100

## PARÁMETROS VARIABLES DE PSO
C_MAX = 1.5
C_MIN = 0.5

W_MAX = 0.7
W_MIN = 0.4

## VELOCIDAD MÁXIMA
VELOCIDAD_MAXIMA = 0.2 * (LIMITE_SUPERIOR - LIMITE_INFERIOR)


## CONVERTIR AMINOÁCIDOS A H Y P
## Utilizamos el código de una letra para cada aminoácido.
## CONVERTIR AMINOÁCIDOS AL MODELO HP
def convertir_a_hp(cadena):
    cadena = cadena.strip().upper()

    aminoacidos_validos = "ACDEFGHIKLMNPQRSTVWY"
    hidrofobicos = "IVLFCMA"

    if len(cadena) < 2:
        return ""

    cadena_hp = ""

    for letra in cadena:
        if letra not in aminoacidos_validos:
            return ""

        if letra in hidrofobicos:
            cadena_hp += "H"
        else:
            cadena_hp += "P"

    return cadena_hp


## Cada valor real de PSO se convierte en uno de 4 movimientos:
## U = arriba, D = abajo, L = izquierda, R = derecha.
def obtener_coordenadas(posicion):
    paso = (LIMITE_SUPERIOR - LIMITE_INFERIOR) / 4
    corte1 = LIMITE_INFERIOR + paso
    corte2 = LIMITE_INFERIOR + 2 * paso
    corte3 = LIMITE_INFERIOR + 3 * paso

    x = 0
    y = 0
    coordenadas = [(x, y)]

    for valor in posicion:
        if valor < corte1:        # U
            y += 1
        elif valor < corte2:      # D
            y -= 1
        elif valor < corte3:      # L
            x -= 1
        else:                    # R
            x += 1

        coordenadas.append((x, y))

    return coordenadas


## FUNCIÓN DE ENERGÍA HP
def energia_hp(posicion, cadena_hp):
    coordenadas = obtener_coordenadas(posicion)
    ## CONTAR CONTACTOS H-H VECINOS QUE NO SEAN CONSECUTIVOS
    contactos_hh = 0
    for i in range(len(cadena_hp)):
        if cadena_hp[i] == "H":
            for j in range(i + 2, len(cadena_hp)):
                if cadena_hp[j] == "H":
                    distancia_x = abs(coordenadas[i][0] - coordenadas[j][0])
                    distancia_y = abs(coordenadas[i][1] - coordenadas[j][1])
                    if distancia_x + distancia_y == 1:
                        contactos_hh += 1

    ## PENALIZAR AMINOÁCIDOS EN LA MISMA CELDA
    solapamientos = 0
    ocupadas = []
    for coordenada in coordenadas:
        if coordenada in ocupadas:
            solapamientos += 1
        else:
            ocupadas.append(coordenada)

    penalizacion = len(cadena_hp) ** 2
    energia = -contactos_hh + penalizacion * solapamientos
    return energia

def pso_variable(dimensiones, cadena_hp, semilla=None):
    generadorRandom = np.random.default_rng(semilla)

    ## 1- CREAR POBLACIÓN Y VELOCIDADES INICIALES
    posiciones = generadorRandom.uniform(LIMITE_INFERIOR, LIMITE_SUPERIOR, size=(NUM_PARTICULAS, dimensiones))
    velocidades = generadorRandom.uniform(-VELOCIDAD_MAXIMA, VELOCIDAD_MAXIMA, size=(NUM_PARTICULAS, dimensiones))

    ## 3- EVALUAR SOLUCIONES INICIALES
    aptitudes = np.array([energia_hp(pos, cadena_hp) for pos in posiciones])

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

            aptitud_actual = energia_hp(posiciones[i], cadena_hp)

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


## MOSTRAR MATRIZ Y GRÁFICA DEL MEJOR PLEGAMIENTO
def mostrar_plegamiento(mejor_posicion, cadena_hp):
    coordenadas = obtener_coordenadas(mejor_posicion)
    xs = []
    ys = []
    celdas = {}
    solapamientos = 0

    for i in range(len(coordenadas)):
        x, y = coordenadas[i]
        xs.append(x)
        ys.append(y)
        if (x, y) in celdas:
            celdas[(x, y)] = "X"
            solapamientos += 1
        else:
            celdas[(x, y)] = cadena_hp[i]

    print("\nMATRIZ DEL MEJOR PLEGAMIENTO (. = vacío, X = solapamiento):")
    for y in range(max(ys), min(ys) - 1, -1):
        fila = ""
        for x in range(min(xs), max(xs) + 1):
            if (x, y) in celdas:
                fila += celdas[(x, y)] + " "
            else:
                fila += ". "
        print(fila)

    print("Solapamientos:", solapamientos)
    if solapamientos == 0:
        print("Conformación válida: Sí")
    else:
        print("Conformación válida: No")

    ## GRÁFICA DEL PLEGAMIENTO
    plt.figure(figsize=(8, 7))
    plt.plot(xs, ys, "-", color="gray", linewidth=1.5, zorder=1)

    for i in range(len(cadena_hp)):
        if cadena_hp[i] == "H":
            color = "tomato"
        else:
            color = "cornflowerblue"
        plt.scatter(xs[i], ys[i], s=220, c=color, edgecolors="black", zorder=2)
        plt.text(xs[i], ys[i], str(i + 1), ha="center", va="center", fontsize=8, zorder=3)

    plt.title("Mejor plegamiento encontrado (Modelo HP)")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.xticks(range(min(xs), max(xs) + 1))
    plt.yticks(range(min(ys), max(ys) + 1))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)


## EXPERIMENTO Y GRÁFICAS
def ejecutar_experimento(cadena_hp):
    NUM_EJECUCIONES = 10
    DIMENSIONES_PRUEBA = [len(cadena_hp) - 1]
    mejor_posicion_global = None
    mejor_aptitud_global = float("inf")

    plt.figure(figsize=(10, 6))

    for dimensiones in DIMENSIONES_PRUEBA:
        resultados_finales = []
        historiales_completos = []

        print("\n" + "=" * 70)
        print(f"EXPERIMENTO con {dimensiones} DIMENSIONES")
        print("=" * 70)

        for ejecucion in range(1, NUM_EJECUCIONES + 1):
            mejor_posicion, mejor_aptitud, historial, parametros_registrados = pso_variable(dimensiones, cadena_hp, ejecucion)

            resultados_finales.append(mejor_aptitud)
            historiales_completos.append(historial)
            print(f"Ejecución {ejecucion}: {mejor_aptitud:.4f}")

            if mejor_aptitud < mejor_aptitud_global:
                mejor_aptitud_global = mejor_aptitud
                mejor_posicion_global = mejor_posicion.copy()

        print("\n--- EVOLUCIÓN DE PARÁMETROS ---")
        print(f"{'Iteración':<10} | {'Inercia (W)':<12} | {'C1 (Social)':<12} | {'C2 (Cognitivo)':<15}")
        print("-" * 55)
        for t, w_val, c1_val, c2_val in parametros_registrados:
            print(f"{t:<10} | {w_val:<12.4f} | {c1_val:<12.4f} | {c2_val:<15.4f}")

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
    plt.ylabel('Energía (Modelo HP)', fontsize=12)
    # Sin escala logarítmica: la energía puede ser negativa o cero.
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    print("\nMejor energía de las 10 ejecuciones:", mejor_aptitud_global)
    mostrar_plegamiento(mejor_posicion_global, cadena_hp)

    print("\nGenerando gráficas de convergencia y plegamiento...")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("PSO CON PARÁMETROS VARIABLES - MODELO HP")
    print("=" * 70)
    print(f"W  = {W_MAX} -> {W_MIN}")
    print(f"C1 (Social) = {C_MIN} -> {C_MAX}")
    print(f"C2 (Cognitivo) = {C_MAX} -> {C_MIN}")
    print(f"Vmax = {VELOCIDAD_MAXIMA:.4f}")

    cadena = input("Introduce la cadena de aminoácidos: ")
    cadena_hp = convertir_a_hp(cadena)

    if cadena_hp == "":
        print("Error: introduce al menos 2 aminoácidos válidos (código de una letra).")
    else:
        print("Secuencia de aminoácidos:", cadena.strip().upper())
        print("Cadena HP:", cadena_hp)
        ejecutar_experimento(cadena_hp)
