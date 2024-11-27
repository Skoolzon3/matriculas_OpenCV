import cv2
import imutils
import os
from os import listdir

# PARÁMETROS VISUALES

# Color utilizado al encontrar coincidencia de matrícula (verde).
colorBien = (0, 255, 0)

# Color utilizado al no encontrar coincidencia de matrícula (rojo).
colorMal = (0, 0, 255)

# Nivel de transparencia (0: completamente transparente, 1: completamente opaco).
transparencia = 0.5 

# Color del texto mostrado (amarillo).
colorTexto = (0, 255, 255)

# Grosor del texto y de las líneas del rectángulo
grosor = 2

# Fuente del texto
fuente = cv2.FONT_HERSHEY_DUPLEX

# Escala del texto
escala = 1

# Ruta del clasificador
ruta_clasificador = './haarcascade_russian_plate_number.xml'


# MATRÍCULAS

# Directorio de matrículas
carpeta_matriculas = './imagenes/matriculas/'

# Listas para almacenar nombres y patrones de matrículas
lista_nombres = []
lista_matr = []
error = False

# Índice que almacena el número de matrículas leídas de carpeta_matriculas
i = 0

# Cargar patrones de matrículas
for archivo in listdir(carpeta_matriculas):
    lista_nombres.append(archivo.replace(".jpg", ""))
    lista_matr.append(cv2.imread(carpeta_matriculas + archivo, 0))  # Cargar en escala de grises
    i+=1

# Si no existen matrículas en la carpeta, no se ejecuta el programa
if (i == 0): 
    print("Error: no existen archivos en la carpeta seleccionada")
    error = True

# Si no se detecta el clasificador mediante su ruta, no se ejecuta el programa
if os.path.exists(ruta_clasificador):
    clasificadorMatriculas = cv2.CascadeClassifier(ruta_clasificador)
    print("Clasificador cargado correctamente.")
else:
    print(f"Error: No se detecta el archivo '{ruta_clasificador}'.")
    error = True

if not error:

    # Cargar la imagen del vehículo
    img = cv2.imread('./imagenes/vehiculos/cocheVWG.jpg')
    img = imutils.resize(img, width=1000)
    img_byn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bandera que notifica si se ha encontrado una coincidencia
    encontrado = False

    # Crear una copia de la imagen original para superponer efectos sin modificarla directamente
    overlay = img.copy()

    # Detectar posibles matrículas en la imagen
    matriculas = clasificadorMatriculas.detectMultiScale(img_byn, minNeighbors=5)

    # Procesar cada matrícula detectada
    for (x, y, ancho, alto) in matriculas:

        # Recortar la región de interés (ROI) detectada por el clasificador
        roi = img_byn[y:y+alto, x:x+ancho]
        
        # Variables para almacenar el mejor valor y el mejor patrón
        mejor_min_val = float('inf')  # Valor inicial muy grande
        mejor_patron_indice = -1      # Inicialmente sin patrón

        # Comparar con cada patrón de matrícula
        for indice in range(len(lista_matr)):
            
            # Redimensionar el patrón al tamaño de la región detectada
            patron = cv2.resize(lista_matr[indice], (ancho, alto))

            # Comparar el patrón con la región de interés
            res = cv2.matchTemplate(roi, patron, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Actualizar mejor_min_val si se encuentra uno mejor (mejor coincidencia)
            if min_val < mejor_min_val:
                mejor_min_val = min_val # Guardar el mejor valor mínimo
                mejor_patron_indice = indice  # Guardar el índice del patrón

        # Si se encontró un patrón con un valor suficientemente bajo (menor que 0.5)
        if mejor_min_val < 0.5:  # Nota: es posible ajustar este umbral para regular el grado de coincidencia de imágenes

            # Obtener el nombre del patrón con mejor coincidencia
            nombre_patron = lista_nombres[mejor_patron_indice]
            print(nombre_patron)

            # Se caulcula el tamaño del texto (título del patrón)
            (texto_ancho, texto_alto), _ = cv2.getTextSize(nombre_patron, fuente, escala, grosor)

            # Cálculo de coordenadas para centrar el texto con respecto al rectángulo
            texto_x = x + (ancho - texto_ancho) // 2
            
            # Dibujar el rectángulo y el texto en la imagen
            cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), colorBien, -1)
            cv2.addWeighted(overlay, transparencia, img, 1 - transparencia, 0, img)
            cv2.rectangle(img, (x, y), (x + ancho, y + alto), colorBien, grosor)
            cv2.putText(img, nombre_patron, (texto_x, y - 10), fuente, escala, colorTexto, grosor)

            # Bandera para notificar coincidencia
            encontrado = True

    # Si no se encuentra coincidencia, se indica en la imagen, indicando la matrícula sobre la que debería existir un patrón
    if not encontrado:
        # Cálculo del tamaño del texto
        (texto_ancho, texto_alto), _ = cv2.getTextSize("Matricula no encontrada", fuente, escala, grosor)

        # Cálculo de las coordenadas para centrar el texto con respecto al rectángulo
        texto_x = x + (ancho - texto_ancho) // 2

        # Dibujar el rectángulo y el texto en la imagen
        cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), colorMal, -1)
        cv2.addWeighted(overlay, transparencia, img, 1 - transparencia, 0, img)
        cv2.rectangle(img, (x, y), (x + ancho, y + alto), colorMal, grosor)
        cv2.putText(img, "Matricula no encontrada", (texto_x, y - 10), fuente, escala, colorTexto, grosor)

    # Mostrar la imagen con la matrícula identificada
    cv2.imshow('Identificacion de matriculas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()