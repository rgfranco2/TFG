# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:13:18 2022

@author: Raquel García Franco.
"""

import cv2
import numpy as np
import math
from datetime import datetime as dt
from datetime import date
import pvlib as pv
import pandas as pd
import glob
import scipy.io as mat

factor_escala = 0.25


def obtencion_imagenesYfechas(nombre_imagen='2022-06-03_19_23'):
    '''
    Devuelve las imágenes encontrazas con el nombre o parte del nombre introducido, así como sus fechas de captura

    Parameters
    -------
    nombre_imagen: string
        Es el nombre o parte del nombre de las imágenes que el usuario quiere seleccionar.

    Returns
    -------
    imagenes_str: list
        Imágenes encontradas en formato string.
    dates: list
        Fechas de las imágenes como objeto datetime.
    dates_formato: list
        Fechas de las imágenes con el formato: '%Y/%m/%d %H:%M'.
    dates_fichero: list
        Fechas de las imágenes con el formato: '%Y_%m_%d'.
    '''

    dir = 'D:/Software/Anaconda/Proyectos/'
    imagenes_str = glob.glob(dir + nombre_imagen+'*.jpg')  # selección imágenes

    dates = []
    dates_fichero = []
    dates_formato = []
    for imagen_str in imagenes_str:
        # Obtención de las fechas desde el nombre de la imagen
        fecha = dt.strptime(imagen_str.split('\\')[1].split('.')[
                            0], '%Y-%m-%d_%H_%M_%S')
        # Se convierten las fechas a los dos formatos que serán necesarios
        date_formato = fecha.strftime('%Y/%m/%d %H:%M')
        date_fichero = fecha.strftime('%Y_%m_%d')
        dates.append(fecha)
        dates_formato.append(date_formato)
        dates_fichero.append(date_fichero)
    return imagenes_str, dates, dates_formato, dates_fichero


def obtencion_angulos(date, posicion, FLAG_CENIT_APARENTE=False):
    '''
    Devuelve los ángulos cenit y acimut en función de la posición solar

    Parameters
    -------
    date: datetime
        Fecha de la imagen que se está procesando.
    posicion: DataFrame
        Posición solar.
    FLAG_CENIT_APARENTE: bool
        Indica si se devuelve eL cenit aparente o no.

    Returns
    -------
    cenit: float
        Valor del cenit en grados.
    acimut: float
        Valor del acimut en grados.
    cenit_aparente: float
        Valor del cenit_aparente en grados.

    '''
    cenit = posicion.loc[date]['zenith']
    acimut = posicion.loc[date]['azimuth']
    cenit_aparente = posicion.loc[date]['apparent_zenith']
    
    if FLAG_CENIT_APARENTE:
        return cenit, acimut, cenit_aparente  # EN GRADOS
    else:
        acimut = acimut+22.5  # Se añade el offset de colocación de la cámara
        return cenit, acimut  # EN GRADOS


def reescalado(imagen, factor_escala=0):
    '''
    Reescala las imágenes

    Parameters
    -------
    imagen: 3D array
        Imagen que se está procesando.
    factor_escala: float
        Valor del factor con el que se quiere reescalar.

    Returns
    -------
    imagen_peq: 3D array
        Imagen reescalada.

    '''
    # reescalar imagen. Recibe imagen en formato foto
    alto = round(imagen.shape[0] * factor_escala)
    ancho = round(imagen.shape[1] * factor_escala)
    tamaño_deseado = (ancho, alto)
    imagen_peq = cv2.resize(imagen, tamaño_deseado)
    return imagen_peq


def centro_imagen(imagen):
    # imagen=cv2.imread(imagen)
    Y, X, canales = imagen.shape  # alto,ancho,canales
    centro = (round(X/2), round(Y/2))

    return centro


def world2image(angulo_cenital_deg, acimut, centro=(1440*factor_escala, 1440*factor_escala)):
    '''
    Calibración angular de la cámara.

    Parameters
    -------
    angulo_cenital_deg: float
        Ángulo cenital en grados.
    acimut: float
        Ángulo acimutal en grados.
    centro: float
        Centro de la imagen.

    Returns
    -------
    r: float
        Radio al que corresponden los ángulos recibidos como parámetros. 
    px: float
        Número de píxeles en el eje x al que corresponden los ángulos recibidos como parámetros.
    py: float
        Número de píxeles en el eje y al que corresponden los ángulos recibidos como parámetros.

    '''

    # Cálculo del radio en función del ángulo introducido
    r = -0.0013*angulo_cenital_deg**3+0.1323 * \
        angulo_cenital_deg**2+15.412*angulo_cenital_deg
    r = r*factor_escala
    # Cálculo de la distancia en cada eje desde el centro
    dx = r*np.sin((acimut+180)*np.pi/180)
    dy = r*np.cos((acimut+180)*np.pi/180)
    # Cálculo de la distancia en cada eje desde el (0,0)
    px = round(centro[0]+dx)
    py = round(centro[1]+dy)
    return r, px, py


def dibujo_angulos(imagen_dibujo, cenit, acimut):
    '''
    Dibujos ángulos acimut y cenit en la imagen.

    Parameters
    -------
    imagen_dibujo: 3D array
        Imagen en la que se van a dibujar los ángulos.
    cenit: float
        Ángulo cenital en grados.
    acimut: float
        Ángulo acimutal en grados.

    Returns
    -------
    imagen_dibujo: 3D array
        Imagen con el dibujo de los ángulos incorporado.
    '''
    # Dibuja ángulos acimut y cenit. Recibe imagen en formato foto
    # Cálculo del centro de la imagen
    centro = centro_imagen(imagen_dibujo)
    # Cálculo del centroide del Sol
    centroide, sol_cubierto = centroide_sol(imagen_dibujo)
    # Cálculo coordenadas en la imagen de los ángulos
    r, px, py = world2image(cenit, acimut)

    if sol_cubierto == 0:  # Si detecta el Sol utiliza el centroide
        # Dibujo del acimut
        cv2.line(imagen_dibujo, centro, centroide, (255, 255, 0), 2)
        r = ((centroide[0]-centro[0])**2+(centroide[1]-centro[1])**2)**0.5
        # Dibujo del cenit
        cv2.circle(imagen_dibujo, centro, round(r), (255, 0, 0), 2)
    else:  # Si no detecta el Sol utiliza la estimación
        # Dibujo del acimut
        cv2.line(imagen_dibujo, centro, (px, py), (255, 255, 0), 2)
        # Dibujo del cenit
        cv2.circle(imagen_dibujo, centro, round(r), (255, 0, 0), 2)

    return imagen_dibujo


def centroide_sol(imagen):
    '''
    Calcula el centroide del contorno solar

    Parameters
    -------
    imagen: 3D array
        Imagen que se está procesando.

    Returns
    -------
    centroide: tuple
        Centroide del contorno solar
    sol_cubierto: int
        Vale 1 si el Sol está cubierto por nubes u otros elementos, y 0 si no.

    '''
    # Calcula el centroide del Sol. Recibe la imagen en formato foto
    # Convierte la imagen al espacio de color HLS
    imagen_hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS)
    L = imagen_hls[:, :, 1]

    # Se consideran los valores entre 240 y 255
    mascara_hls = cv2.inRange(L, 240, 255)
    mascara_hls_cnt = cv2.medianBlur(mascara_hls.astype(np.uint8), 3)
    # Búsqueda de los contornos externos
    contornos, jerarquia = cv2.findContours(
        mascara_hls_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Se obtiene el contorno del Sol (mayor área)
    area = 0.0
    if len(contornos) > 0:
        for i in contornos:
            # Área del contorno
            area_actual = cv2.contourArea(i)
            if area_actual >= area:
                area = area_actual
                contorno_max = i

    # Cálculo del perímetro del sol y de su circularidad (c)
        perimetro = cv2.arcLength(contorno_max, True)
        if perimetro > 0:
            c = 4 * np.pi * area / perimetro**2
        else:
            return 0.0, 1
    else:  # No se ha detectado el sol
        return 0.0, 1
    # Se asume que si hay una circularidad alta sí que se ha detectado el Sol de forma clara
    # print(c)
    if c > 0.4:
        sol_cubierto = 0  # El Sol no está cubierto
    else:  # Si no, se utiliza nuestra aproximación
        sol_cubierto = 1  # El Sol sí está cubierto

    # Cálculo del centroide con los momentos del contorno de mayor tamaño
    momentos = cv2.moments(contorno_max)
    # Si el área del contorno (['m00']) es mayor que 0, se calcula el centroide
    if (momentos['m00'] > 0):  # & (moments['m00']>0):
        centroide = (int(momentos['m10']/momentos['m00']),
                     int(momentos['m01']/momentos['m00']))  # (cx,cy)
    else:  # No se ha detectado el sol
        return 0.0, 1
    return centroide, sol_cubierto


def quitar_pixeles_saturados(imagen, centro_sol):
    '''
    Máscara que elimina los píxeles saturados.

    Parameters
    -------
    imagen: 3D array
        Imagen que se está procesando.
    acentro_sol: tuple
        Centro solar en coordenadas de la imagen.

    Returns
    -------
    imagen: 3D array
        Imagen procesada sin píxeles saturados.

    '''

    # Recibe imagen en formato foto
    # Máscara para quitar los píxeles saturados alrededor del sol
    mascara_ang = mascara_angulos(imagen, 2.5, 7, centro_sol)

    imagen_mascara_angulos = cv2.bitwise_and(imagen, imagen, mask=mascara_ang)

    # Máscara para sacar el valor medio a sustituir en los píxeles quitados
    mascara_calculo = mascara_angulos(imagen, 7, 8.5, centro_sol)
    imagen_calculos = cv2.bitwise_and(imagen, imagen, mask=mascara_calculo)

    # Cálculo media brillo en el exterior de los píxeles saturados
    R_cal = imagen_calculos[:, :, 2]
    G_cal = imagen_calculos[:, :, 1]
    B_cal = imagen_calculos[:, :, 0]
    I_calculo = np.array(B_cal/3 + G_cal/3 + R_cal/3, np.uint8)

    # Cáculo I a sustituir en los pixeles saturados
    R_mas = imagen_mascara_angulos[:, :, 2]
    G_mas = imagen_mascara_angulos[:, :, 1]
    B_mas = imagen_mascara_angulos[:, :, 0]
    I_mascara = np.array(B_mas/3 + G_mas/3 + R_mas/3, np.uint8)

    I_R = 0
    I_B = 0
    I_G = 0
    filas_sustituir = []
    columnas_sustituir = []
    valor_sustituir = []
    contador = 0
    # Media sin contar los pixeles negros de la imagen para solo contar el área de interés
    for columna in range(len(I_calculo[0])):
        for fila in range(len(I_calculo)):
            if I_calculo[fila, columna] != 0:
                I_R += R_cal[fila, columna]
                I_B += B_cal[fila, columna]
                I_G += G_cal[fila, columna]
                contador += 1

    I_R /= contador
    I_B /= contador
    I_G /= contador

    # Anoto dónde están los valores a sustituir y cuáles son para sustituirlos en la imagen original
    for columna in range(len(I_mascara[0])):
        for fila in range(len(I_mascara)):
            if I_mascara[fila, columna] != 0:
                filas_sustituir.append(fila)
                columnas_sustituir.append(columna)
                valor_sustituir.append([I_B, I_G, I_R])

    # Sustituyo los valores en la imagen original
    for (fila_s, columna_s, valor_s) in zip(filas_sustituir, columnas_sustituir, valor_sustituir):
        imagen[fila_s, columna_s] = valor_s
    return imagen


def mascara_sol(imagen, cenit, acimut, centro_sol):
    '''
    Máscara solar, que tapa un radio de 2,5º alrededor del Sol como si de una bola de sombra se tratase.

    Parameters
    -------
    imagen: 3D array
        Imagen que se está procesando.
    cenit: float
        Ángulo cenital en grados.
    acimut: float
        Ángulo acimutal en grados.
    centro_sol: tupla
        Centro solar en coordenadas de la imagen.

    Returns
    -------
    imagen: 3D array 
        Imagen procesada.
    '''
    # Recibe imagen en formarto foto
    radio_sol, px, py = world2image(2.5, acimut)
    cv2.circle(imagen, centro_sol, round(radio_sol), (0, 0, 0), -1)
    return imagen


def calculo_centro_nuevo_plano(imagen, angulo_ini=0, angulo_fin=90, inclinacion_cenit=0, inclinacion_acimut=0):
    '''
    Cálculo el nuevo centro de la imagen en función de los ángulos introducidos por el usuario.

    Parameters
    -------
    imagen: 3D array
        Imagen que se está procesando.
    angulo_ini: float
        Ángulo inicial en grados que se quiere considerar.
    angulo_fin: float
        Ángulo final en grados que se quiere considerar.
    inclinacion_cenit:
        Ángulo de inclinación cenital en grados del módulo.
    inclinacion_acimut:
        Ángulo de inclinación acimutal en grados del módulo.

    Returns
    -------
    r_ini: float
        Radio inicial al que corresponden los ángulos recibidos como parámetros. 
    r_fin: float
        Radio final al que corresponden los ángulos recibidos como parámetros.
    centro_inclinacion: tuple
        centro de la imagen en el nuevo plano formado por los ángulos introducidos.
    '''
    # Recibe imagen en formato foto
    centro = centro_imagen(imagen)
    # Calcula el nuevo centro en función de la inclinación
    r, px, py = world2image(inclinacion_cenit, inclinacion_acimut, centro)
    centro_inclinacion = (px, py)
    # Calcula las coordenadas del ángulo inicial y del final
    r_ini, px_ini, py_ini = world2image(angulo_ini, inclinacion_acimut)
    r_fin, px_fin, py_fin = world2image(angulo_fin, inclinacion_acimut)

    return r_ini, r_fin, centro_inclinacion


def mascara_angulos(imagen, angulo_ini=0, angulo_fin=90, centroide=0, inclinacion_cenit=0, inclinacion_acimut=0):
    '''
    Máscara elegida por el usuario en función de los ángulos introducidos.

    Parameters
    -------
    imagen: 3D array
        Imagen que se está procesando.
    angulo_ini: float
        Ángulo inicial en grados que se quiere considerar.
    angulo_fin: float
        Ángulo final en grados que se quiere considerar.
    centroide: tuple
        Si se introduce un centroide se usa ese punto como centro.
    inclinacion_cenit:
        Ángulo de inclinación cenital en grados del módulo.
    inclinacion_acimut:
        Ángulo de inclinación acimutal en grados del módulo.

    Returns
    -------
    imagen_mascara: 3D array
        Imagen de la máscara realizada.

    '''
    # Recibe imagen en formato foto
    r_ini, r_fin, centro_inclinacion = calculo_centro_nuevo_plano(
        imagen, angulo_ini, angulo_fin, inclinacion_cenit, inclinacion_acimut)
    # Si nos dan ya el centroide se usa ese punto (para la máscara solar)
    if centroide != 0:
        centro_inclinacion = centroide
    # Las máscaras deben ser en escala de grises
    imagen_mascara = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Constituimos la máscara para que quede una corona blanca en los pixeles a mantener
    cv2.rectangle(imagen_mascara, (0, 0), (2880, 2880),
                  (0, 0, 0), thickness=-1)  # thickness=-1 ->sólido
    cv2.circle(imagen_mascara, centro_inclinacion, round(
        r_fin), color=(255, 255, 255), thickness=-1)
    cv2.circle(imagen_mascara, centro_inclinacion,
               round(r_ini), color=(0, 0, 0), thickness=-1)
    return imagen_mascara


def lectura_parametros_imagenes(imagen_str):
    '''
    Realiza la lectura del valor de los parámetros de la imagen.

    Parameters
    -------
    imagen_str: string
        Imagen que se está procesando.

    Returns
    -------
    EXP: int
        Tiempo de exposición.
    GNG: int
        Ganancia del canal de color verde.
    GNR: int
        Ganancia del canal de color rojo.
    GNB: int
        Ganancia del canal de color azul.
    GN2: int
        Ganancia sensor.
    '''
    # Lectura parámetros imágenes
    f = open(imagen_str, mode='rb')
    mensaje = f.read(1434)  # lee hasta los parámetros de los sensores
    mensaje = mensaje.decode('unicode_escape')  # Lo traduce a ASCII

    # lectura de los parámetros
    EXP = float(mensaje.split('EXP=')[1].split('\r')[0])
    GNG = float(mensaje.split('GNG=')[1].split('\r')[0])
    GNR = float(mensaje.split('GNR=')[1].split('\r')[0])
    GNB = float(mensaje.split('GNB=')[1].split('\r')[0])
    GN2 = float(mensaje.split('GN2=')[1].split('\r')[0])
    # Cierre del fichero
    f.close()

    return EXP, GNG, GNR, GNB, GN2


def calculo_radiacion(imagen_str, mascara, area_normalizada, contador_completo):
    '''
    Calcula la radiación difusa.

    Parameters
    -------
    imagen_str: string
        Imagen que se está procesando.
    área_nomalizada: float
        Relación entre los píxeles considerados y los píxeles del sensor completo.
    contador_completo: int
        Número de píxeles del sensor completo.

    Returns
    -------
    rad_fin: float
        Radiación difusa calculada.
    '''

    EXP, GNG, GNR, GNB, GN2 = lectura_parametros_imagenes(imagen_str)
    cte_unidades = 1e6
    sensibilidad = -7e-5

    data = cv2.imread(mascara)
    # Descomposición de la imagen en RGB
    B = data[:, :, 0]
    G = data[:, :, 1]
    R = data[:, :, 2]

    # Ajuste lineal con la ganancia
    B_val = ((media(B, contador_completo))/(GNB*EXP)) * \
        (1+sensibilidad*GNB)*cte_unidades
    G_val = ((media(G, contador_completo))/(GNG*EXP)) * \
        (1+sensibilidad*GNG)*cte_unidades
    R_val = ((media(R, contador_completo))/(GNR*EXP)) * \
        (1+sensibilidad*GNR)*cte_unidades

    RGB_pesos = (R_val*0.514+G_val*0.484+B_val*0.002) / \
        1  # Pesos de cada canal de color
    rad_fin = -1e-5*RGB_pesos**3+0.0063*RGB_pesos**2+0.4367 * \
        RGB_pesos  # Ajuste polinómico de la calibración de intensidad
    # Factor de ajuste al ángulo introducido por el usuario
    print('Antes normalizar',rad_fin)
    rad_fin /=area_normalizada

    return rad_fin


def lectura_valor_piranometro(date_formato, date_fichero):
    '''
    Lectura de los valores de los piranómetros, obtenidos del archivo meteo.

    Parameters
    -------
    imagen_str: string
        Imagen que se está procesando.
    date_formato: str
        Fecha de las imagen con el formato: '%Y/%m/%d %H:%M'.
    date_fichero: str
        Fecha de la imagen con el formato: '%Y_%m_%d'.

    Returns
    -------
    pir_difusa: float
        Valor del piranómetro de difusa.
    pir_global: float
        Valor del piranómetro de global.
    pir_directa: float
        Valor del piranómetro de directa.
    pir_41: float
        Valor del piranómetro de global inclinado 41º.
    '''
    # Apertura del archivo
    f = open(('meteo' + date_fichero+'.txt'), mode='rb')
    mensaje = f.read()  # lee el archivo entero
    mensaje = mensaje.decode('unicode_escape')  # Lo traduce a ASCII
    pir_difusa = float(mensaje.split(date_formato)[1].split(
        '\t')[4].split('\t')[0])  # Valor del piranómetro de difusa
    # Valor del piranómetro inclinado 41º en cenit y 180º en acimut
    pir_41 = float(mensaje.split(date_formato)[
                   1].split('\t')[19].split('\t')[0])
    pir_global = float(mensaje.split(date_formato)[1].split(
        '\t')[3].split('\t')[0])  # Valor piranómetro global
    pir_directa = float(mensaje.split(date_formato)[1].split('\t')[2].split('\t')[
                        0])  # Valor piranómetro directa (pirheliómetro)
    # Cierre del archivo
    f.close()
    return pir_difusa, pir_global, pir_directa, pir_41


def calculo_difusa_modelos(date_formato, date_fichero, date, posicion, angulo_inclinacion, angulo_acimut):
    '''
    Cálculo de la radiación difusa según los modelos.

    Parameters
    -------
    date_formato: str
        Fecha de las imagen con el formato: '%Y/%m/%d %H:%M'.
    date_fichero: str
        Fecha de la imagen con el formato: '%Y_%m_%d'.
    date: datetime
        Fecha de la imagen que se está procesando.
    posicion: DataFrame
        Posición solar.
    angulo_inclinacion:
        Ángulo de inclinación cenital en grados del módulo.
    angulo_acimut:
        Ángulo de inclinación acimutal en grados del módulo.

    Returns
    -------
    DHI: float
        DHI estimada como: DHI=GHI-DNI*COS(cenit)
    Ed_POA_iso: float
        Valor de la irradiancia difusa según el modelo isotrópico.
    Ed_POA_haydavie: float
        Valor de la irradiancia difusa según el modelo de Hay y Davies.
    Ed_POA_reindl: float
        Valor de la irradiancia difusa según el modelo de Reindl.
    Ed_POA_perez: float
        Valor de la irradiancia difusa según el modelo de Pérez.
    D(41): float
        Difusa estimada a 41º como: D(41)=G(41)-DNI*COS(ANGULO INCIDENCIA) 

    '''
    cenit, acimut, cenit_aparente = obtencion_angulos(
        date, posicion, FLAG_CENIT_APARENTE=True)
    pir_difusa, pir_global, pir_directa, pir_41 = lectura_valor_piranometro(
        date_formato, date_fichero)
    dni_extra = pv.irradiance.get_extra_radiation(dt.strptime(
        date_formato, '%Y/%m/%d %H:%M'), solar_constant=1366.1, method='spencer', epoch_year=2022)  # , **kwargs)

    
    DHI = pir_global-pir_directa*math.cos(cenit*np.pi/180)# DHI=GHI-DNI*COS(cenit)
    Ed_POA_iso = pv.irradiance.isotropic(
        angulo_inclinacion, pir_difusa)  # Ed_POA_iso
    Ed_POA_haydavie = pv.irradiance.haydavies(
        angulo_inclinacion, angulo_acimut, pir_difusa, pir_directa, dni_extra, cenit_aparente, acimut, projection_ratio=None)
    Ed_POA_reindl = pv.irradiance.reindl(
        angulo_inclinacion, angulo_acimut, pir_difusa, pir_directa, pir_global, dni_extra, cenit_aparente, acimut)
    airmass = pv.atmosphere.get_relative_airmass(cenit)
    Ed_POA_perez = pv.irradiance.perez(angulo_inclinacion, angulo_acimut, pir_difusa, pir_directa,
                                       dni_extra, cenit_aparente, acimut, airmass, model='allsitescomposite1990', return_components=False)
    
    angulo_incidencia = pv.irradiance.aoi(
        angulo_inclinacion, angulo_acimut, cenit, acimut)
    D_41 = pir_41-pir_directa*math.cos(angulo_incidencia*np.pi/180)# D(41)=G(41)-DNI*COS(ANGULO INCIDENCIA)
    return DHI, Ed_POA_iso, Ed_POA_haydavie, Ed_POA_reindl, Ed_POA_perez, D_41,angulo_incidencia


def establecer_factor_escala(factor_escala_nuevo):
    '''
    Cambio del factor de escala con el que se reescala la imagen.

    Parameters
    -------
    factor_escala_nuevo: float
        Nuevo factor de escala. Debe ser mayor que 0 o se reestablecerá el anterior.

    Returns
    -------
    None.
    '''
    global factor_escala
    factor_escala = factor_escala_nuevo


def media(matriz, contador_completo):
    '''
    Media del valor de los píxeles de la imagen sin contar los píxeles negros; es decir, comtando solo el área de interés.

    Parameters
    -------
    matriz: 3D array
        Matriz de la que se va a realizar la media de los valores.
    contador_completo: int
        Número de píxeles del sensor completo.

    Returns
    -------
    media: float
        Valor medio de los píxeles considerados.
    '''

    media = 0
    # Media sin contar los píxeles negros
    for columna in range(len(matriz[0])):
        for fila in range(len(matriz)):
            media += matriz[fila, columna]

    media /= contador_completo

    return media
