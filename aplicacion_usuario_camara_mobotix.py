# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:17:13 2022

@author: Raquel García Franco.
"""
from datetime import datetime as dt
from datetime import date
import pvlib as pv
import pandas as pd
import cv2
import glob
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import ImageTk, Image
import funciones_internas_camara_mobotix as ft


# PARÁMETROS CONFIGURACIÓN
longitud = -3.7269817  # grados
latitud = 40.4534752  # grados
Altitud = 695  # metros
factor_escala = 0.25
direccion = 'D:/Software/Anaconda/Proyectos/'
# Imagenes por defecto
nombre_imagenes = '2022-05-25_14'  # nombre por defecto de las imágenes a procesar
tiempo_visualizacion = 2000  # ms

# El centroide debe calcularse con las imágenes sin modificar y ser accesible
centroides = []
sol_cubiertos = []
# Ángulos inclinación módulo
angulo_inclinacion = 0  # Ángulo de inclinación en zenith del módulo
angulo_azimuth = 0  # Ángulo de inclinación en azimuth del módulo
# Ángulos a observar por el usuario
angulo_ini = 0
angulo_fin = 90

# FLAGS
# Control de la existencia o no de máscaras de Sol y sin pixeles saturados
FLAG_MASCARAS_GUARDADAS = False
# Si además hay máscaras o no de ángulos elegidos por el usuario
FLAG_MASCARAS_USUARIO_GUARDADAS = False
# Vale 1 en el caso del cálculo de la radiación difusa
FLAG_OBLIGATORIO_GUARDAR_MASCARAS = False
# Activación (True) o desactivación (False) del menú visual
FLAG_MENU_VISUAL = True
# El programa se lanza con unas imágenes por defecto
FLAG_VALORES_POR_DEFECTO = True
# Por defecto se elige el cielo completo y el módulo en horizontal
AREA_NORMALIZADA = 1
# Número de píxeles del sensor completo para un factor de escala de 0.25
CONTADOR_COMPLETO = 437758


def eleccion_imagenes():
    '''
    La función busca todas las imágenes con el nombre o trozo del nombre introducido.
    Por ejemplo, para buscar todas las imágenes del 2022 introducir: 2022
    En cambio, si se quieren procesar todas las imágenes de mayo de este año, introducir: 2022-05
    Si se quiere procesar la imagen del 25 de mayo de 2022 a las 19:47, introducir 2022-05-25_19_47

    Parameters
    -------
    None.

    Returns
    -------
    None.

    '''
    # Si se cambian las imágenes modifica los valores globales
    global nombre_imagenes, imagenes_str, dates, dates_formato, dates_fichero, times, localizacion, posicion
    global FLAG_MASCARAS_GUARDADAS, FLAG_MASCARAS_USUARIO_GUARDADAS
    if FLAG_VALORES_POR_DEFECTO == False:
        print('Fecha o nombre_imagenes imagen a analizar (EJ 2022-05-25_19_47 (día y hora específicos) o 2022-05 (mes específico))')
        nombre_imagenes = input()
        print('Imágenes a analizar', nombre_imagenes)
    imagenes_str, dates, dates_formato, dates_fichero = ft.obtencion_imagenesYfechas(
        nombre_imagenes)
    # Hora y posición del Sol actuales
    times = pd.DatetimeIndex(data=dates, tz='Europe/Madrid')
    localizacion = pv.location.Location(
        latitud, longitud, tz='Europe/Madrid', altitude=Altitud)
    posicion = localizacion.get_solarposition(times)
    almacenar_centroide()

    # Compruebo si el usuario ya tiene esa máscaras guardadas
    mascaras_solYpixeles = glob.glob(
        direccion+'mascaraYsolTapadoSinCirculo_'+nombre_imagenes+'*.jpg')
    mascaras_usuario = glob.glob(direccion+'mascaraUsuario_'+str(angulo_inclinacion)+'zenith' +
                                 str(angulo_azimuth)+'azimuth_'+nombre_imagenes+'*.jpg')
    if len(mascaras_solYpixeles) == 0:
        FLAG_MASCARAS_GUARDADAS = False  # Para estas nuevas imágenes no hay máscaras
    else:
        FLAG_MASCARAS_GUARDADAS = True  # Para estas nuevas imágenes sí hay máscaras
        print('Ya hay máscaras con el Sol tapado y sin píxeles saturados')
    if len(mascaras_usuario) == 0:
        # Para estas nuevas imágenes no hay máscaras
        FLAG_MASCARAS_USUARIO_GUARDADAS = False
    else:
        # Para estas nuevas imágenes sí hay máscaras
        FLAG_MASCARAS_USUARIO_GUARDADAS = True
        print('Ya hay máscaras de usuario guardadas')
    print('Elección de imágenes correcta')


def dibujo_angulos_usuario():
    '''
    La función dibuja los ángulos acimut y cenit en la imagen o imágenes seleccionadas en "elección imágenes".

    Parameters
    -------
    None.

    Returns
    -------
    None.

    '''
    for imagen_str, date, date_formato, date_fichero in zip(imagenes_str, dates, dates_formato, dates_fichero):
        print('IMAGEN:', imagen_str)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith, azimuth = ft.obtencion_angulos(date, posicion)
        # Reescalado de la foto
        imagen_ang = ft.reescalado(imagen, factor_escala)
        # Dibujo zenith y azimuth
        imagen_ang = ft.dibujo_angulos(imagen_ang, zenith, azimuth)
        # Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_ang)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()
        print('Fin dibujo ángulos')


def aplicar_mascara_sol():
    '''
    La función tapa el disco solar (un radio de 2,5 grados) como si de una bola de sombra se tratase.
    Además, da al usuario la opción de guardar las imágenes generadas.

    Parameters
    -------
    None.

    Returns
    -------
    None.

    '''
    print('Imágenes a aplicar máscara sol', imagenes_str)
    print('¿Quieres guardar las imágenes con la máscara solar?[s/n]')
    eleccion = input()
    for imagen_str, date, centroide, sol_cubierto in zip(imagenes_str, dates, centroides, sol_cubiertos):
        print(imagen_str)
        # Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith, azimuth = ft.obtencion_angulos(date, posicion)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = ft.reescalado(imagen, factor_escala)
        # Uso del centroide, si el Sol no está cubierto, o de la aproximación
        if sol_cubierto == 0:  # El Sol no está cubierto
            centro_sol = centroide
        else:
            r, px, py = ft.world2image(zenith, azimuth)
            centro_sol = (px, py)
        # Máscara sol Debe ir después o devolver el centroide porque si no tapado luego cambia
        imagen_mascara = ft.mascara_sol(
            imagen_peq, zenith, azimuth, centro_sol)
        if eleccion == 's':
            # Guardado de las máscaras
            nombre_imagenes = 'SolTapado_' + \
                imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre_imagenes, imagen_mascara)
        # Muestra la foto al usuario
        cv2.imshow('Foto_sol', imagen_mascara)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()
        print('Fin máscaras sol')


def mascara_solYangulos():
    '''
    La función tapa el disco solar (un radio de 2,5 grados) como si de una bola de sombra se tratase y sustituye los 
    píxeles saturados del Sol por el valor medio de los píxeles colindantes.
    Además, da al usuario la opción de guardar las imágenes generadas.

    Parameters
    -------
    None.

    Returns
    -------
    None.

    '''
    global FLAG_MASCARAS_GUARDADAS
    print('Imágenes a aplicar máscara sin Sol ni píxeles saturados', imagenes_str)
    if FLAG_OBLIGATORIO_GUARDAR_MASCARAS == False:
        print('¿Quieres guardar las imágenes?[s/n]')
        eleccion = input()
    else:  # Para el cálculo de la radiación difusa es obligatorio, el usuario no tiene opción
        eleccion = 's'
    for imagen_str, date, centroide, sol_cubierto in zip(imagenes_str, dates, centroides, sol_cubiertos):
        print(imagen_str)
        # Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith, azimuth = ft.obtencion_angulos(date, posicion)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = ft.reescalado(imagen, factor_escala)
        # Uso del centroide, si el Sol no está cubierto, o de la aproximación
        print(centroide, sol_cubierto)
        if sol_cubierto == 0:  # El Sol no está cubierto
            centro_sol = centroide
        else:
            r, px, py = ft.world2image(zenith, azimuth)
            centro_sol = (px, py)
        # Mascara pixeles saturados
        imagen_pixeles = ft.quitar_pixeles_saturados(imagen_peq, centro_sol)

        # Máscara sol
        imagen_mascara = ft.mascara_sol(
            imagen_pixeles, zenith, azimuth, centro_sol)

        # Quito círculo rojo y letras imagen original
        # ángulo-ini,ángulo-fin,centroide,incl-zenith,incli-azimuth
        mascara_ang1 = ft.mascara_angulos(imagen_mascara, 0, 90, 0, 0, 0)
        imagen_mascara1 = cv2.bitwise_and(
            imagen_mascara, imagen_mascara, mask=mascara_ang1)

        if eleccion == 's':
            # Guardado de las máscaras
            nombre_imagenes_mascara = 'mascaraYSolTapadoSinCirculo_' + \
                imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre_imagenes_mascara, imagen_mascara)
        # Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_mascara1)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()

    if eleccion == 's':
        FLAG_MASCARAS_GUARDADAS = True  # Las máscaras se han guardado
    print('Fin máscaras sol y píxeles')


def mascara_eleccion_usuario():
    '''
    Utiliza la máscara con el Sol tapado y sin píxeles saturados. 
    Permite al usuario elegir las inclinaciones en cenit y en azimut del módulo, y también la corona circular 
    (ángulo inicial y final) que se desea visualizar.
    Además, da al usuario la opción de guardar las imágenes generadas.

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''

    global FLAG_OBLIGATORIO_GUARDAR_MASCARAS, FLAG_MASCARAS_USUARIO_GUARDADAS, angulo_inclinacion, angulo_azimuth, angulo_ini, angulo_fin
    if FLAG_MASCARAS_GUARDADAS == False:  # Se deben generar las máscaras sin Sol ni píxeles saturados
        FLAG_OBLIGATORIO_GUARDAR_MASCARAS = True
        mascara_solYangulos()
    imagenes_mascaras = glob.glob(
        direccion+'mascaraYsolTapadoSinCirculo_'+nombre_imagenes+'*.jpg')  # selección imágenes
    # Pide los valores al usuario
    print('Introduce inclinación en zenith (grados)')
    angulo_inclinacion_str = input()
    angulo_inclinacion = float(angulo_inclinacion_str)
    print('Introduce inclinación en azimuth (grados)')
    angulo_azimuth_str = input()
    angulo_azimuth = float(angulo_azimuth_str)
    print('Introduce ángulo inicial (grados)')
    angulo_ini = float(input())
    print('Introduce ángulo final (grados)')
    angulo_fin = float(input())
    print('Parámetros introducidos correctamente')
    # Se comprueba si ya existen máscaras de usuario para los parámetros introducidos
    mascaras = glob.glob(direccion+'mascaraUsuario_'+angulo_inclinacion_str+'zenith' +
                         angulo_azimuth_str+'azimuth_'+nombre_imagenes+'*.jpg')
    if len(mascaras) == 0:  # No hay máscaras
        print('¿Quieres guardar las imágenes?[s/n]')
        eleccion = input()
    else:
        print('Ya hay imágenes guardadas, ¿quieres sobreescribirlas?[s/n]')
        eleccion = input()
    for imagen_mascara, imagen_str in zip(imagenes_mascaras, imagenes_str):
        # Máscara elección ángulos
        # ángulo-ini,ángulo-fin,centroide,incl-zenith,incli-azimuth
        imagen_mascara = cv2.imread(imagen_mascara)
        mascara_ang = ft.mascara_angulos(
            imagen_mascara, angulo_ini, angulo_fin, 0, angulo_inclinacion, angulo_azimuth)
        imagen_mascara_angulos = cv2.bitwise_and(
            imagen_mascara, imagen_mascara, mask=mascara_ang)
        print('Mostrando máscaras')
        cv2.imshow('Foto2',  mascara_ang)
        cv2.waitKey(2000)
        cv2.imshow('Mascara_usuario', imagen_mascara_angulos)
        cv2.waitKey(tiempo_visualizacion)
        if eleccion == 's':
            # Guardado de las máscaras
            nombre_imagenes_usuario = 'mascaraUsuario_'+angulo_inclinacion_str+'zenith' + \
                angulo_azimuth_str+'azimuth_' + \
                imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre_imagenes_usuario, imagen_mascara_angulos)
            print('Guardando las imagenes')
    # Aviso de que hay máscaras de usuario guardadas
    FLAG_MASCARAS_USUARIO_GUARDADAS = True
    # Ya no es obligatorio tener las máscaras generadas
    FLAG_OBLIGATORIO_GUARDAR_MASCARAS = False
    cv2.destroyAllWindows()
    print('Fin elección máscaras usuario')


def almacenar_centroide():
    '''
    Calcula el controide del contorno solar para cada imagen seleccionada.
    También almacena la variable sol_cubierto, que permite saber si el Sol está cubierto, por nubes u otros elementos, o no

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    # Modifica las variables centroides y sol_cubiertos globales
    global centroides, sol_cubiertos
    centroides = []  # Pongo el vector a 0
    sol_cubiertos = []
    for imagen_str in imagenes_str:
        # print(imagen_str)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = ft.reescalado(imagen, factor_escala)
        # Obtención del centroide de todas las fotos seleccionadas
        centroide, sol_cubierto = ft.centroide_sol(imagen_peq)
        centroides.append(centroide)
        sol_cubiertos.append(sol_cubierto)
    print(imagenes_str)


def mostrar_imagenes():
    '''
    Muestra al usuario las imágenes seleccionadas sin ningún procesado.

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    for imagen_str in imagenes_str:
        print('IMAGEN', imagen_str)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = ft.reescalado(imagen, factor_escala)
        cv2.imshow('Foto_original', imagen_peq)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()
    print('Fin mostrar imágenes')


def cambiar_tamaño_imagenes():
    '''
    Permite al usuario elegir el factor de escala deseado para las imágenes originales.

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    global factor_escala
    # Elección del factor de escala
    print('Ayuda al usuario: un factor de 0.25 significa que se muestra la imagen al 25% del tamaño original')
    print('¿Utilizar el factor de reducción por defecto (0.25)?[s/n]')
    eleccion = input()
    if eleccion == 'n':
        print('Introduce nuevo factor')
        factor_nuevo = float(input())
        if factor_nuevo > 0:  # Protección contra el usuario
            factor_escala = factor_nuevo
        else:
            print(
                'El factor no puede ser negativo, se configura el factor por defecto (0.25)')
            factor_escala = 0.25
    else:
        factor_escala = 0.25  # Si el usuario lo había cambiado vuelve al de defecto

    # Cambio del factor de escala para las funciones internas
    ft.establecer_factor_escala(factor_escala)
    print('Fin del cambio del factor de escala, valor: ', factor_escala)


def calculo_radiacion_difusa():
    '''
    Calcula la irradiancia difusa estimada y genera un excel con los parámetros y fechas de las imágenes 
    (tiempo de exposición y ganancias, entre otros), además de con la estimación estimada y el resultado
    de aplicar distintos modelos para el cálculo de la readiación difusa.

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    global FLAG_OBLIGATORIO_GUARDAR_MASCARAS
    # No se pregunta al usuario, se guardan las máscaras
    FLAG_OBLIGATORIO_GUARDAR_MASCARAS = True
    # Se recalculan los contadores por si se ha cambiado el reescalado o los ángulos del usuario
    cambio_contadores()
    # Creación de los vectores donde se almacenarán los resultados
    EXPs = []
    GNGs = []
    GNRs = []
    GNBs = []
    GN2s = []
    irradiancia_final = []
    pir_difusas = []
    pir_globales = []
    pir_directas = []
    pir_globales_41 = []
    DHIs = []
    Ed_POA_isos = []
    difusa_havdavies = []
    Ed_POA_reindls = []
    Ed_POA_perezs = []
    DHI_est_41s = []

    # Añadir elección de máscaras
    if FLAG_MASCARAS_GUARDADAS and FLAG_MASCARAS_USUARIO_GUARDADAS:
        # Preguntar al usuario
        print('¿Quieres las máscaras globales o las personalizadas?[g/p]')
        eleccion = input()
        if eleccion == 'p':
            mascaras = glob.glob(direccion+'mascaraUsuario_'+str(angulo_inclinacion)+'zenith' +
                                 str(angulo_azimuth)+'azimuth_'+nombre_imagenes+'*.jpg')  # selección imágenes
        else:  # Si el usuario eleige otra letra cualquiera se utilizarán las globales
            mascaras = glob.glob(direccion+'mascaraYsolTapadoSinCirculo_' +
                                 nombre_imagenes+'*.jpg')  # selección imágenes
    # Si no hay ningunas se generan las globales
    elif FLAG_MASCARAS_GUARDADAS == False and FLAG_MASCARAS_USUARIO_GUARDADAS == False:
        print('Se van a guardar primero las máscaras sin Sol y sin pixeles saturados')
        mascara_solYangulos()
        mascaras = glob.glob(direccion+'mascaraYsolTapadoSinCirculo_' +
                             nombre_imagenes+'*.jpg')  # selección imágenes
    elif FLAG_MASCARAS_GUARDADAS:  # Solo están las globales
        mascaras = glob.glob(direccion+'mascaraYsolTapadoSinCirculo_' +
                             nombre_imagenes+'*.jpg')  # selección imágenes
    else:  # Solo están las personalizadas
        mascaras = glob.glob(direccion+'mascaraUsuario_'+angulo_inclinacion+'zenith' +
                             angulo_azimuth+'azimuth_'+nombre_imagenes+'*.jpg')  # selección imágenes
    for (imagen_str, mascara, date_fichero, date_formato, date) in zip(imagenes_str, mascaras, dates_fichero, dates_formato, dates):
        # Lectura parámetros imágenes
        EXP, GNG, GNR, GNB, GN2 = ft.lectura_parametros_imagenes(imagen_str)
        # Introducción del valor en los vectores
        EXPs.append(EXP)
        GNGs.append(GNG)
        GNRs.append(GNR)
        GNBs.append(GNB)
        GN2s.append(GN2)

        # Cálculo radiación difusa
        irradiancia = ft.calculo_radiacion(
            imagen_str, mascara, AREA_NORMALIZADA, CONTADOR_COMPLETO)  # energia/superficie
        irradiancia_final.append(irradiancia)
        # Cálculo de los modelos de radiación difusa
        DHI, Ed_POA_iso, Ed_POA_haydavie, Ed_POA_reindl, Ed_POA_perez, DHI_est_41 = ft.calculo_difusa_modelos(
            date_formato, date_fichero, date, posicion, angulo_inclinacion, angulo_azimuth)
        # Introducción del valor en los vectores
        DHIs.append(DHI)
        Ed_POA_isos.append(Ed_POA_iso)
        difusa_havdavies.append(Ed_POA_haydavie)
        Ed_POA_reindls.append(Ed_POA_reindl)
        Ed_POA_perezs.append(Ed_POA_perez)
        DHI_est_41s.append(DHI_est_41)
        # Obtención valores archivo meteo
        pir_difusa, pir_global, pir_directa, pir_41 = ft.lectura_valor_piranometro(
            date_formato, date_fichero)
        # Introducción del valor en los vectores
        pir_difusas.append(pir_difusa)
        pir_globales.append(pir_global)
        pir_directas.append(pir_directa)
        pir_globales_41.append(pir_41)
        # Muestra al usuario la irradiancia difusa calculada
        print(date_formato, 'Irradiancia difusa estimada', irradiancia)

    cv2.destroyAllWindows()

    datos = {'nombre_imagenes': dates, 'pir_difusa': pir_difusas, 'pir_global': pir_globales, 'pir_directa': pir_directas,
             'GN2': GN2s, 'EXP': EXPs, 'GNG': GNGs, 'GNR': GNRs, 'GNB': GNBs,
             'irradiancia final': irradiancia_final, 'valor_pir_41_grados': pir_globales_41, 'calculo_difusa_0_grados': DHIs,
             'difusa mod_isotropico': Ed_POA_isos, 'difusa mod_havdavies': difusa_havdavies, 'difusa mod_reindl': Ed_POA_reindls,
             'difusa mod_perez': Ed_POA_perezs, 'DHI est 41 grados': DHI_est_41s}

    df = pd.DataFrame(datos)
    print(df)
    df.to_excel('valores_parametros_'+nombre_imagenes+'_.xlsx')
    print('Generado archivo:', 'valores_parametros_'+nombre_imagenes+'_.xlsx')
    # Ya no es obligatorio tener las máscaras generadas
    FLAG_OBLIGATORIO_GUARDAR_MASCARAS = False
    print('Fin cálculo radiación difusa')


def eleccion_tiempo_visualizacion():
    '''
    Permite elegir al usuario el tiempo de visualización de las imágenes (en milisegundos).

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    global tiempo_visualizacion
    print('Introduce tiempo visualización de las imágenes en milisegundos')
    eleccion = input()
    if eleccion > 0:  # Protección contra el usuario
        tiempo_visualizacion = int(eleccion)
    else:
        print('No se pueden introducir valores negativos. Se guardará el anterior valor establecido.')
    print('Tiempo de visualización guardado:', tiempo_visualizacion, 'ms')


def cambio_contadores():
    '''
    Calcula el área normalizada, en función de los ángulos elegidos por el usuario, para 
    un correcto cálculo de la irradiancia difusa.

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    # El contador de ángulos depende del ángulo elegido por el usuario en la máscara personalizada
    # Área angulos
    imagen_mascara = cv2.imread("2022-05-25_15_39_25.694.jpg")
    imagen_mascara = ft.reescalado(imagen_mascara, 0.25)
    mascara_ang = ft.mascara_angulos(
        imagen_mascara, angulo_ini, angulo_fin, 0, angulo_inclinacion, angulo_azimuth)
    # Muestra la máscara aplicada para contar los píxeles evaluados
    cv2.imshow('FotoMascara',  mascara_ang)
    cv2.waitKey(2000)
    contador_ang = 0
    # Cuento los píxeles blancos de la mascara
    for columna in range(len(mascara_ang[0])):
        for fila in range(len(mascara_ang)):
            if mascara_ang[fila, columna] != 0:
                contador_ang += 1

    AREA_NORMALIZADA = contador_ang/CONTADOR_COMPLETO


def print_estado_parametros_globales():
    '''
    Imprime por pantalla el valor de los parámetros que el usuario puede modificar, de forma que el 
    usuario pueda conocer su valor actual.

    Parameters
    -------
    None.

    Returns
    -------
    None.
    '''
    print('PARÁMETROS')
    print('Fechas analizadas analizadas:')
    for imagen_str in imagenes_str:
        print(imagen_str.split('\\')[1].split('.')[0])
    print('Máscaras almacenadas del Sol:', FLAG_MASCARAS_GUARDADAS)
    print('Máscaras almacenadas sin píxeles saturados:',
          FLAG_MASCARAS_USUARIO_GUARDADAS)
    print('Tiempo de visualización actual:', tiempo_visualizacion, 'ms')
    print('Factor de reescalado actual:', factor_escala)
    print('Ángulo inclinación en zenith del módulo:',
          angulo_inclinacion, 'grados')
    print('Ángulo inclinación en azimuth del módulo:', angulo_azimuth, 'grados')
    print('--Fin visualización estado parámetros--')


# Inicio programa con valores por defecto
eleccion_imagenes()
# A partir de ahora el usuario elegirá los valores
FLAG_VALORES_POR_DEFECTO = False

print('FUNCIONANDO')

if FLAG_MENU_VISUAL:
    # Menú visual usuario
    root = tk.Tk()
    # Tamaño del menú
    root.config(width=700, height=1200)
    # nombre_imagenes del menú
    root.title("Menú usuario")

    # Elección de las fotos de los botones
    photo_eleccion_imagenes = ImageTk.PhotoImage(
        file=direccion+'eleccion_imagenes.png')
    photo_dibujo_angulos = ImageTk.PhotoImage(
        file=direccion+'dibujo_angulos.png')
    photo_mascara_solar = ImageTk.PhotoImage(
        file=direccion+'mascara_solar.png')
    photo_visualizar_imagenes = ImageTk.PhotoImage(
        file=direccion+'visualizar_imagenes.png')
    photo_estado_parametros = ImageTk.PhotoImage(
        file=direccion+'estado_parametros.png')
    photo_cambiar_tamaño_imagenes = ImageTk.PhotoImage(
        file=direccion+'cambiar_tamaño_imagenes.png')
    photo_calculo_radiacion = ImageTk.PhotoImage(
        file=direccion+'calculo_radiacion.png')
    photo_mascara_solar_sin_pixeles_saturados = ImageTk.PhotoImage(
        file=direccion+'mascara_solar_sin_pixeles_saturados.png')
    photo_mascara_eleccion_usuario = ImageTk.PhotoImage(
        file=direccion+'mascara_eleccion_usuario.png')
    photo_eleccion_tiempo_imagen = ImageTk.PhotoImage(
        file=direccion+'eleccion_tiempo_imagen.png')

    # Colocación de los botones en el menú visual del usuario
    boton = ttk.Button(image=photo_eleccion_imagenes,
                       command=eleccion_imagenes)
    boton.place(x=50, y=50)
    boton2 = ttk.Button(image=photo_dibujo_angulos,
                        command=dibujo_angulos_usuario)
    boton2.place(x=50, y=200)
    boton2 = ttk.Button(image=photo_mascara_solar,
                        command=aplicar_mascara_sol)
    boton2.place(x=50, y=350)
    boton2 = ttk.Button(image=photo_visualizar_imagenes,
                        command=mostrar_imagenes)
    boton2.place(x=50, y=500)
    boton2 = ttk.Button(image=photo_estado_parametros,
                        command=print_estado_parametros_globales)
    boton2.place(x=50, y=650)
    boton2 = ttk.Button(image=photo_cambiar_tamaño_imagenes,
                        command=cambiar_tamaño_imagenes)
    boton2.place(x=350, y=50)
    boton2 = ttk.Button(image=photo_calculo_radiacion,
                        command=calculo_radiacion_difusa)
    boton2.place(x=350, y=200)
    boton2 = ttk.Button(image=photo_mascara_solar_sin_pixeles_saturados,
                        command=mascara_solYangulos)
    boton2.place(x=350, y=350)
    boton2 = ttk.Button(image=photo_mascara_eleccion_usuario,
                        command=mascara_eleccion_usuario)
    boton2.place(x=350, y=500)
    boton2 = ttk.Button(image=photo_eleccion_tiempo_imagen,
                        command=eleccion_tiempo_visualizacion)
    boton2.place(x=350, y=650)

    root.mainloop()
