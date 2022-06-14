# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:17:13 2022

@author: raque
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
#from FuncionesTFG import obtencion_imagenesYfechas,obtencion_angulos,reescalado,dibujo_angulos,centroide_sol, mascara_sol, quitar_pixeles_saturados
from FuncionesTFG import*


#PARÁMETROS CONFIGURACIÓN
longitud = -3.7269817  # grados
latitud = 40.4534752  # grados
Altitud = 695  # metros
factor_escala = 0.25
direccion = 'D:/Software/Anaconda/Proyectos/'
# Imagenes por defecto
nombre = '2022-05-25_14'  # Nombre por defecto de las imágenes a procesar
tiempo_visualizacion = 2000  # ms

# El centroide debe calcularse con las imágenes sin modificar y ser accesible
centroides = []
sol_cubiertos = []
#Ángulos inclinación módulo
angulo_inclinacion = 0  # Ángulo de inclinación en zenith del módulo
angulo_azimuth = 0  # Ángulo de inclinación en azimuth del módulo

#FLAGS
# Control de la existencia o no de máscaras de Sol y sin pixeles saturados
FLAG_MASCARAS_GUARDADAS = 0
# Si además hay máscaras o no de ángulos elegidos por el usuario
FLAG_MASCARAS_USUARIO_GUARDADAS = 0
# Vale 1 en el caso del cálculo de la radiación difusa
FLAG_OBLIGATORIO_GUARDAR_MASCARAS = 0

# Por defecto el usuario elige el cielo completo y el módulo en horizontal
area_normalizada = 1
# Número de píxeles del sensor completo para un factor de escala de 0.25
contador_completo = 437758


def eleccion_imagenes():
    # Si se cambian las imágenes modifica los valores globales
    global nombre, imagenes_str, dates, dates_formato, dates_fichero, times, localizacion, posicion
    print('Fecha o nombre imagen a analizar (EJ 2022-05-25_19_47 (día y hora específicos) o 2022-05 (mes específico')
    nombre = input()
    print(nombre)
    imagenes_str, dates, dates_formato, dates_fichero = obtencion_imagenesYfechas(
        nombre)
    # Hora y posición del Sol actuales
    times = pd.DatetimeIndex(data=dates, tz='Europe/Madrid')
    localizacion = pv.location.Location(
        latitud, longitud, tz='Europe/Madrid', altitude=Altitud)
    posicion = localizacion.get_solarposition(times)
    almacenar_centroide()
    FLAG_MASCARAS_GUARDADAS = 0  # Para estas nuevas imágenes no hay máscaras
    print('Elección de imágenes correcta')


def dibujo_angulos_usuario():
    for imagen_str, date, date_formato, date_fichero in zip(imagenes_str, dates, dates_formato, dates_fichero):
        print('IMAGEN:',imagen_str)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith, azimuth = obtencion_angulos(date, posicion)
        # Reescalado de la foto
        imagen_ang = reescalado(imagen, factor_escala)
        # Dibujo zenith y azimuth
        imagen_ang = dibujo_angulos(imagen_ang, zenith, azimuth)
        # Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_ang)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()
        print('Fin dibujo ángulos')


def aplicar_mascara_sol():
    print(imagenes_str)
    print('¿Quieres guardar las imágenes con la máscara solar?[s/n]')
    eleccion = input()
    for imagen_str, date, centroide, sol_cubierto in zip(imagenes_str, dates, centroides, sol_cubiertos):
        print(imagen_str)
        # Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith, azimuth = obtencion_angulos(date, posicion)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = reescalado(imagen, factor_escala)
        # Uso del centroide, si el Sol no está cubierto, o de la aproximación
        print(centroide, sol_cubierto)
        if sol_cubierto == 0:  # El Sol no está cubierto
            centro_sol = centroide
        else:
            r, px, py = world2image(zenith, azimuth,centro_imagen())
            centro_sol = (px, py)
        # Máscara sol Debe ir después o devolver el centroide porque si no tapado luego cambia
        imagen_mascara = mascara_sol(imagen_peq, zenith, azimuth, centro_sol)
        if eleccion == 's':
            # Guardado de las máscaras
            nombre = 'SolTapado_' + \
                imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre, imagen_mascara)
        # Muestra la foto al usuario
        cv2.imshow('Foto_sol', imagen_mascara)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()
        print('Fin máscaras sol')


def mascara_solYangulos():
    global FLAG_MASCARAS_GUARDADAS
    print(imagenes_str)
    print(FLAG_MASCARAS_GUARDADAS, FLAG_OBLIGATORIO_GUARDAR_MASCARAS)
    if FLAG_OBLIGATORIO_GUARDAR_MASCARAS == 0:
        print('¿Quieres guardar las imágenes?[s/n]')
        eleccion = input()
    else:  # Para el cálculo de la radiación difusa es obligatorio, el usuario no tiene opción
        eleccion = 's'
    for imagen_str, date, centroide, sol_cubierto in zip(imagenes_str, dates, centroides, sol_cubiertos):
        print(imagen_str)
        # Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith, azimuth = obtencion_angulos(date, posicion)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = reescalado(imagen, factor_escala)
        # Uso del centroide, si el Sol no está cubierto, o de la aproximación
        print(centroide, sol_cubierto)
        if sol_cubierto == 0:  # El Sol no está cubierto
            centro_sol = centroide
        else:
            r, px, py = world2image(zenith, azimuth,centro_imagen())
            centro_sol = (px, py)
        # Mascara pixeles saturados
        imagen_pixeles = quitar_pixeles_saturados(imagen_peq, centro_sol)

        # Máscara sol
        imagen_mascara = mascara_sol(
            imagen_pixeles, zenith, azimuth, centro_sol)

        # Quito círculo rojo y letras imagen original
        # ángulo-ini,ángulo-fin,centroide,incl-zenith,incli-azimuth
        mascara_ang1 = mascara_angulos(imagen_mascara, 0, 90, 0, 0, 0)
        imagen_mascara1 = cv2.bitwise_and(
            imagen_mascara, imagen_mascara, mask=mascara_ang1)

        if eleccion == 's':
            # Guardado de las máscaras
            nombre_mascara = 'mascaraYSolTapado_' + \
                imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre_mascara, imagen_mascara)
        # Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_mascara1)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()

    if eleccion == 's':
        FLAG_MASCARAS_GUARDADAS = 1  # Las máscaras se han guardado
    print('Fin máscaras sol y píxeles')


def mascara_eleccion_usuario():
    global FLAG_OBLIGATORIO_GUARDAR_MASCARAS, FLAG_MASCARAS_USUARIO_GUARDADAS, angulo_inclinacion, inclinacion_azimuth
    if FLAG_MASCARAS_GUARDADAS == 0:  # Se deben generar las máscaras sin Sol ni píxeles saturados
        FLAG_OBLIGATORIO_GUARDAR_MASCARAS = 1
        mascara_solYangulos()
    imagenes_mascaras = glob.glob(
        direccion+'mascaraYsolTapado_'+nombre+'*.jpg')  # selección imágenes
    # Pide los valores al usuario
    print('Introduce inclinación en zenith')
    angulo_inclinacion = input()
    print('Introduce inclinación en azimuth')
    angulo_azimuth = input()
    print('Introduce ángulo inicial')
    angulo_ini = input()
    print('Introduce ángulo final')
    angulo_fin = input()
    for imagen_mascara in imagenes_mascaras:
        # Máscara elección ángulos
        # ángulo-ini,ángulo-fin,centroide,incl-zenith,incli-azimuth
        mascara_ang = mascara_angulos(
            imagen_mascara, angulo_ini, angulo_fin, 0, angulo_inclinacion, angulo_azimuth)
        imagen_mascara_angulos = cv2.bitwise_and(
            imagen_mascara, imagen_mascara, mask=mascara_ang)
        cv2.imshow('Mascara_usuario', imagen_mascara_angulos)
        cv2.waitKey(tiempo_visualizacion)
        # CAMBIAR OBLIGATORIEDAD
        # Guardado de las máscaras
        nombre_usuario = 'mascaraUsuario_'+inclinacion_zenith+'zenith' + \
            inclinacion_azimuth+'azimuth_' + \
            imagen_str.split('\\')[1].split('.')[0]+'.jpg'
        cv2.imwrite(nombre_usuario, imagen_mascara_angulos)
    # Aviso de que hay máscaras de usuario guardadas
    FLAG_MASCARAS_USUARIO_GUARDADAS = 1
    # Ya no es obligatorio tener las máscaras generadas
    FLAG_OBLIGATORIO_GUARDAR_MASCARAS = 0


def almacenar_centroide():
    # Modifica las variables centroides y sol_cubiertos globales
    global centroides, sol_cubiertos
    centroides = []  # Pongo el vector a 0
    sol_cubiertos = []
    for imagen_str in imagenes_str:
        # print(imagen_str)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = reescalado(imagen, factor_escala)
        # Obtención del centroide de todas las fotos seleccionadas
        centroide, sol_cubierto = centroide_sol(imagen_peq)
        centroides.append(centroide)
        sol_cubiertos.append(sol_cubierto)
    print(imagenes_str)


def mostrar_imagenes():
    for imagen_str in imagenes_str:
        # print(imagen_str)
        # imagen original en formato foto, no en formato string
        imagen = cv2.imread(imagen_str, 1)
        # Reescalado de la imagen
        imagen_peq = reescalado(imagen, factor_escala)
        cv2.imshow('Foto_original', imagen_peq)
        cv2.waitKey(tiempo_visualizacion)
        cv2.destroyAllWindows()
    print('Fin mostrar imágenes')


def cambiar_tamaño_imagenes():
    global factor_escala
    # Elección del factor de escala
    print('Ayuda al usuario: un factor de 0.25 significa que se muestra la imagen al 25% del tamaño original')
    print('¿Utilizar el factor de reducción por defecto (0.25)?[s/n]')
    eleccion = input()
    if eleccion == 'n':
        print('Introduce nuevo factor')
        factor_nuevo = float(input())
        if factor_nuevo != 0:  # Protección contra el usuario
            factor_escala = factor_nuevo
        else:
            print('El factor no puede valer 0, se configura el factor por defecto (0.25)')
            factor_escala = 0.25
    else:
        factor_escala = 0.25  # Si el usuario lo había cambiado vuelve al de defecto

    print('Éxito en cambio factor escala, valor: ', factor_escala)


def calculo_radiacion_difusa():
    global FLAG_OBLIGATORIO_GUARDAR_MASCARAS
    # No se pregunta al usuario, se guardan las máscaras
    FLAG_OBLIGATORIO_GUARDAR_MASCARAS = 1
    # Se recalculan los contadores por si se ha cambiado el reescalado o los ángulos del usuario
    cambio_contadores()
    # resultado=pd.Series([imagenes])
    EXPs = []
    GNGs = []
    GNRs = []
    GNBs = []
    GN2s = []
    irradiancia_final = []
    pir_difusas = []
    pir_globales = []
    pir_directas = []
    difusa_calculos = []
    difusa_isos = []
    difusa_havdavies = []
    difusa_reindls = []
    difusa_perezs = []

    # Añadir elección de máscaras
    if FLAG_MASCARAS_GUARDADAS == 1 and FLAG_MASCARAS_USUARIO_GUARDADAS == 1:
        # Preguntar al usuario
        print('¿Quieres las máscaras globales o las personalizadas?[g/p]')
        eleccion = input()
        if eleccion == 'p':
            mascaras = glob.glob(direccion+'mascaraUsuario_'+inclinacion_zenith+'zenith' +
                                 inclinacion_azimuth+'azimuth_'+nombre+'*.jpg')  # selección imágenes
        else:  # Si el usuario eleige otra letra cualquiera se utilizarán las globales
            mascaras = glob.glob(direccion+'mascaraYsolTapado_' +
                                 nombre+'*.jpg')  # selección imágenes
    # Si no hay ningunas se generan las globales
    elif FLAG_MASCARAS_GUARDADAS == 0 and FLAG_MASCARAS_USUARIO_GUARDADAS == 0:
        print('Se van a guardar primero las máscaras sin Sol y sin pixeles saturados')
        mascara_solYangulos()
        mascaras = glob.glob(direccion+'mascaraYsolTapado_' +
                             nombre+'*.jpg')  # selección imágenes
    elif FLAG_MASCARAS_GUARDADAS == 1:  # Solo están las globales
        mascaras = glob.glob(direccion+'mascaraYsolTapado_' +
                             nombre+'*.jpg')  # selección imágenes
    else:  # Solo están las personalizadas
        mascaras = glob.glob(direccion+'mascaraUsuario_'+nombre +
                             '*.jpg')  # selección imágenes
    for (imagen_str, mascara, date_fichero, date_formato, date) in zip(imagenes_str, mascaras, dates_fichero, dates_formato, dates):
        # print(imagen)

        # Lectura parámetros imágenes
        EXP, GNG, GNR, GNB, GN2 = lectura_parametros_imagenes(imagen_str)

        EXPs.append(EXP)
        GNGs.append(GNG)
        GNRs.append(GNR)
        GNBs.append(GNB)
        GN2s.append(GN2)

        # Cálculo radiación difusa
        irradiancia = calculo_radiacion(imagen_str, mascara, area_normalizada, contador_completo)  # energia/superficie
        
        irradiancia_final.append(irradiancia)

        difusa_calculo, difusa_iso, difusa_haydavie, difusa_reindl, difusa_perez = calculo_difusa_modelos(
            date_formato, date_fichero, date, posicion, angulo_inclinacion, angulo_azimuth)

        difusa_calculos.append(difusa_calculo)
        difusa_isos.append(difusa_iso)
        difusa_havdavies.append(difusa_havdavie)
        difusa_reindls.append(difusa_reindl)
        difusa_perezs.append(difusa_perez)
        DHI_est_41s.append(DHI_est_41)

        print(date_fichero)
        pir_difusa, pir_global, pir_directa, pir_41 = lectura_valor_piranometro(
            date_formato, date_fichero)

        pir_difusas.append(pir_difusa)
        pir_globales.append(pir_global)
        pir_directas.append(pir_directa)

        print(date, ' RADICION DIFUSA: ', irradiancia)

    cv2.destroyAllWindows()

    datos = {'nombre': dates, 'pir_difusa': pir_difusas, 'pir_global': pir_globales, 'pir_directa': pir_directas,
             'GN2': GN2s, 'EXP': EXPs, 'GNG': GNGs, 'GNR': GNRs, 'GNB': GNBs,
             'irradiancia_final': irradiancia_RGB, 'B_gamma(media)': B_gammas, 'R_gamma(media)': R_gammas, 'G_gamma(media)': G_gammas,
             'radiancia final': irradiancia_final, 'valor_pir_41_grados': valores_pir_41, 'calculo_difusa_0_grados': difusa_calculos,
             'difusa mod_isotropico': difusa_isos, 'difusa mod_havdavies': difusa_havdavies, 'difusa mod_reindl': difusa_reindls,
             'difusa mod_perez': difusa_perezs, 'DHI est 41 grados': DHI_est_41s}

    df = pd.DataFrame(datos)
    print(df)
    df.to_excel('valores_parametros_'+nombre+'_.xlsx')

    # Ya no es obligatorio tener las máscaras generadas
    FLAG_OBLIGATORIO_GUARDAR_MASCARAS = 0
    print('Fin cálculo radiación difusa')


def eleccion_tiempo_visualizacion():
    global tiempo_visualizacion
    print('Introduce tiempo visualización de las imágenes en milisegundos')
    eleccion = input()
    if eleccion != 0:  # Protección contra el usuario
        tiempo_visualizacion = int(eleccion)


def cambio_contadores():
    # El contador de ángulos depende del ángulo elegido por el usuario en la máscara personalizada
    # Ambos contadores dependen del factor de reescalado de la imagen
    # Área completa sensor
    global area_normalizada, contador_completo
    imagen_mascara = cv2.imread("2022-05-25_15_39_25.694.jpg")
    imagen_mascara = reescalado(imagen_mascara, 0.25)
    mascara_ang = mascara_angulos(imagen_mascara, 0, 90, 0, 0, 0)
    contador_completo = 0
    # Media sin contar los pixeles negros
    for columna in range(len(mascara_ang[0])):
        for fila in range(len(mascara_ang)):
            if mascara_ang[fila, columna] != 0:  # Cuento los píxeles blancos de la mascara
                contador_completo += 1
    # print(len(mascara_ang),contador_completo)
    # Área angulos
    imagen_mascara = cv2.imread("2022-05-25_15_39_25.694.jpg")
    imagen_mascara = reescalado(imagen_mascara, 0.25)
    mascara_ang = mascara_angulos(
        imagen_mascara, 0, 90, 0, angulo_inclinacion, angulo_azimuth)
    # mascara_ang=mascara_angulos(imagen_mascara,0,90,0,0,0)
    cv2.imshow('Foto2',  mascara_ang)
    cv2.waitKey(2000)
    # media_m=0
    contador_ang = 0
    # Media sin contar los pixeles negros
    for columna in range(len(mascara_ang[0])):
        for fila in range(len(mascara_ang)):
            if mascara_ang[fila, columna] != 0:  # Cuento los píxeles blancos de la mascara
                # media_m+=mascara_ang[fila,columna]
                contador_ang += 1
    # print(len(mascara_ang),contador_ang)

    area_normalizada = contador_ang/contador_completo


def print_estado_parametros_globales():
    print('PARÁMETROS')
    print('Fechas analizadas analizadas:')
    for imagen_str in imagenes_str:
        print(imagen_str.split('\\')[1].split('.')[0])
    print('Máscaras almacenadas del Sol:', FLAG_MASCARAS_GUARDADAS)
    print('Máscaras almacenadas sin píxeles saturados:',
          FLAG_MASCARAS_USUARIO_GUARDADAS)
    print('Tiempo de visualización actual:', tiempo_visualizacion)
    print('Factor de reescalado actual:', factor_escala)
    print('Ángulo inclinación en zenith del módulo:', angulo_inclinacion)
    print('Ángulo inclinación en azimuth del módulo:', angulo_azimuth)


# Inicio programa con valores por defecto
# Obtención de las imágenes con la fecha o nombre introducido
imagenes_str, dates, dates_formato, dates_fichero = obtencion_imagenesYfechas(
    nombre)
# Hora y posición del Sol actuales
times = pd.DatetimeIndex(data=dates, tz='Europe/Madrid')
localizacion = pv.location.Location(
    latitud, longitud, tz='Europe/Madrid', altitude=Altitud)
posicion = localizacion.get_solarposition(times)
# Obtención del centroide antes del procesamiento de las imágenes
almacenar_centroide()

print('FUNCIONANDO')
eleccion_imagenes()

'''
# Menú visual usuario
root = tk.Tk()
# Tamaño del menú
root.config(width=700, height=1200)
# Nombre del menú
root.title("Menú usuario")
s = ttk.Style()
s.configure(
    "MyButton.TButton",
    foreground="#ff0000",  # Color texto
    background="#000000",  # Color fondo
    padding=20,  # Espacio entre textos y extremos botón
    font=("Times", 14),  # Tipo y tamaño letra
    anchor="w"  # Alineación texto
)
# Colocación de los botones en el menú visual del usuario
boton = ttk.Button(text="Elección imágenes",
                   command=eleccion_imagenes, style="MyButton.TButton")
boton.place(x=50, y=50)
boton2 = ttk.Button(text="Dibujo ángulos",
                    command=dibujo_angulos_usuario, style="MyButton.TButton")
boton2.place(x=50, y=200)
boton2 = ttk.Button(text="Máscara solar",
                    command=aplicar_mascara_sol, style="MyButton.TButton")
boton2.place(x=50, y=350)
boton2 = ttk.Button(text="Visualizar imágenes",
                    command=mostrar_imagenes, style="MyButton.TButton")
boton2.place(x=50, y=500)
boton2 = ttk.Button(text="Estado parámetros",
                    command=print_estado_parametros_globales, style="MyButton.TButton")
boton2.place(x=50, y=650)
boton2 = ttk.Button(text="Cambiar tamaño imágenes",
                    command=cambiar_tamaño_imagenes, style="MyButton.TButton")
boton2.place(x=350, y=50)
boton2 = ttk.Button(text="Cálculo radiación difusa",
                    command=calculo_radiacion_difusa, style="MyButton.TButton")
boton2.place(x=350, y=200)
boton2 = ttk.Button(text="Máscara solar sin píxeles saturados",
                    command=mascara_solYangulos, style="MyButton.TButton")
boton2.place(x=350, y=350)
boton2 = ttk.Button(text="Máscara elección parte cielo",
                    command=mascara_eleccion_usuario, style="MyButton.TButton")
boton2.place(x=350, y=500)
boton2 = ttk.Button(text="Elección tiempo imagen",
                    command=eleccion_tiempo_visualizacion, style="MyButton.TButton")
boton2.place(x=350, y=650)
root.mainloop()
'''
