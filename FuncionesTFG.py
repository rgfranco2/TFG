# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:13:18 2022

@author: raque
"""

import cv2
#import matplotlib.pyplot as plt
import numpy as np
#import math
from datetime import datetime as dt
from datetime import date
import pvlib as pv
import pandas as pd
import glob
import scipy.io as mat

factor_escala=0.25


def obtencion_imagenesYfechas(nombre='2022-06-01_18_12'):
    '''
    print('Fecha o nombre imagen a analizar')
    nombre=input()
    '''
    #nombre='2022-05-27'
    dir= 'D:/Software/Anaconda/Proyectos/'
    imagenes_str=glob.glob(dir + nombre+'*.jpg')#selección imágenes

    dates=[]
    dates_fichero=[]
    dates_formato=[]
    for imagen_str in imagenes_str:
        #Obtención de las fechas desde el nombre de la imagen
        fecha = dt.strptime(imagen_str.split('\\')[1].split('.')[0], '%Y-%m-%d_%H_%M_%S')
        #Se convierten las fechas a los dos formatos que serán necesarios
        date_formato=fecha.strftime('%Y/%m/%d %H:%M')
        date_fichero=fecha.strftime('%Y_%m_%d')
        
        dates.append(fecha)
        dates_formato.append(date_formato)
        dates_fichero.append(date_fichero)
    return imagenes_str,dates, dates_formato,dates_fichero


def obtencion_angulos (date, posicion):
    zenith = posicion.loc[date]['zenith']
    azimuth = posicion.loc[date]['azimuth']
    azimuth=azimuth+22.5#Se añade el offset de colocación de la cámara
    return zenith,azimuth #EN GRADOS

def reescalado(imagen, factor_escala=0):
    #reescalar imagen. Recibe imagen en formato foto
    alto = round(imagen.shape[0] * factor_escala)
    ancho = round(imagen.shape[1] * factor_escala)
    tamaño_deseado=(ancho,alto)
    imagen_peq = cv2.resize(imagen, tamaño_deseado) 
    return imagen_peq

def centro_imagen(imagen):
    #imagen=cv2.imread(imagen)
    Y, X, canales = imagen.shape #alto,ancho,canales
    centro = (round(X/2), round(Y/2))
    return centro

def centro_plano(imagen):
    centro = (355, 377)
    return centro    

def world2image(angulo_cenital_deg,azimuth,centro=(1440*factor_escala,1440*factor_escala)):
    
    #Cálculo del radio en función del ángulo introducido
    r=-0.0013*angulo_cenital_deg**3+0.1323*angulo_cenital_deg**2+15.412*angulo_cenital_deg
    r=r*factor_escala
    #Cálculo de la distancia en cada eje desde el centro
    dx=r*np.sin((azimuth+180)*np.pi/180)
    dy=r*np.cos((azimuth+180)*np.pi/180)
    #Cálculo de la distancia en cada eje desde el (0,0)
    px=round(centro[0]+dx)
    py=round(centro[1]+dy)
    return r,px,py
    

def dibujo_angulos(imagen_dibujo,zenith,azimuth):
    #Dibuja ángulos azimuth y zenith. Recibe imagen en formato foto
    #Cálculo del centro de la imagen
    centro=centro_imagen(imagen_dibujo)
    #Cálculo del centroide del Sol
    centroide,sol_cubierto=centroide_sol(imagen_dibujo)
    #Cálculo coordenadas en la imagen de los ángulos
    r,px,py=world2image(zenith, azimuth)
       
    if sol_cubierto==0:#Si detecta el Sol utiliza el centroide
        # Dibujo del Azimuth
        cv2.line(imagen_dibujo, centro, centroide, (255,255,0), 2)
        r=((centroide[0]-centro[0])**2+(centroide[1]-centro[1])**2)**0.5
        #Dibujo del zenith
        cv2.circle(imagen_dibujo,centro,round(r), (255, 0, 0), 2)
    else:#Si no detecta el Sol utiliza la estimación
        # Dibujo del Azimuth
        cv2.line(imagen_dibujo, centro, (px,py), (255,255,0), 2)    
        #Dibujo del zenith
        cv2.circle(imagen_dibujo,centro,round(r), (255, 0, 0), 2)
   
    return imagen_dibujo

def centroide_sol(imagen):
    #Calcula el centroide del Sol. Recibe la imagen en formato foto
    #Convierte la imagen al espacio de color HLS
    imagen_hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS)
    L = imagen_hls[:,:,1]

    mascara_hls = cv2.inRange(L, 240, 255)#Se consideran los valores entre 240 y 255
    mascara_hls_cnt = cv2.medianBlur(mascara_hls.astype(np.uint8),3)
    #Búsqueda de los contornos externos
    contornos, jerarquia = cv2.findContours(mascara_hls_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Se obtiene el contorno del Sol (mayor área)
    area = 0.0
    if len(contornos) > 0:
        for i in contornos:
            #Área del contorno
            area_actual = cv2.contourArea(i)
            if area_actual >= area:
                area = area_actual
                contorno_max = i
            
    # Cálculo del perímetro del sol y de su circularidad (c)
        perimetro = cv2.arcLength(contorno_max, True)
        if perimetro > 0:
            c = 4 * np.pi * area / perimetro**2
        else:
            return 0.0,1
    else: #No se ha detectado el sol
        return 0.0,1
    print('c: ',c)
    #Se asume que si hay una circularidad alta sí que se ha detectado el Sol de forma clara
    #print(c)
    if c>0.4:
        sol_cubierto=0 #El Sol no está cubierto
    else:#Si no, se utiliza nuestra aproximación
        sol_cubierto=1 #El Sol sí está cubierto
        #print('SOL CUBIERTO')
    
    
    # Cálculo del centroide con los momentos del contorno de mayor tamaño
    momentos = cv2.moments(contorno_max)
    #Si el área del contorno (['m00']) es mayor que 0, se calcula el centroide
    if (momentos['m00']>0):# & (moments['m00']>0):
        centroide = (int(momentos['m10']/momentos['m00']), int(momentos['m01']/momentos['m00']))#(cx,cy)
    else: #No se ha detectado el sol
        return 0.0,1
    #centroide[0]=centroide[0]*factor_escala
    #centroide[1]=centroide[1]*factor_escala
    return centroide,sol_cubierto
'''
def almacena_centroide():
    
    return centroide
'''
def quitar_pixeles_saturados(imagen,centro_sol):
    
    #Recibe imagen en formato foto
    ##Lo hace con el centroide pero si no detecta el sol pasarle la aproximación!!!!
    ##centroide,sol_cubierto=centroide_sol(imagen)
    #Máscara para quitar los píxeles saturados alrededor del sol
    mascara_ang=mascara_angulos(imagen,2.5,7,centro_sol)
    
    imagen_mascara_angulos=cv2.bitwise_and(imagen,imagen, mask=mascara_ang)
    
    #Máscara para sacar el valor medio a sustituir en los píxeles quitados
    mascara_calculo=mascara_angulos(imagen,7,8.5,centro_sol)
    imagen_calculos=cv2.bitwise_and(imagen,imagen, mask=mascara_calculo)
    
    #Cálculo media brillo en el exterior de los píxeles saturados
    R_cal = imagen_calculos[:,:,2]
    G_cal = imagen_calculos[:,:,1]
    B_cal = imagen_calculos[:,:,0]  
    I_calculo = np.array(B_cal/3 + G_cal/3 + R_cal/3, np.uint8)
    
    #Cáculo I a sustituir en los pixeles saturados 
    R_mas = imagen_mascara_angulos[:,:,2]
    G_mas = imagen_mascara_angulos[:,:,1]
    B_mas = imagen_mascara_angulos[:,:,0] 
    I_mascara = np.array(B_mas/3 + G_mas/3 + R_mas/3, np.uint8)
    
    I_R=0
    I_B=0
    I_G=0
    filas_sustituir=[]
    columnas_sustituir=[]
    valor_sustituir=[]
    contador=0  
    #Media sin contar los pixeles negros de la imagen para solo contar el área de interés
    for columna in range(len(I_calculo[0])):
        for fila in range(len(I_calculo)):
            if I_calculo[fila,columna]!=0:
                I_R+=R_cal[fila,columna]
                I_B+=B_cal[fila,columna]
                I_G+=G_cal[fila,columna]
                contador+=1
    
    I_R/=contador
    I_B/=contador
    I_G/=contador
    
    #Anoto dónde están los valores a sustituir y cuáles son para sustituirlos en la imagen original
    for columna in range(len(I_mascara[0])):
        for fila in range(len(I_mascara)):
            if I_mascara[fila,columna]!=0:
                filas_sustituir.append(fila)
                columnas_sustituir.append(columna)
                valor_sustituir.append([I_B,I_G,I_R])
                
    #Sustituyo los valores en la imagen original           
    for (fila_s,columna_s,valor_s) in zip(filas_sustituir,columnas_sustituir,valor_sustituir):
        imagen[fila_s,columna_s]=valor_s
    return imagen
    
def mascara_sol(imagen,zenith,azimuth,centro_sol):
    #Recibe imagen en formarto foto
    radio_sol,px,py=world2image(2.5,azimuth)
    cv2.circle(imagen,centro_sol,round(radio_sol), (0, 0, 0), -1)
    return imagen   
'''
def adjust_gamma(image, gamma=1):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
'''
def calculo_centro_nuevo_plano(imagen,angulo_ini=0,angulo_fin=90,inclinacion_zenith=0,inclinacion_azimuth=0):
    
    #TRANSFORMAR DE ANGULOS A COORD Y METER INCLINACION
    #Recibe imagen en formato foto
    centro=centro_plano(imagen)
    #Calcula el nuevo centro en función de la inclinación
    r,px,py=world2image(inclinacion_zenith,inclinacion_azimuth,centro)
    centro_inclinacion=(px,py)
    #Calcula las coordenadas del ángulo inicial y del final
    r_ini,px_ini,py_ini=world2image(angulo_ini,inclinacion_azimuth)
    r_fin,px_fin,py_fin=world2image(angulo_fin,inclinacion_azimuth)
    
    return r_ini,r_fin, centro_inclinacion

def mascara_angulos(imagen,angulo_ini=0,angulo_fin=90,centroide=0,inclinacion_zenith=0,inclinacion_azimuth=0):
    #Recibe imagen en formato foto
    r_ini,r_fin,centro_inclinacion=calculo_centro_nuevo_plano(imagen,angulo_ini,angulo_fin,inclinacion_zenith,inclinacion_azimuth)
    #Si nos dan ya el centroide se usa ese punto (para la máscara solar)
    if centroide!=0:
        centro_inclinacion=centroide
    #Las máscaras deben ser en escala de grises
    imagen_mascara= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    #Constituimos la máscara para que quede una corona blanca en los pixeles a mantener
    cv2.rectangle(imagen_mascara, (0,0), (2880,2880), (0,0,0), thickness=-1)#thickness=-1 ->sólido
    cv2.circle(imagen_mascara, centro_inclinacion, round(r_fin), color=(255,255,255), thickness=-1)
    cv2.circle(imagen_mascara, centro_inclinacion,round( r_ini), color=(0,0,0), thickness=-1)
    return imagen_mascara

def lectura_parametros_imagenes(imagen_str):
    #Lectura parámetros imágenes
    f = open (imagen_str,mode='rb')
    mensaje=f.read(1434) ##lee hasta los parámetros de los sensores
    mensaje=mensaje.decode('unicode_escape')#Lo traduce a ASCII
    
    
    #lectura de los parámetros
    #EXI =float(mensaje.split('EXI=')[1].split('\r')[0])
    #EXA =float(mensaje.split('EXA=')[1].split('\r')[0])
    EXP =float(mensaje.split('EXP=')[1].split('\r')[0])
    #NOI =float(mensaje.split('NOI=')[1].split('\r')[0])
    #AVB =float(mensaje.split('AVB=')[1].split('\r')[0])
    GNG =float(mensaje.split('GNG=')[1].split('\r')[0])
    GNR =float(mensaje.split('GNR=')[1].split('\r')[0])
    GNB =float(mensaje.split('GNB=')[1].split('\r')[0])
    GN2 =float(mensaje.split('GN2=')[1].split('\r')[0])
    #CCG =float(mensaje.split('CCG=')[1].split('\r')[0])
    #CCR =float(mensaje.split('CCR=')[1].split('\r')[0])
    #CCB =float(mensaje.split('CCB=')[1].split('\r')[0])
    #CC2 =float(mensaje.split('CC2=')[1].split('\r')[0])
    #LXR =float(mensaje.split('LXR=')[1].split('\r')[0])
    #Cierre del fichero
    f.close()
    
    return EXP,GNG,GNR,GNB,GN2

def calculo_radiacion(imagen_str,mascara):
    
    EXP,GNG,GNR,GNB,GN2=lectura_parametros_imagenes(imagen_str)
    cte_unidades=1e6
    #sensibilidad=3.8e-5 #Relación ganancia global con los valores medios de cada canal 
    sensibilidad=-7e-5
    
    data=cv2.imread(mascara)
    #Descomposición de la imagen en RGB
    B=data[:,:,0]
    G=data[:,:,1]
    R=data[:,:,2]
    '''
        
    B_gamma=adjust_gamma(B)
    G_gamma=adjust_gamma(G)
    R_gamma=adjust_gamma(R)
    #print(B_gamma)
    '''
#
    B_val=((np.mean(B))/(GNB*EXP))*(1+sensibilidad*GNB)*cte_unidades
    G_val=((np.mean(G))/(GNG*EXP))*(1+sensibilidad*GNG)*cte_unidades
    R_val=((np.mean(R))/(GNR*EXP))*(1+sensibilidad*GNR)*cte_unidades
    '''
    B_gammas.append(np.mean(B))
    R_gammas.append(np.mean(R))
    G_gammas.append(np.mean(G))
    '''
    ##ÑADIR PESOS EN CÁLCULO RADIANCIA    
    RGB_mean=(R_val+G_val+B_val)/3
    return RGB_mean

def lectura_valor_piranometro(date_formato,date_fichero):
    #Apertura del archivo
    f = open (('meteo'+ date_fichero+'.txt'),mode='rb')
        
    mensaje=f.read() ##lee el archivo entero
    mensaje=mensaje.decode('unicode_escape')#Lo traduce a ASCII  
    pir =float(mensaje.split(date_formato)[1].split('\t')[4].split('\t')[0])
    #Cierre del archivo
    f.close()
    return pir
