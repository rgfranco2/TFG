# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:27:39 2022

@author: raque
"""
#Importamos todas las funciones
from FuncionesTFG import*

import cv2
#import matplotlib.pyplot as plt
#import numpy as np
#import math
#from datetime import datetime as dt
from datetime import date
import pvlib as pv
import pandas as pd
#import glob
#import scipy.io as mat

longitud=-3.7269817 #grados
latitud=40.4534752 #grados
Altitud=650 #metros
factor_escala=0.25
dir= 'D:/Software/Anaconda/Proyectos/'


#Obtención de las imágenes con la fecha o nombre introducido
imagenes_str, dates, dates_formato, dates_fichero=obtencion_imagenesYfechas()
#Hora y posición del Sol actuales
times = pd.DatetimeIndex(data=dates, tz='Europe/Madrid')
localizacion=pv.location.Location(latitud, longitud, tz='Europe/Madrid', altitude=Altitud)
posicion=localizacion.get_solarposition(times)

for imagen_str, date, date_formato,date_fichero in zip(imagenes_str,dates,dates_formato,dates_fichero):
    print(imagen_str)
    #Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
    zenith,azimuth=obtencion_angulos(date, posicion)
    imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
    #Reescalado de la imagen
    imagen_peq=reescalado(imagen,factor_escala)  
    #Uso del centroide, si el Sol no está cubierto, o de la aproximación
    centroide,sol_cubierto=centroide_sol(imagen_peq)
    print(centroide, sol_cubierto)
    if sol_cubierto==0:#El Sol no está cubierto
        centro_sol=centroide
    else:
        r,px,py=world2image(zenith, azimuth)
        centro_sol=(px,py)
        
    print(centroide,centro_sol)
    #Dibujo azimuth y zenith
    '''
    imagen_ang=reescalado(imagen,factor_escala)
    imagen_ang=dibujo_angulos(imagen_ang,zenith,azimuth)
    cv2.imshow('Foto_angulos', imagen_ang)
    cv2.waitKey(2000) 
    '''
    cv2.imshow('Foto_mascara_sol', imagen_peq)
    cv2.waitKey(2000)
    
    #Mascara pixeles saturados No hace falta devolver nada
    imagen_pixeles=quitar_pixeles_saturados(imagen_peq,centro_sol)
    
    #Máscara sol Debe ir después o devolver el centroide porque si no tapado luego cambia
    imagen_mascara=mascara_sol(imagen_pixeles,zenith,azimuth,centro_sol)
      
    
    #Máscara elección ángulos
    mascara_ang=mascara_angulos(imagen_mascara,0,90,0,41,180)#ángulo-ini,ángulo-fin,centroide,incl-zenith,incli-azimuth
    imagen_mascara_angulos=cv2.bitwise_and(imagen_mascara,imagen_mascara, mask=mascara_ang)
    
    '''
    #Guardado de las máscaras
    nombre='mascaraYsolTapado_'+imagen_str.split('\\')[1].split('.')[0]+'.jpg'
    cv2.imwrite(nombre,imagen_mascara)
    '''
    #Guardado de las máscaras usuario
    nombre='mascaraUsuario_'+imagen_str.split('\\')[1].split('.')[0]+'.jpg'
    cv2.imwrite(nombre,imagen_mascara_angulos)
    
    
    
    cv2.imshow('Foto_mascara_sol', imagen_mascara)
    cv2.waitKey(2000)
    cv2.imshow('Foto_mascara_sol', mascara_ang)
    cv2.waitKey(2000) 
    cv2.imshow('Foto2', imagen_mascara_angulos)
    cv2.waitKey(2000) 
    
    cv2.destroyAllWindows()
   

