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
import tkinter as tk
from tkinter import messagebox, ttk
from FuncionesTFG import obtencion_imagenesYfechas,obtencion_angulos,reescalado,dibujo_angulos,centroide_sol, mascara_sol, quitar_pixeles_saturados

longitud=-3.7269817 #grados
latitud=40.4534752 #grados
Altitud=650 #metros
factor_escala=0.25
dir= 'D:/Software/Anaconda/Proyectos/'
#Imagenes por defecto
nombre='2022-05-25_14'


def eleccion_imagenes():
    #Si se cambian las imágenes modifica los valores globales
    global nombre, imagenes_str,dates,dates_formato,dates_fichero, times,localizacion, posicion
    print('Fecha o nombre imagen a analizar')
    nombre=input()
    ####¿?
    print(nombre)
    imagenes_str, dates, dates_formato, dates_fichero=obtencion_imagenesYfechas(nombre)
    #Hora y posición del Sol actuales
    times = pd.DatetimeIndex(data=dates, tz='Europe/Madrid')
    localizacion=pv.location.Location(latitud, longitud, tz='Europe/Madrid', altitude=Altitud)
    posicion=localizacion.get_solarposition(times)
    almacenar_centroide()
    print(centroides)
    print('Elección de imágenes correcta')
    

def dibujo_angulos_usuario():
    for imagen_str, date, date_formato,date_fichero in zip(imagenes_str,dates,dates_formato,dates_fichero):
        print(imagen_str)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith,azimuth=obtencion_angulos(date, posicion)
        #Reescalado de la foto
        imagen_ang=reescalado(imagen,factor_escala)
        #Dibujo zenith y azimuth
        imagen_ang=dibujo_angulos(imagen_ang,zenith,azimuth)
        #Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_ang)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()
        print('Fin dibujo ángulos')

def aplicar_mascara_sol():
    print(imagenes_str)
    print('¿Quieres guardar las imágenes?[s/n]')
    eleccion=input()
    for imagen_str, date, centroide in zip(imagenes_str,dates,centroides):
        print(imagen_str)
        #Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith,azimuth=obtencion_angulos(date, posicion)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Reescalado de la imagen
        imagen_peq=reescalado(imagen,factor_escala)     
        
        #Máscara sol Debe ir después o devolver el centroide porque si no tapado luego cambia
        imagen_mascara=mascara_sol(imagen_peq,zenith,azimuth,centroide)
        if eleccion=='s':
            #Guardado de las máscaras
            nombre='SolTapado_'+imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre,imagen_mascara)
        #Muestra la foto al usuario
        cv2.imshow('Foto_sol', imagen_mascara)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()
        print('Fin máscaras sol')
def mascara_solYangulos():
    print(imagenes_str)
    print('¿Quieres guardar las imágenes?[s/n]')
    eleccion=input()
    for imagen_str, date, centroide in zip(imagenes_str,dates,centroides):
        print(imagen_str)
        #Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith,azimuth=obtencion_angulos(date, posicion)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Reescalado de la imagen
        imagen_peq=reescalado(imagen,factor_escala)     
        
        #Mascara pixeles saturados No hace falta devolver nada
        imagen_pixeles,centroide=quitar_pixeles_saturados(imagen_peq)
        #Máscara sol Debe ir después o devolver el centroide porque si no tapado luego cambia
        imagen_mascara=mascara_sol(imagen_pixeles,zenith,azimuth,centroide)
        if eleccion=='s':
            #Guardado de las máscaras
            nombre='mascaraYSolTapado_'+imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre,imagen_mascara)
        #Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_mascara)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()
    print('Fin máscaras sol y ángulos')
    
def almacenar_centroide():
    global centroides #Modifica la variable centroides global
    centroides=[]#Pongo el vector a 0
    for imagen_str in imagenes_str:
        #print(imagen_str)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Reescalado de la imagen
        imagen_peq=reescalado(imagen,factor_escala)   
        #Obtención del centroide de todas las fotos seleccionadas
        centroide,sol_cubierto=centroide_sol(imagen_peq)
        centroides.append(centroide)
    #print(centroides)
    print(imagenes_str)

def mostrar_imagenes():
    
    for imagen_str in imagenes_str:
        #print(imagen_str)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Reescalado de la imagen
        imagen_peq=reescalado(imagen,factor_escala)
        cv2.imshow('Foto_original', imagen_peq)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows() 
    print('Fin mostrar imágenes')
        
def cambiar_tamaño_imagenes():
    global factor_escala
    #Elección del factor de escala
    print('Ayuda al usuario: un factor de 0.25 significa que se muestra la imagen al 25% del tamaño original')
    print('¿Utilizar el factor de reducción por defecto (0.25)?[s/n]')
    eleccion=input()
    if eleccion=='n':
        print('Introduce nuevo factor')
        factor_nuevo=float(input())
        if factor_nuevo!=0:#Protección contra el usuario
            factor_escala=factor_nuevo
        else:
            print('El factor no puede valer 0, se configura el factor por defecto (0.25)')
            factor_escala=0.25
    else:
        factor_escala=0.25#Si el usuario lo había cambiado vuelve al de defecto
    print('Éxito en cambio factor escala, valor: ',factor_escala)
    
#Inicio programa con valores por defecto    
#El centroide debe calcularse con las imágenes sin modificar y ser accesible
centroides=[]

#Obtención de las imágenes con la fecha o nombre introducido
imagenes_str, dates, dates_formato, dates_fichero=obtencion_imagenesYfechas(nombre)
#Hora y posición del Sol actuales
times = pd.DatetimeIndex(data=dates, tz='Europe/Madrid')
localizacion=pv.location.Location(latitud, longitud, tz='Europe/Madrid', altitude=Altitud)
posicion=localizacion.get_solarposition(times)
#Obtención del centroide antes del procesamiento de las imágenes
almacenar_centroide()
print('FUNCIONANDO')

#Menú visual usuario
root = tk.Tk()
#Tamaño del menú
root.config(width=700, height=1200)
#Nombre del menú
root.title("Menú usuario")
s = ttk.Style()
s.configure(
    "MyButton.TButton",
    foreground="#ff0000",#Color texto
    background="#000000",#Color fondo
    padding=20, #Espacio entre textos y extremos botón
    font=("Times", 14),#Tipo y tamaño letra
    anchor="w" #Alineación texto
)
boton = ttk.Button(text="Eleccion imágenes", command=eleccion_imagenes, style="MyButton.TButton")
boton.place(x=50, y=150)
boton2=ttk.Button(text="Dibujo ángulos", command=dibujo_angulos_usuario, style="MyButton.TButton")
boton2.place(x=50, y=300)
boton2=ttk.Button(text="Máscara solar", command=aplicar_mascara_sol, style="MyButton.TButton")
boton2.place(x=50, y=600)
boton2=ttk.Button(text="Máscara solar y ángulos", command=mascara_solYangulos, style="MyButton.TButton")
boton2.place(x=50, y=450)
boton2=ttk.Button(text="Visualizar imágenes", command=mostrar_imagenes, style="MyButton.TButton")
boton2.place(x=300, y=150)
boton2=ttk.Button(text="Cambiar tamaño imágenes", command=cambiar_tamaño_imagenes, style="MyButton.TButton")
boton2.place(x=300, y=300)
root.mainloop()