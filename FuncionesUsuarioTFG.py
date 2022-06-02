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

longitud=-3.7269817 #grados
latitud=40.4534752 #grados
Altitud=650 #metros
factor_escala=0.25
dir= 'D:/Software/Anaconda/Proyectos/'
#Imagenes por defecto
nombre='2022-05-25_14'
  
#El centroide debe calcularse con las imágenes sin modificar y ser accesible
centroides=[]
sol_cubiertos=[]
#Control de la existencia o no de máscaras de Sol y sin pixeles saturados
flag_mascaras_guardadas=0
flag_mascaras_usuario_guardadas=0#Si además hay máscaras o no de ángulos elegidos por el usuario
flag_obligatorio_guardar_mascaras=0#Vale 1 en el caso del cálculo de la radiación difusa


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
    flag_mascaras_guardadas=0#Para estas nuevas imágenes no hay máscaras
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
    print('¿Quieres guardar las imágenes con la máscara solar?[s/n]')
    eleccion=input()
    for imagen_str, date, centroide in zip(imagenes_str,dates,centroides):
        print(imagen_str)
        #Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith,azimuth=obtencion_angulos(date, posicion)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Reescalado de la imagen
        imagen_peq=reescalado(imagen,factor_escala)     
        #Uso del centroide, si el Sol no está cubierto, o de la aproximación
        print(centroide, sol_cubierto)
        if sol_cubierto==0:#El Sol no está cubierto
            centro_sol=centroide
        else:
            r,px,py=world2image(zenith, azimuth)
            centro_sol=(px,py)
        #Máscara sol Debe ir después o devolver el centroide porque si no tapado luego cambia
        imagen_mascara=mascara_sol(imagen_peq,zenith,azimuth,centro_sol)
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
    global flag_mascaras_guardadas
    print(imagenes_str)
    print(flag_mascaras_guardadas,flag_obligatorio_guardar_mascaras)
    if flag_obligatorio_guardar_mascaras==0:
        print('¿Quieres guardar las imágenes?[s/n]')
        eleccion=input()
    else:#Para el cálculo de la radiación difusa es obligatorio, el usuario no tiene opción
        eleccion='s'
    for imagen_str, date, centroide, sol_cubierto in zip(imagenes_str,dates,centroides,sol_cubiertos):
        print(imagen_str)
        #Obtención de los ángulos zenith y azimuth para la fecha y posición actuales
        zenith,azimuth=obtencion_angulos(date, posicion)
        imagen=cv2.imread(imagen_str,1)#imagen original en formato foto, no en formato string
        #Reescalado de la imagen
        imagen_peq=reescalado(imagen,factor_escala)
        #Uso del centroide, si el Sol no está cubierto, o de la aproximación
        print(centroide, sol_cubierto)
        if sol_cubierto==0:#El Sol no está cubierto
            centro_sol=centroide
        else:
            r,px,py=world2image(zenith, azimuth)
            centro_sol=(px,py)
        #Mascara pixeles saturados 
        imagen_pixeles=quitar_pixeles_saturados(imagen_peq,centro_sol)
        #Máscara sol
        imagen_mascara=mascara_sol(imagen_pixeles,zenith,azimuth,centro_sol)
        if eleccion=='s':
            #Guardado de las máscaras
            nombre='mascaraYSolTapado_'+imagen_str.split('\\')[1].split('.')[0]+'.jpg'
            cv2.imwrite(nombre,imagen_mascara)
        #Muestra la foto al usuario
        cv2.imshow('Foto_angulos', imagen_mascara)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()
    
    if eleccion=='s':
        flag_mascaras_guardadas=1#Las máscaras se han guardado
    print('Fin máscaras sol y píxeles')
    
def mascara_eleccion_usuario():
    global flag_obligatorio_guardar_mascaras,flag_mascaras_usuario_guardadas
    if flag_mascaras_guardadas==0:#Se deben generar las máscaras sin Sol ni píxeles saturados
        flag_obligatorio_guardar_mascaras=1
        mascara_solYangulos()
    imagenes_mascaras=glob.glob(dir+'mascaraYsolTapado_'+nombre+'*.jpg')#selección imágenes
    #Pide los valores al usuario
    print('Introduce inclinación en zenith')
    inclinacion_zenith=input()
    print('Introduce inclinación en azimuth')
    inclinacion_azimuth=input()
    print('Introduce ángulo inicial')
    angulo_ini=input()
    print('Introduce ángulo final')
    angulo_fin=input()
    for imagen_mascara in imagenes_mascaras:
        #Máscara elección ángulos
        mascara_ang=mascara_angulos(imagen_mascara,angulo_ini,angulo_fin,0,inclinacion_zenith,inclinacion_azimuth)#ángulo-ini,ángulo-fin,centroide,incl-zenith,incli-azimuth
        imagen_mascara_angulos=cv2.bitwise_and(imagen_mascara,imagen_mascara, mask=mascara_ang)
        cv2.imshow('Foto2', imagen_mascara_angulos)
        cv2.waitKey(2000)
        ##CAMBIAR OBLIGATORIEDAD
        #Guardado de las máscaras
        nombre='mascaraUsuario_'+imagen_str.split('\\')[1].split('.')[0]+'.jpg'
        cv2.imwrite(nombre,imagen_mascara_angulos)
    flag_mascaras_usuario_guardadas=1
    flag_obligatorio_guardar_mascaras=0#Ya no es obligatorio tener las máscaras generadas
        
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
        sol_cubiertos.append(sol_cubierto)
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
    
def calculo_radiacion_difusa():
    global flag_obligatorio_guardar_mascaras
    flag_obligatorio_guardar_mascaras=1#No se pregunta al usuario, se guardan las máscaras
    
    #resultado=pd.Series([imagenes])
    '''
    EXIs=[]
    EXAs=[]
    '''
    EXPs=[]
    '''
    NOIs =[]
    AVBs=[]
    '''
    GNGs =[]
    GNRs =[]
    GNBs =[]
    GN2s =[]
    '''
    CCGs =[]
    CCBs =[]
    CCRs =[]
    CC2s =[]
    LXRs =[]
    valores=[]
    valores_G=[]
    valores_R=[]
    valores_B=[]
    valores_RGB=[]
    valores_pir=[]
    valores_G_pir=[]
    valores_R_pir=[]
    valores_B_pir=[]
    valores_RGB_pir=[]
    '''
    radiancia_RGB=[]

    #valores_comp=[]
    valores_piranometro=[]
    #v_ts=[]
    '''
    B_gammas=[]
    R_gammas=[]
    G_gammas=[]
    '''
    #Añadir elección de máscaras
    if flag_mascaras_guardadas==1 and flag_mascaras_usuario_guardadas==0:
        ##Preguntar al usuario
        print('¿Quieres las máscaras globales o las personalizadas')####TERMINAR
    elif flag_mascaras_guardadas==0 and flag_mascaras_usuario_guardadas==0:
        print('Se van a guardar primero las máscaras sin Sol y sin pixeles saturados')
        mascara_solYangulos()
        mascaras=glob.glob(dir+'mascaraYsolTapado_'+nombre+'*.jpg')#selección imágenes
    elif flag_mascaras_guardadas==1:
        mascaras=glob.glob(dir+'mascaraYsolTapado_'+nombre+'*.jpg')#selección imágenes
    else:
        mascaras=glob.glob(dir+'mascaraUsuario_'+nombre+'*.jpg')#selección imágenes
    for (imagen_str,mascara,date_fichero,date_formato) in zip(imagenes_str,mascaras,dates_fichero,dates_formato):
        #print(imagen)
        
        #Lectura parámetros imágenes
        EXP,GNG,GNR,GNB,GN2=lectura_parametros_imagenes(imagen_str)
        
        #EXIs.append(EXI)
        #EXAs.append(EXA)
        EXPs.append(EXP)
        #NOIs.append(NOI)
        #AVBs.append(AVB)
        GNGs.append(GNG)
        GNRs.append(GNR)
        GNBs.append(GNB)
        GN2s.append(GN2)
        #CCGs.append(CCG)
        #CCRs.append(CCR)
        #CCBs.append(CCB)
        #CC2s.append(CC2)
        #LXRs.append(LXR)
        
        #Cálculo radiación difusa
        RGB_mean=calculo_radiacion(imagen_str,mascara)
        ##AÑADIR AJUSTE FINAL
        radiancia_RGB.append(RGB_mean)
        
        
        #Obtención valor piranómetro de difusa
        #print(date_fichero)
        print(date_fichero)
        pir=lectura_valor_piranometro(date_formato,date_fichero)
        #print('AQUI',pir)
        #print(RGB_mean/pir)
        valores_piranometro.append(pir)
        print(date,' RADICION DIFUSA: ',RGB_mean)
        
        cv2.destroyAllWindows()

    datos = {'nombre':dates,'pir':valores_piranometro,'GN2':GN2s,'EXP':EXPs,'GNG':GNGs,'GNR':GNRs,'GNB':GNBs,
             'radiancia_RGB':radiancia_RGB}
    
    df = pd.DataFrame(datos)
    print(df)
    df.to_excel('valores_parametros_'+nombre+'_MascaraYSolTapado.xlsx')
    flag_obligatorio_guardar_mascaras=0#Ya no es obligatorio tener las máscaras generadas
    print('Fin cálculo radiación difusa')
    
    
#Inicio programa con valores por defecto  
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
boton = ttk.Button(text="Elección imágenes", command=eleccion_imagenes, style="MyButton.TButton")
boton.place(x=50, y=150)
boton2=ttk.Button(text="Dibujo ángulos", command=dibujo_angulos_usuario, style="MyButton.TButton")
boton2.place(x=50, y=300)
boton2=ttk.Button(text="Máscara solar", command=aplicar_mascara_sol, style="MyButton.TButton")
boton2.place(x=50, y=450)
boton2=ttk.Button(text="Máscara solar sin píxeles saturados", command=mascara_solYangulos, style="MyButton.TButton")
boton2.place(x=50, y=600)
boton2=ttk.Button(text="Visualizar imágenes", command=mostrar_imagenes, style="MyButton.TButton")
boton2.place(x=300, y=150)
boton2=ttk.Button(text="Cambiar tamaño imágenes", command=cambiar_tamaño_imagenes, style="MyButton.TButton")
boton2.place(x=300, y=300)
boton2=ttk.Button(text="Calculo radiación difusa", command=calculo_radiacion_difusa, style="MyButton.TButton")
boton2.place(x=300, y=450)
boton2=ttk.Button(text="Máscara elección parte cielo", command=mascara_eleccion_usuario, style="MyButton.TButton")
boton2.place(x=300, y=600)
root.mainloop()