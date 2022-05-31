# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:03:29 2022

@author: raque
"""
from __future__ import print_function
import argparse
from datetime import datetime as dt
import pandas as pd
import glob
import cv2
import numpy as np

dir= 'D:/Software/Anaconda/Proyectos/'
imagenes=glob.glob(dir+'2022-05-29'+'*.jpg')#selección imágenes
mascaras=glob.glob(dir+'mascaraYsolTapado_2022-05-29'+'*.jpg')#selección imágenes

#obtener formato fecha a buscar en el fichero de datos
dates=[]
dates_fichero=[]
for imagen in imagenes:
    fecha = dt.strptime(imagen.split('\\')[1].split('.')[0], '%Y-%m-%d_%H_%M_%S')
    #print(fecha)
    date=fecha.strftime('%Y/%m/%d %H:%M')
    date_fichero=fecha.strftime('%Y_%m_%d')
    
    dates.append(date)
    dates_fichero.append(date_fichero)
    #print(fecha)
    

resultado=pd.Series([imagenes])
EXIs=[]
EXAs=[]
EXPs=[]
NOIs =[]
AVBs=[]
GNGs =[]
GNRs =[]
GNBs =[]
GN2s =[]
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

radiancia_RGB=[]

valores_comp=[]
valores_piranometro=[]
v_ts=[]

B_gammas=[]
R_gammas=[]
G_gammas=[]

cte_unidades=1e6
#sensibilidad=3.8e-5 #Relación ganancia global con los valores medios de cada canal 
sensibilidad=-7e-5

def adjust_gamma(imagen, gamma=1):#Invierte correccion gamma
    '''
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
    '''
    '''
    cv2.imshow('Sin gamma', imagen)
    cv2.waitKey(2000)
    '''
    for columna in range(len(imagen[0])):
        for fila in range(len(imagen)):    
            aux=float(imagen[fila,columna]/255)
            if aux<0.04045:
                aux=aux/12.92
                imagen[fila,columna]=aux*255
            else:
                aux=((aux+0.055)/1.055)**gamma
                imagen[fila,columna]=aux*255
    '''
    cv2.imshow('Con gamma', imagen)
    cv2.waitKey(2000)
    '''
    return imagen

for (imagen,mascara,date_fichero,date) in zip(imagenes,mascaras,dates_fichero,dates):
    #print(imagen)
    #Lectura parámetros imágenes
    f = open (imagen,mode='rb')
    mensaje=f.read(1434) ##lee hasta los parámetros de los sensores
    mensaje=mensaje.decode('unicode_escape')#Lo traduce a ASCII
    '''
    cv2.imshow('Foto', imagen)
    cv2.waitKey(2000)
    '''
    
    #cv2.imshow('Foto', gamma)
    #cv2.waitKey(2000)
    
    #lectura de los parámetros
    EXI =float(mensaje.split('EXI=')[1].split('\r')[0])
    EXA =float(mensaje.split('EXA=')[1].split('\r')[0])
    EXP =float(mensaje.split('EXP=')[1].split('\r')[0])
    NOI =float(mensaje.split('NOI=')[1].split('\r')[0])
    AVB =float(mensaje.split('AVB=')[1].split('\r')[0])
    GNG =float(mensaje.split('GNG=')[1].split('\r')[0])
    GNR =float(mensaje.split('GNR=')[1].split('\r')[0])
    GNB =float(mensaje.split('GNB=')[1].split('\r')[0])
    GN2 =float(mensaje.split('GN2=')[1].split('\r')[0])
    CCG =float(mensaje.split('CCG=')[1].split('\r')[0])
    CCR =float(mensaje.split('CCR=')[1].split('\r')[0])
    CCB =float(mensaje.split('CCB=')[1].split('\r')[0])
    CC2 =float(mensaje.split('CC2=')[1].split('\r')[0])
    LXR =float(mensaje.split('LXR=')[1].split('\r')[0])
    
    EXIs.append(EXI)
    EXAs.append(EXA)
    EXPs.append(EXP)
    NOIs.append(NOI)
    AVBs.append(AVB)
    GNGs.append(GNG)
    GNRs.append(GNR)
    GNBs.append(GNB)
    GN2s.append(GN2)
    CCGs.append(CCG)
    CCRs.append(CCR)
    CCBs.append(CCB)
    CC2s.append(CC2)
    LXRs.append(LXR)
    
    #ANTES
    '''
    data=cv2.imread(imagen)
    #cv2.imshow('Foto', data)
    #cv2.waitKey(2000)
    
    #cv2.imshow('Foto', imagen_gamma)
    #cv2.waitKey(2000)
    imagen_gamma=adjust_gamma(data)
    data_hsv = cv2.cvtColor(imagen_gamma, cv2.COLOR_BGR2HSV)
    #data_hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV) ##funciona
    #print(data_hsv)
    '''
    
    data=cv2.imread(mascara)
    #cv2.imshow('Foto', data)
    #cv2.waitKey(2000)
    
    #cv2.imshow('Foto', imagen_gamma)
    #cv2.waitKey(2000)
    
    #HSV_value = (np.mean(v[:])/255)#En matlab es sobre 1 y en python sobre 255
    B=data[:,:,0]
    G=data[:,:,1]
    R=data[:,:,2]
    
    B_gamma=adjust_gamma(B)
    G_gamma=adjust_gamma(G)
    R_gamma=adjust_gamma(R)
    #print(B_gamma)
    #
    B_val=((np.mean(B_gamma))/(GNB*EXP))*(1+sensibilidad*GNB)*cte_unidades
    G_val=((np.mean(G_gamma))/(GNG*EXP))*(1+sensibilidad*GNG)*cte_unidades
    R_val=((np.mean(R_gamma))/(GNR*EXP))*(1+sensibilidad*GNR)*cte_unidades
    B_gammas.append(np.mean(B_gamma))
    R_gammas.append(np.mean(R_gamma))
    G_gammas.append(np.mean(G_gamma))
    
    #SIN NEGROS
    '''
    B_val_sin_negros=0
    contador=0  
    #Media sin contar los pixeles negros
    for columna in range(len(B_gamma[0])):
        for fila in range(len(B_gamma)):
            if B_gamma[fila,columna]!=0:
                B_val_sin_negros+=B_gamma[fila,columna]
                contador+=1
    B_val_sin_negros/=contador
    
    B_val_sin_negros=(B_val_sin_negros/(GNB*EXP))*(1+sensibilidad*GNB)*1e6
    #
    R_val_sin_negros=0
    contador=0  
    #Media sin contar los pixeles negros
    for columna in range(len(R_gamma[0])):
        for fila in range(len(R_gamma)):
            if R_gamma[fila,columna]!=0:
                R_val_sin_negros+=R_gamma[fila,columna]
                contador+=1
    R_val_sin_negros/=contador
    
    R_val_sin_negros=(R_val_sin_negros/(GNR*EXP))*(1+sensibilidad*GNR)*1e6
    
    #
    G_val_sin_negros=0
    contador=0  
    #Media sin contar los pixeles negros
    for columna in range(len(G_gamma[0])):
        for fila in range(len(G_gamma)):
            if G_gamma[fila,columna]!=0:
                G_val_sin_negros+=G_gamma[fila,columna]
                contador+=1
    G_val_sin_negros/=contador
    
    G_val_sin_negros=(G_val_sin_negros/(GNG*EXP))*(1+sensibilidad*GNG)*1e6
    
    RGB_mean_sin_negros=(R_val_sin_negros+G_val_sin_negros+B_val_sin_negros)/3
    
    '''
    RGB_mean=(R_val+G_val+B_val)/3
    radiancia_RGB.append(RGB_mean)
    
    
    #Valor piranómetro
    print(date_fichero)
    f = open (('meteo'+ date_fichero+'.txt'),mode='rb')
    
    mensaje=f.read() ##lee el archivo entero
    mensaje=mensaje.decode('unicode_escape')#Lo traduce a ASCII  
    pir =float(mensaje.split(date)[1].split('\t')[4].split('\t')[0])
    #print('AQUI',pir)
    print(RGB_mean/pir)
    valores_piranometro.append(pir)
    #print(HSV_value, G,R,B, RGB_mean)
    '''
    valor_comp=HSV_value/(GN2*EXP)
    valores_comp.append(valor_comp)
    
    #Con LXR
    valor=HSV_value/(GN2*EXP)/LXR*10000000000
    valores.append(valor)
    valor_G=G_val/(GNG*EXP)/LXR*100000000
    valores_G.append(valor_G)
    valor_R=R_val/(GNR*EXP)/LXR*100000000
    valores_R.append(valor_R)
    valor_B=B_val/(GNB*EXP)/LXR*100000000
    valores_B.append(valor_B)
    valor_RGB=RGB_mean/(GN2*EXP)/LXR*10000000000
    valores_RGB.append(valor_RGB)
    
    #Con piranómetro
    valor_pir=HSV_value/(GN2*EXP)/pir*10000000000
    valores_pir.append(valor_pir)
    valor_G_pir=G_val/(GNG*EXP)/pir*100000000
    valores_G_pir.append(valor_G_pir)
    valor_R_pir=R_val/(GNR*EXP)/pir*100000000
    valores_R_pir.append(valor_R_pir)
    valor_B_pir=B_val/(GNB*EXP)/pir*100000000
    valores_B_pir.append(valor_B_pir)
    valor_RGB_pir=RGB_mean/(GN2*EXP)/pir*10000000000
    valores_RGB_pir.append(valor_RGB_pir)
    '''
    
    #print(imagen,valor,valor_G,valor_R,valor_B,valor_RGB)
    
    f.close()
    cv2.destroyAllWindows()
'''    
datos = {'nombre':dates,'valor':valores,'valor_G':valores_G,'valor_R':valores_R,'valor_B':valores_B,'valor_RGB':valores_RGB,
         'EXI':EXIs,'EXA':EXAs,'EXP':EXPs,'NOI':NOIs,'AVB':AVBs,'GNG':GNGs,'GNR':GNRs,'GNB':GNBs,'GN2':GN2s,
         'CCG':CCGs, 'CCR':CCRs,'CCB':CCBs,'CC2':CC2s, 'LXR':LXRs}
'''
'''
datos = {'nombre':dates,'HSV_value/(GN2*EXP)/LXR*10000000000':valores,'G_val/(GNG*EXP)/LXR*100000000':valores_G,'R_val/(GNR*EXP)/LXR*100000000':valores_R,
         'B_val/(GNB*EXP)/LXR*100000000':valores_B,'RGB_mean/(GN2*EXP)/LXR*10000000000':valores_RGB,
         'HSV_value/(GN2*EXP)/pir*10000000000':valores_pir,'G_val/(GNG*EXP)/pir*100000000':valores_G_pir,'R_val/(GNR*EXP)/pir*100000000':valores_R_pir,
         'B_val/(GNB*EXP)/pir*100000000':valores_B_pir,'RGB_mean/(GN2*EXP)/pir*10000000000':valores_RGB_pir,
         'HSV_value/(GN2*EXP)':valores_comp,'LXR':LXRs,'pir':valores_piranometro,'GN2':GN2s,'radiancia_RGB':radiancia_RGB}
'''
print(len(dates),len(valores_piranometro),len(imagenes),len(mascaras))
datos = {'nombre':dates,'pir':valores_piranometro,'GN2':GN2s,'EXP':EXPs,'GNG':GNGs,'GNR':GNRs,'GNB':GNBs,
         'radiancia_RGB':radiancia_RGB,'B_gamma(media)':B_gammas,'R_gamma(media)':R_gammas,'G_gamma(media)':G_gammas}
#B_val=((np.mean(B_gamma))/(GNB*EXP))*(1+sensibilidad*GNB)*cte_unidades
df = pd.DataFrame(datos)
print(df)
df.to_excel('valores_parametros_2022-05-29_MascaraYSolTapado.xlsx')