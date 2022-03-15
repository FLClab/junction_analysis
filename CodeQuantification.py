# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:59:04 2022

@author: olory
"""

import os
import re
path="C:/Users/olory/Desktop/PostDoc/ProjetIntestin/CodeQuantificationImageEllen/Patchlist/"
os.chdir(path)

def quantificationPatchlist(file):

    liste=[line.strip().split(";") for line in open (file, 'r')]
    
    # valeur 0 pour Nothing, 1 pour une structure et 2 pour ambiguous
    nothing=0
    structure=0
    ambiguous=0
    characters="[]"
    classComp05=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for i in range(len(liste)):
        if int(liste[i][4]) == 0:
            nothing+=1
        elif int(liste[i][4])==1:
            structure+=1
            liss=re.sub("\[|\]", "", liste[i][5] ).replace(" ", "" ).strip().split(",")
            liss=[float(a) for a in liss]
            for j in range(4):
                if liss[j]<0.5:
                    classComp05[j][0]+=1
                elif liss[j]==0.5:
                    classComp05[j][1]+=1
                elif liss[j]>0.5:
                    classComp05[j][2]+=1
                    
            
        elif int(liste[i][4])==2:
            ambiguous+=1
        
    return nothing, structure, ambiguous, classComp05
        

f=open(path+'Result.csv','w')
f.write("NomPatchlistTxt"+";"+"Nothing"+";"+"structure"+";"+"ambiguous"+";"+"classComp05 \n")

for file in os.listdir(): 
    
    if file.endswith(".txt"): 
        file_path = f"{path}\{file}"
        a,b,c,d=quantificationPatchlist(file_path)
        print(file_path, a,b,c,d)
        f.write(file_path+";"+str(a)+";"+str(b)+";"+str(c)+";"+str(d)+"\n")
        
f.close()

