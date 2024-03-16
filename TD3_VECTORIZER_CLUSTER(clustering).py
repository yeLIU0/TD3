#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""


import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import DistanceMetric 
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import re
from collections import OrderedDict
import time



def lire_fichier (chemin):
    with open(chemin,"r",encoding='utf-8') as json_data: 
        texte =json.load(json_data)
    return texte 

#lire le fichier fichier json, obtenir une liste


def nomfichier(chemin):
    nomfich= chemin.split("/")[-1]
    nomfich= nomfich.split(".")
    nomfich= ("_").join([nomfich[0],nomfich[1]])
    return nomfich
    
def stocker_en_json(data,outpath):
    with open (outpath,'w',encoding='utf-8')as j:
        j.write(json.dumps(data,indent=2,ensure_ascii=False))
    return j


#chemin_entree =



# for subcorpus in glob.glob(path_copora):
# #    print("SUBCORPUS***",subcorpus)
#     liste_nom_fichier =[]#?
    
#     for path in glob.glob("%s/AIMARD-TRAPPEURS_MOD/AIMARD_les-trappeurs_TesseractFra-PNG.txt_SEM_WiNER.ann_SEM.json-concat.json"%subcorpus):
# #        print("PATH*****",path)
        
#         nom_fichier = nomfichier(path)
# #        print(nom_fichier)
# #        extraire le nom de fichier
#         liste=lire_fichier(path)
# #       lire le fichier json en une liste d'entités



#### FREQUENCE ########

liste_nom_fichier =[]#?


start_time=time.time()

#corpus="DATA//*//*//*_ents.json"
corpus='DATA\\NOAILLES\\NOAILLES_TesseractFra-PNG\\NOAILLES_TesseractFra-PNG_ents_décomposés.json'
    

for chemin in glob.glob(corpus):
    #print (chemin)
    livre=chemin.split("\\")[2]
    auteur=chemin.split("\\")[1]
    print (livre)
    
    
    liste=lire_fichier(chemin)
    #print (len(liste))
         
    
    dic_mots={}
    i=0
    for mot in liste: 
        if mot not in dic_mots:
            dic_mots[mot] = 1
        else:
            dic_mots[mot] += 1
    #compter les fréquence des ents de la liste
    #stocker le nombre dans un dictionnaire {mot: nb}
        
    i += 1#?????????
    
    # print (len(dic_mots))
    # break  


    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))
    # lambda : trier dic selon t(tuple dans dic)) par ordre alphabétique
    # t[0] renvoie le mot 
    
    # print (new_d)
    # OrdreDict ([(mot:nb)])  un tuple de liste de tuples
    
    freq=len(dic_mots.keys())
    print (freq)
    
    
    Set_00 = set(liste)
    Liste_00 = list(Set_00)
    #obtenir une liste de mots sans répétition.
    print (len(Liste_00))
    

    dic_output = {}
    liste_words=[]
    matrice=[]
    for l in Liste_00:
        
        if len(l)!=1:
            liste_words.append(l)
    #si le mot est plus long qu'un caractère, pour éliminer les espace et la ponctuation?
    # on l'arrange dans la liste_word
    
    
    try: # utilisé avec except pour attraper les exceptions
        words = np.asarray(liste_words)
        # convertit une séquence en un tableau NumPy. 
        # ? performance amélioré par rapport à la liste dans le calcul de distances  
        
        for w in words:
            liste_vecteur=[]
                
            for w2 in words:
            
                    V = CountVectorizer(ngram_range=(2,3), analyzer='char')
                    X = V.fit_transform([w,w2]).toarray()
                    distance_tab1=sklearn.metrics.pairwise.cosine_distances(X)            
                    liste_vecteur.append(distance_tab1[0][1])
                
            matrice.append(liste_vecteur)
        # calculer la distance entre w et les éléments dans ce tableau, soi-même compris 
        # on obtient les similarité!!! de chaque élément dans words avec tous les éléments dans words
        # préparer à regrouper les mots en cluster selon la similarité
        # mot->distance
         
        
        matrice_def=-1*np.array(matrice)
        # psk on calculer la similarité, plus proche de 1 'chiffre plus grand), plus simlaire
        # en convertissant le chiffre en négatif, le chiffre absolu plus petit, plus proche
        #? pq ne calcule directement la distance? plus conforme à nos connaissance de données :)
        
        
        #array()prend en charfe plusieurs types d'argument, comme dans la liste_words, il y a token, espace, ponctuation...
        #asarry(): memes types, comme dans le matrice (distances)
        
        
                          
        affprop = AffinityPropagation(affinity="precomputed", damping= 0.6, random_state = None) 
        # AffinityPropagation regroupe les données similaires dans un cluster 
        # il détermine le nombre de cluster comme il juge bon (à partir de données)
        # donc on n'a pas à préciser le nombre de cluster
        
        # configurer quelques paramètres de l'algotithme:
        #'preccomupted' nous demande de fournir un matrice de similarité pré-calculé en entrée, au lieu de data brut
        # 'damping' contrôle la convergence de l'algorithme. 
        # plus proche de 1 signifie que les mises à jour sont moins atténuées, ce qui peut conduire à une convergence plus rapide mais aussi à une instabilité. 
        # 'random state' donne un nombre aléatoire (interne de l'algo)
        # si 'None', le nombre sera alétoire chaque fois
        # =si on définit un nombre, on peut obtenir même nombre de cluster dans chaque données?  
        
        
        affprop.fit(matrice_def)
        #fit=appliquer cet algorithme au matrice qu'on obtient
        
        
    
        for cluster_id in np.unique(affprop.labels_): 
            #np.unique a pour but de trouver l'élément unique
            #s'applique à np.array, tri par l'ordre croissant, produit un array
            #ex. 1 1 2 2 2 3 4 4 4 > trouve [1, 2, 3, 4] 
            
            #.labels_ : chaque membre d'un même cluster est étiquetté par le même nombre
            
            
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            #rappel: words est array de liste de mots plus long qu'une lettre sans répétition
            
            #cluster_centers_indices indique l'index de mot exemplaire d'un certain label dans l'array
            # ex. id est 0, correspond à clutser centers index 1, mot exemplaire de label 0 est words[1]
            #index -> mot
            
            cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
            #np.zero renvoie le résultat non-zero
            #collecte tous les mots d'un meme label
            
            cluster_str = ", ".join(cluster)
            cluster_list = cluster_str.split(", ")
            
            
            Id = "ID "+str(i)
            for cle, dic in new_d.items(): #dic qui compte les freq des mots et organisé par l'ordre alphbétique
                if cle == exemplar:
                    dic_output[Id] ={}
                    dic_output[Id]["Centroïde"] = exemplar
                    dic_output[Id]["Freq. centroide"] = dic#???
                    dic_output[Id]["Termes"] = cluster_list
            
            i=i+1#index de cluster, renouveller 
            
            
        #    print(dic_output)
        outpath=f'DATA\\{auteur}\\{livre}\\{livre}_clusters.json'
        j=stocker_en_json(dic_output, outpath)



    except :        
        print("**********Non OK***********", chemin)
        liste_nom_fichier.append(chemin)
        #si le code ne marche pas pour le fichier, on print non ok + son chemin.
        
        
        continue 
    break # chq liste
    
    #break #chq ents
end_time=time.time()
time=end_time-start_time
print ("time consumé :", time)


    
    
    
    
    
    
    

    
    
    
    
    
    
    

        
