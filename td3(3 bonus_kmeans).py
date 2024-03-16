# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:49:54 2024

@author: Ye LIU
"""
#IMPORT 
import glob
import json
import time
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud



#FONCTIONS
def lire_fichier (chemin):
    with open(chemin,'r',encoding="utf-8")as f:
        texte=f.read()
    return texte

def lire_json (chemin):
    with open (chemin,'r',encoding="utf-8")as j:
        data=json.load(j)
    return data

def stocker_en_json(data,outpath):
    with open (outpath,'w',encoding='utf-8')as j:
        j.write(json.dumps(data,indent=2,ensure_ascii=False))
    return j


def kmeans(k,data):
    vectorizer=CountVectorizer()
    x=vectorizer.fit_transform(data)#x=matrice
     
    # k=20
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(x)
    
    centres=kmeans.cluster_centers_   
    labels=kmeans.labels_
    #print(len(labels))
    
    
    dic_clusters={}   
    for i, label in enumerate (labels):
        mot=data[i]
        
        if label not in dic_clusters:
            dic_clusters[label]=[]
        else :
            dic_clusters[label].append(mot)
        
    #print (dic_clusters)
    
    
    data_clusters={}
    for label, cluster in dic_clusters.items():
        if len(cluster)>1:
            cen=cluster[1]#??
            dist_termes=[]
            for t in cluster:
                vecteur=CountVectorizer(ngram_range=(2,3),analyzer='char')
                matrice=vecteur.fit_transform([cen,t]).toarray()
                tab_dist_cos=sklearn.metrics.pairwise.cosine_distances(matrice)
                dist_termes.append(tab_dist_cos[0][1])    
                #print (dist_termes)
            #dist_termes=set(dist_termes)
            data_clusters[cen]=list(set(dist_termes))
            
        else :
            print ('label vide:',label)
    
    #print (data_clusters)
    
    return data_clusters


# def dessiner_wordcloud(data,outpath_fig):
#     wordcloud=WordCloud(max_font_size=50,background_color='white')#initialiser un wc.  width=800,height=600
#     # width et height configure la taille de wordcloud, figsize confirgure celle de l'image?
#     # pour que les mots soient plus clair, élargir le wordcloud ou modifie dpi!
#     wordcloud.generate_from_frequencies(data)
  
#     plt.rc("figure", figsize=(10,8))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')#l'axe 
     
#     plt.savefig(outpath_fig,dpi=300,bbox_inches='tight')# = plt.tight_layout()
#     plt.show()
#     return plt


def dessiner_scatter(outpath_scatter, data):
    
    plt.rc("figure",figsize=(10,8))
    plt.rc("font",size=15)
    cmap=plt.cm.Blues
    
    
    for cen, dists in data.items():
        normalized_dists=[1-float(i)/len(dists) for i in range(len(dists))]
        for d, n_d in zip(dists,normalized_dists):#lie dist à son n_d correspondant 
            color=cmap(n_d)
            plt.scatter(d,cen,color=color,s=60)
              
    plt.xlabel('distance du termes au centroïde')
    plt.ylabel("centroïdes")
    plt.title("figure de clustering")
    plt.grid(True)
    plt.tight_layout()
       
    plt.savefig(outpath_fig, dpi=300)
    plt.show()
    
    
    return plt





#CODES 
start_time=time.time()
corpus_j='clustering_08032024//json_pour_cluster//CARRAUD*.json'
#corpus_j='DATA//*//*//*ents1.json'
for chemin in glob.glob(corpus_j):
    
    #print (chemin)
    nom=chemin.split('\\')[1].split("_")[0]
    print (nom)

    # nom=chemin.split('\\')[2]
    # print (nom)
    
    data=lire_json(chemin)
    print('data :', len(data))
    
    k=30
    data=kmeans (k,data)
    print (data)

     
    outpath_fig=f'clustering_08032024//json_pour_cluster//{nom}_scatter.png'
    plt=dessiner_scatter(outpath_fig,data)
    
    
    
end_time=time.time()
time=end_time-start_time
print ("time consumé :", time)


    
    
    
    
    
    