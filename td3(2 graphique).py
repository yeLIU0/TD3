# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:16:32 2024

@author: Ye LIU 
"""

#IMPORTS
import glob
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import DistanceMetric
import numpy as np
import matplotlib.cm as cm


#FONCTIONS
def lire_json(chemin):
    with open(chemin,"r",encoding="utf-8")as j:
        liste=json.load(j)
    return liste

def stocker_en_json(data,outpath):
    with open (outpath,'w',encoding='utf-8')as j:
        j.write(json.dumps(data,indent=2,ensure_ascii=False))
    return j
        
       
# def dessiner_wordcloud(data,outpath_fig):
#     wordcloud=WordCloud(max_font_size=60,background_color='white',width=800,height=600)#initialiser un wc.  width=800,height=600
#     # width et height configure la taille de wordcloud, figsize confirgure celle de l'image?
#     # pour que les mots soient plus clair, élargir le wordcloud ou modifie dpi!
#     wordcloud.generate_from_frequencies(data)
  
#     plt.rc("figure", figsize=(10,8))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')#l'axe 
     
#     plt.savefig(outpath_fig,dpi=300,bbox_inches='tight')# = plt.tight_layout()
#     plt.show()
#     return plt

# def dessiner_scatter(data,outpath_scatter):
#     x=list(data.keys())
#     y=list(data.values())
#     s=[s*50 for s in y]
#     #print (x,y)

#     plt.rc("figure", figsize=(20,18))
#     plt.rc("font",size=10)
#     plt.scatter(y,x,s=s,color="blue",alpha=0.6)
    
#     # for x, y in data.items():
#     #     plt.text(x,y,x,ha='center', va='bottom')          
    
#     plt.ylabel("centroïde")
#     plt.xticks('nombre de termes du centroïde')
#     plt.title("graphique de clustering")
#     plt.grid(True)
#     plt.savefig(outpath_scatter,dpi=300)
#     plt.show()

#     return plt
    

def organiser_data(dic_clusters):
    data={}
    for index, cluster in dic_clusters.items():
        #print (index,cluster)   
        cen=cluster['Centroïde']
        termes=cluster['Termes']
      
        dist_termes=[]
        for t in termes :
            if t!=cen:
                vecteur=CountVectorizer(ngram_range=(2,3),analyzer='char')
                matrice=vecteur.fit_transform([cen,t]).toarray()
                tab_dist_cos=sklearn.metrics.pairwise.cosine_distances(matrice)
                dist_termes.append(tab_dist_cos[0][1])    
        #print (dist_termes)
        dist_termes=sorted(dist_termes)
                
        data[cen]=dist_termes 
    
    return data




def dessiner_scatter(outpath_scatter, data):
    
    plt.rc("figure",figsize=(20,18))
    plt.rc("font",size=15)
    

    cmap=plt.cm.Blues 
    for cen, dists in data.items():
        #print (dists)
        normalized_dists=[1-float(i)/len(dists) for i in range(len(dists))]
        #print (normalized_dists)
        #normalize les distances pour les faire correspondre à la colormap
        #float convertir une valeur en un nombre en virgule flottante, permet d'effectuer une division flottante plutôt qu'une division entière
        #3/10 serait arrond à 0
        
        for d, n_d in zip(dists,normalized_dists):#lie dist à son n_d correspondant 
            color=cmap(n_d)
            plt.scatter(d,cen,color=color,s=60)
            
            
    plt.xlabel('distance du termes au centroïde')
    plt.ylabel("centroïdes")
    plt.title("figure de clustering")
    plt.grid(True)
    plt.tight_layout()
       
    plt.savefig(outpath_scatter, dpi=300)
    plt.show()
    
    return plt






#codes:
corpus_clusters='clustering_08032024//cluster_pour_graphique//*.json'

for chemin in glob.glob(corpus_clusters):
    #print (chemin)

    nom=chemin.split("\\")[1].split("_")[0]
    print (nom)

    dic_clusters=lire_json(chemin)
    #print (dic_clusters)
    print ('nb de clusters :', len (dic_clusters))

    data=organiser_data(dic_clusters)
    #print (data)
    
    outpath_scatter=f'clustering_08032024//cluster_pour_graphique//{nom}_scatter.png'    
    plt=dessiner_scatter(outpath_scatter,data)

    

    


