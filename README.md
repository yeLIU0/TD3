# TD3
16/3/24 Ye LIU 21301082 
1.1 Plan 
1. On lit le fichier bio de la même manière qu’on lit le fichier csv et stocker chaque ligne dans une liste. 
2. Parce que on s’intéresse seulement aux entités nommées, c’est-à
-dire les entités marquées de B ou I. On configure les paramètres pour enlever les lignes vide, les entités marquées de O. 
3. On reconstruit les entités : on parcourt la liste, si la marque commence par B, on examine l’entité suivante, si elle est marquée de I, on la joint dans l’entité précédente, jusqu’à rencontrer une autre entité marquée de B. 
4. On obtient une liste d’entités nommées.


Conclusion : 
On représente les centroïde et leurs termes par les points. Dans ce graphique, l’axe Y présente les centroïde, l’axe X signifie la distance d’un terme à son centroïde correspondant. Plus la couleur est foncée, plus il est proche de centroïde, plus possible qu’il est une entité. Contrairement, il serait considéré comme le bruit.
Par ailleurs, selon le graphique, on peut comparer le nombre de termes regroupés par les centroïde différents. Plus les points sont nombreux, on suppose que ce centroïde est plus fréquents dans le corpus. Plus les points sont proches de 0, ce centroïde est performant pour assembler les termes similaires.



Bonus :
On opte pour kmeans comme alternatif pour cluster les mots. 
Ses avantages sont évidents. Il donne rapidement un résultat, parce qu’il utilise CountVectorizer ou TfidfVectorizer en entrée à classer les mots. 
Néanmoins, ses limites sont multiples. D’un côté, il faut fixer avant tout le nombre de clusters, de l’autre côté, son clustering n’est pas assez précise que Affinity Propa-gation, c’est-à-dire que les termes d’un même cluster peuvent se différer considéra-blement.
