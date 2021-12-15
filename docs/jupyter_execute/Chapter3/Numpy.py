#!/usr/bin/env python
# coding: utf-8

# ## Numpy (tableaux de données multi-dimensionnels) 
# 
# 

# Librairie de calcul numérique permettant notamment de manipuler des tableaux de dimension quelconque.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ### Introduction
# 
#  

# * `numpy` est un module utilisé dans presque tous les projets de calcul numérique sous Python
#    * Il fournit des structures de données performantes pour la manipulation de vecteurs, matrices et tenseurs plus généraux
#    * `numpy` est écrit en C et en Fortran d'où ses performances élevées lorsque les calculs sont vectorisés (formulés comme des opérations sur des vecteurs/matrices)

# Pour utiliser `numpy`  il faut commencer par l'importer :

# In[3]:


import numpy as np


# 
# 
# Dans la terminologie `numpy`, vecteurs, matrices et autres tenseurs sont appelés *arrays*.
# 

# ### Création d'*arrays* `numpy` 
# 
# Plusieurs possibilités:
# 
#  * à partir de listes ou n-uplets Python
#  * en utilisant des fonctions dédiées, telles que `arange`, `linspace`, etc.
#  * par chargement à partir de fichiers
# 
# 

# 
# #### À partir de listes
# 
# Au moyen de la fonction `numpy.array` :

# In[4]:


# un vecteur : l'argument de la fonction est une liste Python
v = np.array([1, 3, 2, 4])
print(v)
print(type(v))


# Pour définir une matrice (array de dimension 2 pour numpy):
# 

# In[5]:


# une matrice : l'argument est une liste de liste
M = np.array([[1, 2], [3, 4]])
print(M)


# In[6]:


M[0, 0]


# Les objets `v` et `M` sont tous deux du type `ndarray` (fourni par `numpy`)

# In[7]:


type(v), type(M)


# `v` et `M` ne diffèrent que par leur taille, que l'on peut obtenir via la propriété `shape` :

# In[8]:


v.shape


# In[9]:


M.shape


# Pour obtenir le nombre d'éléments d'un *array* :

# In[10]:


v.size


# In[11]:


M.size


# On peut aussi utiliser `numpy.shape` et `numpy.size`

# In[12]:


np.shape(M)


# Les *arrays* ont un type qu'on obtient via `dtype` :

# In[13]:


print( M)
print(M.dtype)


# Les types doivent être respectés lors d'assignations à des *arrays*

# In[14]:


M[0,0] = "hello"


# On peut modifier le type d'un array après sa déclaration en utilisant *astype*

# In[217]:


a = np.array([1,2,3], dtype=np.int64)
b = np.array([2,2,3], dtype=np.int64)
b = b.astype(float)
print(a / b)


# On peut définir le type de manière explicite en utilisant le mot clé `dtype` en argument: 

# In[218]:


M = np.array([[1, 2], [3, 4]], dtype=complex)
M


#  * Autres types possibles avec `dtype` : `int`, `float`, `complex`, `bool`, `object`, etc.
# 
#  * On peut aussi spécifier la précision en bits: `int64`, `int16`, `float128`, `complex128`.

# #### En utilisant des fonctions de génération d'*arrays*

# #### arange

# In[219]:


# create a range
x = np.arange(0, 10, 2) # arguments: start, stop, step
x


# In[220]:


x = np.arange(-1, 1, 0.1)
x


# ##### linspace and logspace

# In[221]:


# avec linspace, le début et la fin SONT inclus
np.linspace(0, 10, 25)


# In[222]:


np.linspace(0, 10, 11)


# In[223]:


print(np.logspace(0, 10, 10, base=np.e))


# ##### mgrid

# In[224]:


x, y = np.mgrid[0:5, 0:5] 


# In[225]:


x


# In[226]:


y


# ##### Données aléatoires

# In[227]:


from numpy import random


# In[228]:


# tirage uniforme dans [0,1]
random.rand(5,5)  # ou np.random.rand


# In[229]:


# tirage suivant une loi normale standard
random.randn(5,5)


# ##### diag

# In[230]:


# une matrice diagonale
np.diag([1,2,3])


# In[231]:


# diagonale avec décalage par rapport à la diagonale principale
np.diag([1,2,3], k=1)


# ##### zeros, ones et identity

# In[232]:


np.zeros((3,), dtype=int)  # attention zeros(3,3) est FAUX


# In[233]:


np.ones((3,3))


# In[234]:


print(np.zeros((3,), dtype=int))
print(np.zeros((1, 3), dtype=int))
print(np.zeros((3, 1), dtype=int))


# In[235]:


print(np.identity(3))


# ####  À partir de fichiers d'E/S

# ##### Fichiers séparés par des virgules (CSV)
# 
# Un format fichier classique est le format CSV (comma-separated values), ou bien TSV (tab-separated values). Pour lire de tels fichiers utilisez `numpy.genfromtxt`. Par exemple:

# In[236]:


data = np.genfromtxt('files/DONNEES.csv', delimiter=',')
data


# In[237]:


data.shape


# A l'aide de `numpy.savetxt` on peut enregistrer un *array* `numpy` dans un fichier txt:

# In[238]:


M = random.rand(3,3)
M


# In[239]:


np.savetxt("random-matrix.txt", M)


# In[240]:


np.savetxt("random-matrix.csv", M, fmt='%.5f', delimiter=',') # fmt spécifie le format


# ##### Format de fichier Numpy natif
# 
# Pour sauvegarder et recharger des *array* `numpy` : `numpy.save` et `numpy.load` :

# In[241]:


np.save("random-matrix.npy", M)


# In[242]:


np.load("random-matrix.npy")


# #### Autres propriétés des *arrays* `numpy`

# In[243]:


M


# In[244]:


M.dtype


# In[245]:


M.itemsize # octets par élément


# In[246]:


M.nbytes # nombre d'octets


# In[247]:


M.nbytes / M.size


# In[248]:


M.ndim # nombre de dimensions


# In[249]:


print(np.zeros((3,), dtype=int).ndim)
print( np.zeros((1, 3), dtype=int).ndim)
print (np.zeros((3, 1), dtype=int).ndim)


# ### Manipulation et Opérations sur les *arrays*

# #### Indexation

# In[250]:


# v est un vecteur, il n'a qu'une seule dimension -> un seul indice
v[0]


# In[251]:


# M est une matrice, ou un array à 2 dimensions -> deux indices 
M[1,1]


# Contenu complet :

# In[252]:


M


# La deuxième ligne :

# In[253]:


M[1]


# On peut aussi utiliser `:` 

# In[254]:


M[1,:] # 2 ème ligne (indice 1)


# In[255]:


M[:,1] # 2 ème colonne (indice 1)


# In[256]:


print(M.shape)
print( M[1,:].shape, M[:,1].shape)


# On peut assigner des nouvelles valeurs à certaines cellules :

# In[257]:


M[0,0] = 1


# In[258]:


M


# In[259]:


# on peut aussi assigner des lignes ou des colonnes
M[1,:] = -1
# M[1,:] = [1, 2, 3]


# In[260]:


M


# #### *Slicing* ou accès par tranches
# 
# *Slicing* fait référence à la syntaxe `M[start:stop:step]` pour extraire une partie d'un *array* :

# <center>
# <img src="images/numpy_indexing.png" width="700">
# </center>

# In[261]:


A = np.array([1,2,3,4,5])
A


# In[262]:


A[1:3]


# Les tranches sont modifiables :

# In[263]:


A[1:3] = [-2,-3]
A


# On peut omettre n'importe lequel des argument dans `M[start:stop:step]`:

# In[264]:


A[::] # indices de début, fin, et pas avec leurs valeurs par défaut


# In[265]:


A[::2] # pas = 2, indices de début et de fin par défaut


# In[266]:


A[:3] # les trois premiers éléments


# In[267]:


A[3:] # à partir de l'indice 3


# In[268]:


M = np.arange(12).reshape(4, 3)
print( M)


# On peut utiliser des indices négatifs :

# In[269]:


A = np.array([1,2,3,4,5])


# In[270]:


A[-1] # le dernier élément


# In[271]:


A[-3:] # les 3 derniers éléments


# Le *slicing* fonctionne de façon similaire pour les *array* multi-dimensionnels

# In[272]:


A = np.array([[n+m*10 for n in range(5)] for m in range(5)])

A


# In[273]:


A[1:4, 1:4]  # sous-tableau


# In[274]:


# sauts
A[::2, ::2]


# In[275]:


A


# In[276]:


A[[0, 1, 3]]


# #### Indexation avancée (*fancy indexing*)
# 
# Lorsque qu'on utilise des listes ou des *array* pour définir des tranches : 

# In[277]:


row_indices = [1, 2, 4]
print( A)
print("\n")
print ( A[row_indices])
# print( A.shape)


# In[278]:


A[[1, 2]][:, [3, 4]] = 0  # ATTENTION !
print( A)


# In[279]:


print ( A[[1, 2], [3, 4]])


# In[280]:


A[np.ix_([1, 2], [3, 4])] = 0
print ( A)


# On peut aussi utiliser des masques binaires :

# In[281]:


B = np.arange(5)
B


# In[282]:


row_mask = np.array([True, False, True, False, False])
print(  B[row_mask])
print(  B[[0,2]])


# In[283]:


# de façon équivalente
row_mask = np.array([1,0,1,0,0], dtype=bool)
B[row_mask]


# In[284]:


# ou encore
a = np.array([1, 2, 3, 4, 5])
print(  a < 3)
print(  B[a < 3])


# In[285]:


print(  A,"\n")
print(  A[:, a < 3])


# #### Opérations élément par élément

# On déclare  aa  et  bb  sur lesquelles nous allons illustrer quelques opérations

# In[286]:


a = np.ones((3,2))
b = np.arange(6).reshape(a.shape)
print(a)
b


# Les opérations arithmétiques avec les scalaires, ou entre arrays s'effectuent élément par élément. Lorsque le dtype n'est pas le même ( aa  contient des float,  bb  contient des int), numpy adopte le type le plus "grand" (au sens de l'inclusion).

# In[287]:


print( (a + b)**2 )
print( np.abs( 3*a - b ) )
f = lambda x: np.exp(x-1)
print( f(b) )


# In[288]:


1/b


# #### Broadcasting

# <center>
# <img src="images/fig_broadcast_visual_1.png" width="700" hight="500">
# </center>

# Que se passe-t-il si les dimensions sont différentes?

# In[289]:


c = np.ones(6)
c


# In[290]:


b+c   # déclenche une exception


# In[291]:


c = np.arange(3).reshape((3,1))
print(b,c, sep='\n\n')
b+c


# L'opération précédente fonctionne car numpy effectue ce qu'on appelle un [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) de c : une dimension étant commune, tout se passe comme si on dupliquait c sur la dimension non-partagée avec b. Vous trouverez une explication visuelle simple ici :

# In[292]:


a = np.zeros((3,3))
a[:,0] = -1
b = np.array(range(3))
print(a + b)


# ### Extraction de données à partir d'*arrays* et création d'*arrays*

# #### where
# 
# Un masque binaire peut être converti en indices de positions avec `where`

# In[293]:


x = np.arange(0, 10, 0.5)
print ( x)
mask = (x > 5) * (x < 7.5)
print(  mask)
indices = np.where(mask)
indices


# In[294]:


x[indices] # équivalent à x[mask]


# #### diag
# 
# Extraire la diagonale ou une sous-diagonale d'un *array* :

# In[295]:


print ( A)
np.diag(A)


# In[296]:


np.diag(A, -1)


# ### Algèbre linéaire
# 
# La performance des programmes écrit en Python/Numpy dépend de la capacité à vectoriser les calculs (les écrire comme des opérations sur des vecteurs/matrices) en évitant au maximum les boucles `for/while`.
# 
# 
# 

# Vous avez un éventail de fonctions pour faire de l'algèbre linéaire dans [numpy](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html) ou dans [scipy](https://docs.scipy.org/doc/scipy/reference/linalg.html). Cela peut vous servir si vous cherchez à faire une décomposition matricielle particulière ([LU](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.lu.html), [QR](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.qr.html), [SVD](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html),...), si vous vous intéressez aux valeurs propres d'une matrice, etc...

# #### Opérations scalaires
# 
# On peut effectuer les opérations arithmétiques habituelles pour multiplier, additionner, soustraire et diviser des *arrays* avec/par des scalaires :

# In[297]:


v1 = np.arange(0, 5)
print (v1)


# In[298]:


v1 * 2


# In[299]:


v1 + 2


# In[300]:


A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
print(  A)


# In[301]:


print(  A * 2)


# In[302]:


print(  A + 2)


# #### Opérations terme-à-terme sur les *arrays*
# 
# Les opérations par défaut sont des opérations **terme-à-terme** :

# In[303]:


A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
print ( A)


# In[304]:


A * A # multiplication terme-à-terme


# In[305]:


(A + A.T) / 2


# In[306]:


print(  v1)
print(  v1 * v1)


# En multipliant des *arrays* de tailles compatibles, on obtient des multiplications terme-à-terme par ligne :

# In[307]:


A.shape, v1.shape


# In[308]:


print(  A)
print(  v1)
print(  A * v1)


# #### Algèbre matricielle
# 
# Comment faire des multiplications de matrices ? Deux façons :
#  
#  * en utilisant les fonctions [dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html); 
#  * en utiliser le type [matrix](http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html).
# 

# In[309]:


print( A.shape)
print( A)
print( type(A))


# In[310]:


print( np.dot(A, A))  # multiplication matrice
print( A * A ) # multiplication élément par élément


# In[311]:


A.dot(v1)


# In[312]:


np.dot(v1, v1)


# Avec le type [matrix](http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html) de Numpy

# In[313]:


M = np.matrix(A)
v = np.matrix(v1).T # en faire un vecteur colonne


# In[314]:


M * v


# In[315]:


# produit scalaire
v.T * v


# In[316]:


# avec les objets matrices, c'est les opérations standards sur les matrices qui sont appliquées
v + M*v


# Si les dimensions sont incompatibles on provoque des erreurs :

# In[317]:


v = np.matrix([1,2,3,4,5,6]).T


# In[318]:


np.shape(M), np.shape(v)


# In[319]:


M * v


# Voir également les fonctions : `inner`, `outer`, `cross`, `kron`, `tensordot`. Utiliser par exemple `help(kron)`.

# On peut calculer l'inverse ou le déterminant de  $A$ 

# In[320]:


A = np.tril(np.ones((3,3)))
b = np.diag([1,2, 3])
print(A)
print("-------")
print(np.linalg.det(A))  # déterminant de la matrice A
print("-------")
inv_A = np.linalg.inv(A)   # inverse de la matrice A
print(inv_A)
print("-------")
print(inv_A.dot(A))


# ... résoudre des systèmes d'equations linéaires du type  $Ax=b$ ...

# In[321]:


x = np.linalg.solve(A, np.diag(b))
print(np.diag(b))
print(x)
print(A.dot(x))


# ... ou encore obtenir les valeurs propres de  $A$

# In[322]:


np.linalg.eig(A)


# In[323]:


np.linalg.eigvals(A)


# #### Transformations d'*arrays* ou de matrices

#  * Plus haut `.T` a été utilisé pour transposer l'objet matrice `v`
#  * On peut aussi utiliser la fonction `transpose`
# 
# **Autres transformations :**
# 

# In[324]:


C = np.matrix([[1j, 2j], [3j, 4j]])
C


# In[325]:


np.conjugate(C)


# Transposée conjuguée :

# In[326]:


C.H


# Parties réelles et imaginaires :

# In[327]:


np.real(C) # same as: C.real


# In[328]:


np.imag(C) # same as: C.imag


# Argument et module :

# In[329]:


np.angle(C+1) 


# In[330]:


np.abs(C)


# #### Caclul matriciel

# #### Analyse de données
# 
# Numpy propose des fonctions pour calculer certaines statistiques des données stockées dans des *arrays* :

# In[331]:


data = np.vander([1, 2, 3, 4])
print( data)
print( data.shape)


# ##### mean

# In[332]:


# np.mean(data)
print( np.mean(data, axis=0))


# In[333]:


# la moyenne de la troisième colonne
np.mean(data[:,2])


# ##### variance et écart type

# In[334]:


np.var(data[:,2]), np.std(data[:,2])


# #### min et max

# In[335]:


data[:,2].min()


# In[336]:


data[:,2].max()


# In[337]:


data[:,2].sum()


# In[338]:


data[:,2].prod()


# ##### sum, prod, et trace

# In[339]:


d = np.arange(0, 10)
d


# In[340]:


# somme des éléments
np.sum(d)


# ou encore :

# In[341]:


d.sum()


# In[342]:


# produit des éléments
np.prod(d+1)


# In[343]:


# somme cumulée
np.cumsum(d)


# In[344]:


# produit cumulé
np.cumprod(d+1)


# In[345]:


# équivalent à diag(A).sum()
np.trace(data)


# #### Calculs avec parties d'*arrays*
# 
# en utilisant l'indexation ou n'importe quelle méthode d'extraction de donnés à partir des *arrays*

# In[346]:


data


# In[347]:


np.unique(data[:,1]) 


# In[348]:


mask = data[:,1] == 4


# In[349]:


np.mean(data[mask,3])


# #### Calculs avec données multi-dimensionnelles
# 
# Pour appliquer `min`, `max`, etc., par lignes ou colonnes :

# In[350]:


m = random.rand(3,4)
m


# In[351]:


# max global 
m.max()


# In[352]:


# max dans chaque colonne
m.max(axis=0)


# In[353]:


# max dans chaque ligne
m.max(axis=1)


# Plusieurs autres méthodes des classes `array` et `matrix` acceptent l'argument (optional) `axis` keyword argument.

# ### Génération de nombres aléatoires et statistiques

# Le module : [numpy.random](https://docs.scipy.org/doc/numpy/reference/routines.random.html) apporte à python la possibilité de générer un échantillon de taille  nn  directement, alors que le module natif de python ne produit des tirages que un par un. Le module [numpy.random](https://docs.scipy.org/doc/numpy/reference/routines.random.html) est donc bien plus efficace si on veut tirer des échantillon conséquents. Par ailleurs, [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) fournit des méthodes pour un très grand nombre de distributions et quelques fonctions classiques de statistiques.
# 

# Par exemple, on peut obtenir un array 4x3 de tirages gaussiens standard (soit en utilisant [randn](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html#numpy.random.randn) ou [normal](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html#numpy.random.normal) :

# In[354]:


np.random.randn(4,3)


# Pour se convaincre que *numpy.random* est plus efficace que le module random de base de python. On effectue un grand nombre de tirages gaussiens standard, en python pur et via numpy.

# In[355]:


N = int(1e4)
from random import normalvariate
get_ipython().run_line_magic('timeit', '[normalvariate(0,1) for _ in range(N)]')


# In[356]:


get_ipython().run_line_magic('timeit', 'np.random.randn(N)')


# ### Copy et "deep copy"
# 
# Pour des raisons de performance Python ne copie pas automatiquement les objets (par exemple passage par référence des paramètres de fonctions).

# In[357]:


A = np.array([[0,  2],[ 3,  4]])
A


# In[358]:


B = A


# In[359]:


# changer B affecte A
B[0,0] = 10
B


# In[360]:


A


# In[361]:


B = A
print( B is A)


# Pour éviter ce comportement, on peut demander une *copie profonde* (*deep copy*) de `A` dans `B`

# In[362]:


#B = np.copy(A)
B = A.copy()


# In[363]:


# maintenant en modifiant B, A n'est plus affecté
B[0,0] = -5

B


# In[364]:


A  # A est aussi modifié !


# In[365]:


print( A - A[:,0] ) # FAUX
print (A - A[:,0].reshape((2, 1)))  # OK


# ### Changement de forme et de taille, et concaténation des *arrays*
# 
# 

# In[366]:


A


# In[367]:


n, m = A.shape


# In[368]:


B = A.reshape((1,n*m))
B


# In[369]:


B[0,0:5] = 5 # modifier l'array

B


# In[370]:


A


# #### Attention !
# 
# La variable originale est aussi modifiée ! B n'est qu'une nouvelle *vue* de A.

# Pour transformer un *array* multi-dimmensionel en un vecteur. Mais cette fois-ci, une copie des données est créée :

# In[371]:


B = A.flatten()
B


# In[372]:


B[0:5] = 10
B


# In[373]:


A # A ne change pas car B est une copie de A


# #### Ajouter une nouvelle dimension avec `newaxis`
# 
# par exemple pour convertir un vecteur en une matrice ligne ou colonne :

# In[374]:


v = np.array([1,2,3])


# In[375]:


np.shape(v)


# In[376]:


# créer une matrice à une colonne à partir du vectuer v
v[:, np.newaxis]


# In[377]:


v[:,np.newaxis].shape


# In[378]:


# matrice à une ligne
v[np.newaxis,:].shape


# #### Concaténer, répéter des *arrays*
# 
# En utilisant les fonctions `repeat`, `tile`, `vstack`, `hstack`, et `concatenate`, on peut créer des vecteurs/matrices plus grandes à partir de vecteurs/matrices plus petites :
# 

# ##### repeat et tile

# In[379]:


a = np.array([[1, 2], [3, 4]])
a


# In[380]:


# répéter chaque élément 3 fois
np.repeat(a, 3) # résultat 1-d


# In[381]:


# on peut spécifier l'argument axis
np.repeat(a, 3, axis=1)


# Pour répéter la matrice, il faut utiliser `tile`

# In[382]:


# répéter la matrice 3 fois
np.tile(a, 3)


# ##### concatenate

# In[383]:


b = np.array([[5, 6]])


# In[384]:


np.concatenate((a, b), axis=0)


# In[385]:


np.concatenate((a, b.T), axis=1)


# ##### hstack et vstack

# In[386]:


np.vstack((a,b))


# In[387]:


np.hstack((a,b.T))


# ### Itérer sur les éléments d'un *array*
# 
#  * Dans la mesure du possible, il faut éviter l'itération sur les éléments d'un *array* : c'est beaucoup plus lent que les opérations vectorisées
#  * Mais il arrive que l'on n'ait pas le choix...

# In[388]:


v = np.array([1,2,3,4])

for element in v:
    print(element)


# In[389]:


M = np.array([[1,2], [3,4]])

for row in M:
    print ("row", row)
    
    for element in row:
        print( element)


# Pour obtenir les indices des éléments sur lesquels on itère (par exemple, pour pouvoir les modifier en même temps) on peut utiliser `enumerate` :

# In[390]:


for row_idx, row in enumerate(M):
    print ("row_idx", row_idx, "row", row)
    
    for col_idx, element in enumerate(row):
        print( "col_idx", col_idx, "element", element)
       
        # update the matrix M: square each element
        M[row_idx, col_idx] = element ** 2


# In[391]:


# chaque élément de M a maintenant été élevé au carré
M


# ### Utilisation d'*arrays* dans des conditions
# 
# Losqu'on s'intéresse à des conditions sur tout on une partie d'un *array*, on peut utiliser `any` ou `all` :

# In[392]:


M


# In[393]:


if (M > 5).any():
    print( "au moins un élément de M est plus grand que 5")
else:
    print ("aucun élément de M n'est plus grand que 5")


# In[394]:


if (M > 5).all():
    print ("tous les éléments de M sont plus grands que 5")
else:
    print( "tous les éléments de M sont plus petits que 5")


# ### *Type casting*
# 
# On peut créer une vue d'un autre type que l'original pour un *array*

# In[395]:


M =np.array([[-1,2], [0,4]])
M.dtype


# In[396]:


M2 = M.astype(float)
M2


# In[397]:


M2.dtype


# In[398]:


M3 = M.astype(bool)
M3


# ## Pour aller plus loin
# 
# * http://numpy.scipy.org
# * http://scipy.org/Tentative_NumPy_Tutorial
