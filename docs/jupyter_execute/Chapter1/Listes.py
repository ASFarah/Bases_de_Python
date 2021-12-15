#!/usr/bin/env python
# coding: utf-8

# ## Les listes

# In[1]:


# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# Les listes sont très semblables aux chaînes, sauf que chaque élément peut être de n'importe quel type.
# La syntaxe pour créer des listes en Python est [...]:
# 
#     - Collection ordonnée (de gauche à droite) d’éléments (combinaison de types de données de base,  données hétérogènes)
#     - De taille quelconque, peut grandir, rétrécir, être modifiée, être encapsulée (liste de listes de listes)

# In[2]:


l = [1,2,3,4]

print(type(l))
print(l)


# Les éléments d'une liste peuvent ne pas être du même type :

# In[3]:


l = [1, 'a', 1.0, 1-1j]

print(l)


# Les listes Python peuvent être non-homogènes et arbitrairement imbriquées:

# In[4]:


l1 = [1, [2, [3, [4, [5]]]]]

l1


# ### Slicing
# Les mêmes techniques de slicing utilisées précédement sur les chaînes de caractères peuvent également être utilisées pour manipuler les listes.

# In[5]:


L=[ '14 ' ,'14231363 ' ,'14232541 ', 'MC1R ']
L[3]


# In[6]:


L[0:2]


# In[7]:


print(L[::3])


# ### Manipulation de listes
# On peut modifier les listes en attribuant de nouvelles valeurs aux éléments de la liste. Dans le jargon technique, on dit que les listes sont mutables.

# In[8]:


L [3]= ' ENSOARG00000002239 ' # une liste est modifiable
L


# In[9]:


L [1]= int(L[1]) ; L[2]= int(L[2]) # conversion string -> integer
L # les elements sont de types differents


# Il existe un certain nombre de fonctions pratiques pour générer des listes de différents types. Exemple :

# In[10]:


# Sous Python 3.x, on peut générer une liste en utilisant l'instruction suivante : List (range (start, stop, step)

list(range(1, 20, 6))   


# In[11]:


# Convertir une chaîne en une liste  :
s = "Bonjour"
s2=list(s)
s2


# ### Fonctions/Opérations sur les listes

# * ajoute un élément à la fin de la liste  avec `append`

# In[12]:


List =[3 , 2, 4, 1]
List.append(15)   # ajoute un élément à la fin de la liste  avec append
print(List)


# * étendre une liste avec `extend`

# In[13]:


List.extend ([7 , 8, 9])  # étendre la liste  avec extend
print(List)


# * Insérer un élément à une position spécifique à l'aide de `insert`

# In[14]:


List.insert(0, 111)
print(List)


# * Supprimer le premier élément correspondant à la valeur donnée à l'aide de `remove`

# In[15]:


List.remove(15)
print(List)


# * Supprimer un élément à un emplacement donné à l'aide de `del`

# In[16]:


del List [3] # supprimer l'élément se trouvant à l'index 3 de la liste
print(List)


# * Tri d'une liste avec `sort`

# In[17]:


List.sort () # Tri de liste
List


# * Renoyer le nombre d'occurence d'une valeur dans la liste avec `count`

# In[18]:


List.count (3)   # renvoie le nombre d'occurence de 3 dans la liste


# Voir *help(list)* pour plus de détails, ou lire la documentation en ligne.

# ### Bonnes pratiques

# #### Arrêter d'untiliser = operator pour faire une copie de liste Python. Utilisze plutôt la méthode "copy" 

# Lorsque vous créez une copie d'une liste Python à l'aide de l'opérateur =, une modification de la nouvelle liste entraînera celle de l'ancienne. C'est parce que les deux listes pointent vers le même objet.

# In[19]:


liste_1 = [1, 2, 3]
liste_2 = liste_1 
liste_2.append(4)


# In[20]:


liste_2


# In[21]:


liste_1


# Au lieu d'utiliser l'opérateur **=**, utilisez la méthode `copy()`. Maintenant, votre ancienne liste ne changera pas lorsque vous modifierez votre nouvelle liste.

# In[22]:


liste_1 = [1, 2, 3]
liste_2 = liste_1.copy()
liste_2.append(4)


# In[23]:


liste_2


# In[24]:


liste_1


# #### Enumerate : Obtenir le compteur et la valeur en bouclant

# Généralement pour accéder à la fois à l'index et à la valeur d'un tableau (liste), on a souvent tendance à utiliser : `for i in range(len(tableau))`.
# 
#  Si c'est le cas, utilisez plutôt `enumerate`. Le résultat est le même, mais il est beaucoup plus propre.

# In[25]:


Tableau = ['a', 'b', 'c', 'd', 'e']

# Au lieu de
for i in range(len(Tableau)):
    print(i, Tableau[i])


# In[26]:


# Utilisez ça
for i, val in enumerate(Tableau):
    print(i, val)


# #### Différence entre "append" et "extend"
# 
# Pour ajouter une liste à une autre liste, utilisez la méthode `append`. Pour ajouter des éléments d'une liste à une autre liste, utilisez la méthode `extend`.

# In[27]:


# Ajouter une liste à une liste
a = [1, 2, 3, 4]
a.append([5, 6])
a


# In[28]:


# Ajouter des éléments à une liste
a = [1, 2, 3, 4]
a.extend([5, 6])

a

