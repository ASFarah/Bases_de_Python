#!/usr/bin/env python
# coding: utf-8

# ## Syntaxe de base
# 
# Cette section résume en quelques lignes les éléments essentiels et la syntaxe du langage python.

# In[1]:



# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# Connaitre la version installée :

# In[2]:


import sys
print (sys.version)


# Avec la version 3.x, le langage a introduit quelques changements importants qui seront précisés. Il est préférable de choisir la version 3.5 plutôt que 2.7. Outre le fait qu'elle contient les dernières évolutions, elle est beaucoup plus cohérente en ce qui concerne les chaînes de caractères.

# Quelques précisions sur le langage :
# 
# - **Commentaires :** Les commentaires dans un programme commencent par le symbole **#** et vont jusqu’à la fin de la ligne.
# 
# - Généralement une instruction par ligne, sans marqueur à la fin. Si plusieurs instructions par ligne, les séparer par **;**
# 
# - Contraintes de nommage : Les noms de variable (fonction, classe...) doivent respecter des règles syntaxiques : ils peuvent contenir des lettres, chiffres, des underscore (_) mais doivent commencer par une lettre
# 
# - L’indentation est primordiale.
# 
# - On commence à compter à 0.
#     
# - L’instruction **print** permet d’afficher n’importe quelle information. **print** est une fonction, tout ce qui doit être affiché doit l’être entre parenthèses.
# 
# - L’instruction **help** affiche l’aide associée à une variable, une fonction, une classe, une méthode, un module. Pour une fonction, une classe, une méthode du programme, cette aide correspond à une chaîne de caractères encadrée par trois ". Ce message d’aide peut s’étaler sur plusieurs lignes.

# ### Valeurs, Variables et Affectations

# #### Les variables

# Une variable permet de stocker des données pour les réutiliser plus tard.
# 
# a=< valeur >
# 
# Le type de **<valeur>** détermine le type de la variable **a**. Si une variable porte déjà le même nom, son contenu est écrasé (perdu aussi).

# In[3]:


# Affectation d'une valeur à une variable
a = 1
a


# Une affectation crée une liaison entre un nom et une donnée.

# ##### Contraintes de nommage
# 
# Les noms de variable (de fonction, de classe...) doivent respecter des règles syntaxiques :
#     - peuvent contenir des lettres, chiffres, des underscore (_) mais doivent commencer par une lettre
#     - Par convention les noms de variables sont en minuscule, et les noms de classe commencent par une majuscule.
#     - la casse est importante (ma_variable ≠ Ma_VaRiAbLE)
#     - certains noms *mots-clés* sont réservés par le langage. Ces mots-clés sont :
# 
#     and, as, assert, break, class, continue, def, del, elif, else, except, 
#     exec, finally, for, from, global, if, import, in, is, lambda, not, or,
#     pass, print, raise, return, try, while, with, yield
# 
# Note: faites attention au mot-clé **lambda**, qui pourrait facilement être une variable dans un programme scientifique. Mais étant un mot-clé, il ne peut pas être utilisé comme un nom de variable.

# #### Affectations
# 

# ##### Affectation simple
# 
# 
# L'opérateur d'affectation en Python est effectuée par **=**. Python est un langage de typage dynamique, donc vous n'avez pas besoin de spécifier le type d'une variable lors de sa création.
# 
# L'affectation d'une valeur à une nouvelle variable crée la variable.
# De manière générale nom_variable = valeur.

# In[4]:


# Affectations simples
a = 2
b = 3
a, b


# ##### Expressions
# 
# Une expression combine des variables et des littéraux par l’intermédiaire d’opérateurs et de fonctions.
#     
# Python évalue les expressions : il applique les opérateurs et les fonctions afin de déterminer leur valeur résultat.

# In[5]:


(1, 2) + (3, 4)


# In[6]:


c=5 ; max(a, b) + 5*c


# #### Types de données simples

# Chaque donnée en Python est un objet dont :
#     - le type caractérise la nature de l’objet, ses opérations, cf. `type()`, `dir()`
#     - l’identité caractérise l’objet (e.g. une adresse mémoire), `id()`
#     - la valeur est le contenu des données

# In[7]:


a="Bonjour"
id(a)


# In[8]:


type(a)


# In[9]:


dir(a)


# In[10]:


help(a.upper)


# ##### Types fondamentaux

# In[11]:


# entier (integers)
a = 1
type(a)


# In[12]:


# réel (float)
a = 1.0
type(a)


# In[13]:


# Booléen (boolean)
b1 = True
b2 = False
type(b2)


# In[14]:


# Nombres complexes: notez que l'utilisation de `j` permet de spécifier la partie imaginaire 
a = 1.0 - 1.0j
type(a)


# In[15]:


print(a)
print(a.real, a.imag)


# Vous pouvez également tester si les variables sont de certains types :

# In[16]:


type(a) is float


# ##### Typage dynamique
# 
# Une variable possède un type associé, bien qu'il ne soit pas explicitement spécifiée. Le type est dérivé de la valeur qui lui a été attribuée.
# L’instruction type(a) retourne le type de la variable a.

# In[17]:


a = 4
type(a)


# Si vous attribuez une nouvelle valeur à une variable, son type peut changer.

# In[18]:


a = (3, 8)
type(a)


# ##### Typage fort

# In[19]:


a=12.5
3+a


# In[20]:


"La valeur de a = "+a


# Le **typage fort** signifie que les conversions implicites de types sont formellement interdites.
# 
# Les **seules conversions implicites** de types sont entre types numériques : **int → float → complex**.
# 
# Pour toutes les autres conversions, il faut utiliser explicitement des fonctions de conversion.

# In[49]:


"La valeur de de = "+str(a)


# ### Opérateurs et opérateurs de comparaisons
# 
# La plupart des opérateurs et des opérateurs de comparaisons en Python fonctionnent comme on peut s'y attendre:
# 
#     - Opérateurs arithmétiques +, -, *, /, // (division entière), '**' puissance

# In[50]:


1 + 2, 1 - 2, 1 * 2, 1 / 2


# In[51]:


1.0 + 2.0, 1.0 - 2.0, 1.0 * 2.0, 1.0 / 2.0


# In[52]:


# Division entière des nombres réels
3.0 // 2.0


# **Remarque :** L'opérateur **/** effectue toujours une division en virgule flottante dans Python 3.x. Cela n'est pas vrai dans Python 2.x, où le résultat de / est toujours un entier si les opérandes sont des entiers. Pour être plus précis, 1/2 = 0.5 (float) dans Python 3.x et 1/2 = 0 (int) dans Python 2.x (mais 1.0 / 2 = 0.5 dans Python 2.x).

#     - Les opérateurs booléens sont : and, not, or.

# In[53]:


True and False, not True, True or False


#     - Les opérateurs de comparaison : >, <, >= (supérieur ou égal), <= (inférieur ou égal), == égalité, is identiques.

# In[54]:


2 > 1, 3<=6


# In[55]:


# égalité
[1,2] == [1,2]


# In[56]:


# Objets identiques?
l1 = l2 = [1,2]

l1 is l2

