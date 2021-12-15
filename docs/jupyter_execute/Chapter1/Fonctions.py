#!/usr/bin/env python
# coding: utf-8

# ## Fonctions

# In[1]:



# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# Une fonction en Python est définie à l'aide du mot-clé `def`, suivi d'un nom de fonction, d'un ensemble d'arguments en entrée (ou pas) entre parenthèses `()` et de deux points `:`.
# Le code suivant, avec un niveau d'indentation représente le corps de la fonction.

# In[2]:


def droite (x):
    print(2*x+1)


# In[3]:


droite (2)


# Optionnel, mais fortement recommandé, vous pouvez définir un "docstring", qui est une description des fonctions. Le docstring doit figurer directement après la définition de la fonction, avant le code correspondant au corps de la fonction.

# In[4]:


def droite (x):
    """ Ecrit le res. 2x+1 """
    print(2*x+1)        


# In[5]:


help(droite)


# In[6]:


droite.__doc__


# Les fonctions qui renvoient une valeur utilisent le mot-clé `return`

# In[7]:


def droite (x):
    """ Renvoie le res . 2x+1 """
    return 2*x+1


# In[8]:


droite(4)


# On peut retourner plusieurs valeurs en utilisant des virgules (un tuple
# est renvoyé)

# In[9]:


def puissance(x):
    """
    Retourne certaines puissance de x
    """
    return x ** 2, x ** 3, x ** 4


# In[10]:


puissance(2)


# In[11]:


x2, x3, x4 = puissance(2)

print(x4)


# ### Les arguments par défaut et mots clés

# On peut donner des valeurs par défaut aux arguments que la fonction prend en entrée :

# In[12]:


def mafonction(x, p=2, debug=False):
    if debug:
        print("Evaluer mafonction pour x = " + str(x) + " en utilisant un exposant p = " + str(p))
    return x**p


# Si nous ne fournissons pas une valeur de l'argument debug lors de l'appel de la fonction myfunc, elle prend par défaut la valeur fournie dans la définition de fonction :

# In[13]:


mafonction(4)


# In[14]:


mafonction(4, debug=True)


# Si vous énumérez explicitement le nom des arguments dans les appels de fonction, ils n'ont pas besoin d'être dans le même ordre que dans la définition de la fonction. C'est ce qu'on appelle les arguments *mot-clé*, et est souvent très utile dans les fonctions qui nécessitent beaucoup d'arguments facultatifs.

# In[15]:


mafonction(p=2, debug=True, x=14)


# ### Fonctions de manipulation de séquences
# 

#  ####  Fonction : filter

# Applique la fonction passée en premier argument sur chacun des
# éléments de la séquence passée en second argument et retourne une
# nouvelle liste qui contient tous les éléments de la séquence pour
# lesquels la fonction a retourné une valeur vrai.

# In[16]:


def funct1 (val ):
    return val > 0
list(filter( funct1 , [1, -2, 3, -4, 5]))


#  ####  Fonction : map

# applique la fonction passée en premier argument sur chacun des
# éléments de la ou des séquences passées en paramètre

# In[17]:


def somme (x,y):
    return x+y
L4= map(somme ,[1 , 2, 3], [4, 5, 6])
list(L4)


# **Remarque :** map peut être beaucoup plus rapide qu’une boucle for

#  ####  Fonction : [zip](https://docs.python.org/3/library/functions.html#zip)

# permet de parcourir plusieurs séquences en parallèle

# In[18]:


for (x, y) in zip ([1 , 2, 3] ,[4 , 5, 6]) :
    print(x, '+', y, '=', x + y)


# La fonction zip est très utile pour créer un dictionnaire. En effet, cela permet de raccourcir le code pour créer un dictionnaire à partir de clés et de valeurs séparés. Ca paraît bien plus long de créer les listes des clés et des valeurs. Et pourtant le code suivant peut être considérablement raccourci :

# In[19]:


hist = {'a': 1, 'b': 2, 'er': 1, 'gh': 2}
cles = []
vals = []
for k, v in hist.items():
    cles.append(k)
    vals.append(v)
cles, vals


# Cela devient :

# In[20]:


hist = {'a': 1, 'b': 2, 'er': 1, 'gh': 2}
cles, vals = zip(*hist.items())
cles, vals


# Petite différence, `cles`, `vals` sont sous forme de [tuple](https://docs.python.org/3.5/library/stdtypes.html?highlight=tuple#tuple) mais cela reste très élégant.

#  ####  Fonction : reduce

# Réduit une séquence par l’application récursive d’une fonction sur chacun de ses éléments.
#     - La fonction passée comme premier paramètre doit prendre deux arguments
#     - La fonction reduce prend un troisième paramètre optionnel qui est la valeur initiale du calcul
#     - Importer la fonction reduce à partir du module functools : from functools import reduce

# In[21]:


from functools import reduce
reduce(somme , [1, 2, 3, 4, 5])


# #### Fonctions sans nom (mot-clé lambda)

# Sous Python, vous pouvez également créer des fonctions sans nom, en utilisant le mot-clé `lambda` :

# In[22]:


f1 = lambda x: x**2
    
# est équivalent à 

def f2(x):
    return x**2


# In[23]:


f1(2), f2(2)


# Les lambda expressions permettent une syntaxe plus légère pour déclarer une fonction simple

# Cette technique est utile par exemple lorsque nous voulons passer une fonction simple comme argument à une autre fonction, comme ceci :

# In[24]:


# map est une fonction intégrée de python
map(lambda x: x**2, range(-3,4))


# In[25]:


# Dans python 3 nous pouvons utiliser `list (...)` pour convertir l'itérateur en une liste explicite
list(map(lambda x: x**2, range(-3,4)))


# ### Bon à connaître

# #### **kwargs : Passer plusieurs arguments à une fonction en Python
# 
# Il peut arriver que vous ne connaissiez pas les arguments que vous passerez à une fonction. Dans ce cas, utilisez `**kwargs`.
# 
# Les `**kwargs` vous permettent de passer plusieurs arguments à une fonction en utilisant un dictionnaire. Dans l'exemple ci-dessous, passer `**{'a':1, 'b':2}` à la fonction revient à passer `a=1, b=1` à la fonction.
# 
# Une fois l'argument `**kwargs` passé, vous pouvez le traiter comme un dictionnaire Python.
# 

# In[26]:


parameters = {'a': 1, 'b': 2}

def example(c, **kwargs):
    print(kwargs)
    for val in kwargs.values():
        print(c + val)

example(3, **parameters)


# #### Décorateur en Python
# 
# Si vous voulez ajouter le même bloc de code à différentes fonctions en Python, le mieux c'est d'utiliser le décorateur.
# 
# Dans le code ci-dessous, on crée un décorateur pour suivre le temps de la fonction `Salut`.

# In[27]:


import time 

def time_func(func):
    def wrapper():
        print("Cela se passe avant que la fonction soit appelée.")
        start = time.time()
        func()
        print('Cela se produit après que la fonction soit appelée.')
        end = time.time()
        print('La durée est de', end - start, 's')

    return wrapper


# Maintenant, tout ce qu'il reste à faire est d'ajouter `@time_func` avant la fonction `Salut`.

# In[28]:


@time_func
def Salut():
    print("hello")

Salut()


# Le décorateur rend le code propre et raccourcit le code répétitif. Si on veux regarder le temps d'une autre fonction, par exemple, func2(), on peut simplement utiliser :

# In[29]:


@time_func
def func2():
    pass
func2()

