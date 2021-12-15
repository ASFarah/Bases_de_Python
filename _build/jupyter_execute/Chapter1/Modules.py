#!/usr/bin/env python
# coding: utf-8

# ## Modules

# In[1]:



# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# * Un module contient un ensemble de fonctions et commandes
# * Python dispose d’une bibliothèque de base quand il est initialisé. Et selon nos besoins ces bibliothèques vont être chargées.
# * Pour utiliser un module, il faut l’importer.
#  * Nous avons deux types de modules : ceux disponibles sur Internet (programmés par d’autres) et ceux que l’on programme soi-même.
# * Pour les modules disponibles, les bibliothèques souvent utiles pour faire un programme python scientifique, nous avons :
#     import os
#     import sys
#     import numpy as np
#     import math               
#     import random
#     import csv
#     import scipy
#     import matplotlib . pylab as plt

# ### Syntaxe d'importation

# Syntaxe 1 : importer le module sous son nom

# In[2]:


import math
# on peut utiliser math.sin, math.sqrt...


# Syntaxe 2 : importer le module sous un nom différent
#      - permet d’abréger le nom des modules

# In[3]:


import math as m

# on utilise m.sin, m.sqrt...


# Syntaxe 3 : importer seulement certaines définitions

# In[4]:


from math import sqrt
# on peut utiliser uniquement sqrt (les autres fonctions math.sin..., ne sont pas reconnu)


# * On peut utiliser **help** pour obtenir de l'aide sur les modules importés.

# In[5]:


help(math)


# ou, si on veut connaître en seul coup d’oeil toutes les méthodes ou variables associées à
# un module (ou objet), on peut utiliser la commande **dir**

# In[6]:


print(dir(math))


# #### Modules courants

# Il existe une série de modules que vous serez probablement amenés à utiliser si vous programmez en Python. En voici une liste non exhaustive. Pour la liste complète, reportez-vous à[la page des modules](http://www.python.org/doc/current/modindex.html) sur [le site de Python](http://www.python.org/) :
# 
#   *[math](http://www.python.org/doc/current/library/math.html) : fonctions et constantes mathématiques de base (sin, cos, exp, pi...).
#   
# *[sys](http://www.python.org/doc/current/library/sys.html) : passage d'arguments, gestion de l'entrée/sortie standard...
#     
# *[os](http://www.python.org/doc/current/library/os.html) : dialogue avec le système d'exploitation (e.g. permet de sortir de Python, lancer une commande en {\it shell}, puis de revenir à Python).
#     
# *[random](http://www.python.org/doc/current/library/random.html) : génération de nombres aléatoires.
#     
# *[time](http://www.python.org/doc/current/library/time.html) : permet d'accéder à l'heure de l'ordinateur et aux fonctions gérant le temps.
# 
# *[calendar](http://www.python.org/doc/current/library/calendar.html) : fonctions de calendrier.
# 
# *[profile](http://www.python.org/doc/current/library/profile.html) : permet d'évaluer le temps d'exécution de chaque fonction dans un programme ({\it profiling} en anglais).
# 
# *[urllib2](http://www.python.org/doc/current/library/urllib2.html) : permet de récupérer des données sur internet depuis python.
# 
# *[Tkinter](http://www.python.org/doc/current/library/tkinter.html) : interface python avec Tk (permet de créer des objets graphiques; nécessite d'installer [Tk](http://www.tcl.tk/software/tcltk/index.html).
# 
# *[re](http://www.python.org/doc/current/library/re.html) : gestion des expressions régulières.
# 
# 
# *Je vous conseille vivement d'aller surfer sur les pages de ces modules pour découvrir toutes leurs potentialités.*
# 

# ###  Création de vos propres modules

# * Vous pouvez également définir vos propres modules.
# 
# Considérez l'exemple suivant: le fichier mymodule.py contient des exemples simples d'implémentation d'une variable, d'une fonction et d'une classe :

# In[7]:


get_ipython().run_cell_magic('file', 'monmodule.py', '"""\nExemple de module python. Contient une variable appelée ma_variable,\nUne fonction appelée ma_fonction, et une classe appelée MaClasse.\n"""\n\nma_variable = 0\n\ndef ma_fonction():\n    """\n    Exemple de fonction\n    """\n    return ma_variable*2\n    \nclass MaClasse:\n    """\n    Exemple de classe.\n    """\n\n    def __init__(self):\n        self.variable = ma_variable\n        \n    def set_variable(self, n_val):\n        """\n        Définir self.variable à n_val\n        """\n        self.variable = n_val\n        \n    def get_variable(self):\n        return self.variable')


# On peut importer le module monmodule dans notre programme Python en utilisant import :

# In[8]:


import monmodule


# In[9]:


monmodule.ma_variable


# #### La bibliothèque standard et ses modules

# Une bibliothèque standard Python (Python Standard Library) est une collection de modules qui donne accès à des fonctionnalités de bases : appels au système d'exploitation, gestion des fichiers, gestion des chaînes de caractères, interface réseau, etc.

# #### Références
# 
# * The Python Language Reference: http://docs.python.org/2/reference/index.html
# * The Python Standard Library: http://docs.python.org/2/library/ (Pour une liste complète des modules python)
