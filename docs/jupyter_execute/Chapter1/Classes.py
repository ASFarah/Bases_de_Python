#!/usr/bin/env python
# coding: utf-8

# ## Classes
# 

# Les classes sont les éléments centraux de la programmation orientée objet. Une classe est une structure qui sert à représenter un objet et les opérations qui peuvent être effectuées sur l'objet.
# 
# En Python, une classe peut contenir des *attributs* (variables) et des *méthodes* (fonctions).
# 
# Une classe est définie de manière analogue aux fonctions, mais en utilisant le mot-clé `class`. La définition d'une classe contient généralement un certain nombre de méthodes de classe (des fonctions dans la classe).
# 
# * Le premier argument d'un méthode doit être `self`: argument obligatoire. Cet objet self est une auto-référence.
# 
# * Certains noms de méthode de classe ont un sens particulier, par exemple:
# 
#      * `__init__`: nom de la méthode invoquée à de la création de l'objet.
#      * `__str__`: méthode invoquée lorsqu'une représentation de la classe sous forme de chaîne de caractères est demandée, par exemple quand la classe est passée à la fonction print
#      * Voir http://docs.python.org/2/reference/datamodel.html#special-method-names pour les autres noms de méthode

# In[1]:


class Point:
    """
    Classe simple pour représenter un point dans un système de coordonnées cartésiennes.
    """
    
    def __init__(self, x, y):
        """
        Créer un nouveau point à x, y.
        """
        self.x = x
        self.y = y
        
    def translate(self, dx, dy):
        """
        Calcul de la déviation de direction du point par dx et dy
        """
        self.x += dx
        self.y += dy
        
    def __str__(self):
        return("Point de coordonnées [%f, %f]" % (self.x, self.y))


# Pour créer une nouvelle instance d'une classe:

# In[2]:


p1 = Point (0, 0) # ceci va appeler la méthode __init__ dans la classe Point

print (p1) # ceci va appeler la méthode __str__


# Pour faire appel à une méthode de classe dans l'instance de classe p :

# In[3]:


p2 = Point(1, 1)

p1.translate(0.25, 1.5)

print(p1)
print(p2)


# ### Exceptions

# * Dans Python les erreurs sont gérées à travers des "Exceptions"
# * Une erreur provoque une Exception qui interrompt l'exécution normale du programme
# * L'exécution peut éventuellement reprendre à l'intérieur d'un bloc de code try - except

# * Une utilisation typique: arrêter l'exécution d'une fonction en cas d'erreur:

# def my_function(arguments) :
# 
# if not verify(arguments):
#     raise Expection("Invalid arguments")
#     
# et on continue

# On utilise try et  expect pour maîtriser les erreurs :
# try:
#     # normal code goes here
# except:
#     # code for error handling goes here
#     # this code is not executed unless the code
#     # above generated an error

# Par exemple :

# In[4]:


try:
    print("test_var")
    # genere une erreur: la variable test n'est pas définie
    print(test_var)
except:
    print("Caught an expection")


# Pour obtenir de l'information sur l'erreur : accéder à l'instance de la classe Exception concernée:
# 
# `except Exception as e :`

# In[5]:


try:
    print("test")
    # generate an error: the variable test is not defined
    print(test)
except Exception as e:
    print("Caught an expection:", e)

