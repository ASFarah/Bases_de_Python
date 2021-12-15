#!/usr/bin/env python
# coding: utf-8

# ## Fichiers

# In[1]:


# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# L’écriture et la lecture dans un fichier s’effectuent toujours de la même manière. On ouvre le fichier
# en mode écriture ou lecture, on écrit ou on lit, puis on ferme le fichier, le laissant disponible pour une
# utilisation ultérieure. Ce paragraphe ne présente pas l’écriture ou la lecture dans un format binaire
# car celle-ci est peu utilisée dans ce langage.

# Dans cette partie, on prend l'exemple d'un fichier *SerieTV.txt* ayant le contenu suivant :
# 
# walking dead
# 
# Black Mirror 
# 
# Narcos
# 
# Game of Thrones

# ### Lecture dans un fichier texte

# La lecture dans un fichier texte s’effectue selon le même schéma :

# In[2]:


f = open('files/SerieTV.txt','r') #f = open(' nom_du_fichier.txt ','r')
lignes =f.readlines ()
f.close ()


# * Ouverture du fichier : open(nom,mode)
#     - nom : chaîne de caractère, nom du fichier
#     - mode : chaîne de caractères, accès au fichier
#         (’r’ : read, ’w’ : write, ’a’ : append)
# * Lecture ligne par ligne ; La ligne est affectée à une variable texte
# * Principales méthodes :
#     - read() Lit tout le fichier (jusqu’à EOF) et renvoie un str
#     - read(n) Lit n caractères du fichier à partir de la position courante
#     - readline() Lit une ligne du fichier jusqu’à nn et renvoie la chaîne
#     - readlines() Lit toutes les lignes du fichier, renvoie un objet list
# * Fermeture du fichier

# #### Méthodes seek() et tell()**

# Les méthodes **seek()** et **tell()** permettent respectivement de se déplacer au n ième caractère (plus exactement au n ième octet) d’un fichier et d’afficher où en est la lecture du fichier,
# c’est-à-dire quel caractère (ou octet) est en train d’être lu.

# In[3]:


f1 = open('files/SerieTV.txt', 'r')
f1.readline()
'walking dead\n'
f1.tell()
14
f1.seek(0)
f1.tell()
0
f1.readline()
'walking dead\n'
f1.readline()
'Black Mirror \n'
f1.close()


# On remarque qu'à l’ouverture d’un fichier, le tout premier caractère est indexé par 0 (tout comme le premier élément d’une liste). La méthode seek() permet facilement
# de remonter au début du fichier lorsque l’on est arrivé à la fin ou lorsqu’on en a lu une partie.

# #### Itérations directement sur le fichier**

# Il existe également un moyen à la fois simple et élégant
# de parcourir un fichier.

# In[4]:


f1 = open('files/SerieTV.txt', 'r')
for ligne in f1:
    print(ligne)
    

f1.close()


# La boucle **for** va demander à Python d’aller lire le fichier ligne par ligne. 

# ### Ecriture dans un fichier texte

# La syntaxe d'écriture dans un fichier est la suivante :

# In[5]:


f = open ("nom-fichier.txt", "w") # ouverture en mode écriture "w" ou écriture ajout "a"
s = " Bonjour"
s2 = "Comment tu vas ?"
f.write ( s ) # écriture de la chaîne de caractères s
f.write ( s2 ) # écriture de la chaîne de caractères s2
#...
f.close () # fermeture


# Certains codes sont très utiles lors de l’écriture de fichiers texte comme :
# 
# -- **\n** : passage à la ligne
# 
# -- **\t** : insertion d’une tabulation, indique un passage à la colonne suivante dans le logiciel Excel
# 

# ### Lecture et ecriture d'un fichier CSV

# In[6]:


import csv
def read_csv_file ( filename ):
    """ Lire un fichier CSV et ecrire chaque ligne sous
    forme de liste """
    f = open( filename)
    for row in csv.reader (f):
        print( row )
        f. close()


# In[7]:


import csv
def read_csv_file1 ( filename ):
    """ Lire un fichier CSV et ajouter les elements a la liste . """
    f = open( filename )
    data = []
    for row in csv.reader (f):
        data.append ( row )
        print(data)
        f.close()


# In[8]:


def write_csv(filename):
    import csv
    L = [['Date', 'Nom', 'Notes'],
         ['2016/1/18', 'Martin Luther King Day', 'Federal Holiday'],
         ['2016/2/2','Groundhog Day', 'Observance'],
         ['2016/2/8','Chinese New Year', 'Observance'],
         ['2016/2/14','Valentine\'s Day', 'Obervance'],
         ['2016/5/8','Mother\'s Day', 'Observance'],
         ['2016/8/19','Statehood Day', 'Hawaii Holiday'],
         ['2016/10/28','Nevada Day', 'Nevada Holiday']]
    f = open(filename, 'w', newline='')
    for item in L:
        csv.writer(f).writerow(item)
    f.close()


# ### Méthode optimisée d’ouverture et de fermeture de fichier

# Depuis la version 2.5, Python introduit le mot-clé **with** qui permet d’ouvrir et fermer un
# fichier de manière commode. Si pour une raison ou une autre l’ouverture conduit à une erreur (problème de droits, etc), l’utilisation de **with** garantit la bonne fermeture du fichier (ce
# qui n’est pas le cas avec l’utilisation de la méthode open() invoquée telle quelle). Voici un
# exemple :

# In[9]:


with open('files/SerieTV.txt', 'r') as f1:
    for ligne in f1:
        print(ligne)


# Vous remarquez que **with** introduit un bloc d’indentation. C’est à l’intérieur de ce bloc que
# nous effectuons toutes les opérations sur le fichier. Une fois sorti, Python fermera automatiquement le fichier. Vous n’avez donc plus besoin d’invoquer la fonction close().
