#!/usr/bin/env python
# coding: utf-8

# ## Les chaînes de caractères (String : str)

# In[1]:


# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# Les chaînes sont le type de variable utilisé pour stocker des messages texte.

# ### Syntaxe
# **Trois syntaxes :**
# 
#       1- simples quotes : 'Bonjour, dit-elle'
# 
#       2- doubles quotes : "Que se passe t'il ?"
# 
#       3- triple quotes (simples ou doubles) : '''chaînes multilignes'''
# 
# 
# 
# Passage à la ligne : Ligne 1 \n Ligne 2

# In[2]:


s = "Bonjour"
type(s)


# ### Caractère d'échappement
# 
# Le symbole \ permet :
# 
#     * \n : un saut de ligne
#     * \t : une tabulation
#     * \' : le « ' », permet de ne pas fermer la chaine de caractères, eg., 'aujourd\'hui'
#     * \" : le « " », permet de ne pas fermer la chaine de caractères, eg. " Bonjour \"Pierre\" "
#     * \\ est un « \ »

# ### Opérations sur les caractères

# Comme n’importe quelle séquence, les chaînes de caractères supportent :
# 
#     - le test d’appartenance, la concaténation, la répétition,
#     - la taille, le plus petit/plus grand élément,..

# | Operation           | Rôle                                                                  |
# | :- | :-: |
# |ch1+ch2	          |Concatène (colle l'une à la suite de l'autre) les chaines ch1 et ch2.  |
# |ch1 * n ou n *c h1   | 	Concatène n fois la chaine ch1.                                   |
# 

# In[3]:


# concaténation
phrase = 'Bienvenue '+ 'à l\'ENSAI'   

# Longueur de la chaîne: le nombre de caractères
len(phrase)    


# In[4]:


# test d’appartenance à la chaîne
'u' in phrase    


# In[5]:


# Determine la position du mot "ENSAI" dans la chaîne de caractères "phrase" 
phrase.index('ENSAI')


# En tant que séquence,
# 
# * on peut accéder aux éléments de la chaîne par leur index en utilisant **[index]**. *Note :* Les index sont utilisés de 0 à (n-1)

# In[6]:


s[0]


# * Il est possible d'indéxer en partant de la fin avec des indices négatifs :

# In[7]:


s[-1]


# * On peut extraire une partie d'une chaîne en utilisant la syntaxe **[start: stop]**, qui extrait les caractères entre index start et stop -1 (le caractère à l'index stop n'est pas inclus):

# In[8]:


s[0:12]


# * Si nous omettons l'index start ou stop  (ou les deux) de [start: stop], la valeur par défaut est le début et la fin de la chaîne, respectivement :

# In[9]:


s[:4]


# In[10]:


s[14:]


# In[11]:


s[:]


# * Nous pouvons également définir "le pas (step)" en utilisant la syntaxe [start: end: step] (la valeur par défaut pour step est 1, comme nous l'avons vu ci-dessus) :

# In[12]:


s[::3]


# Cette technique est appelée Slicing. Pour en savoir plus sur la syntaxe, cliquez ici: 'http://docs.python.org/release/2.7.3/library/functions.html?highlight=slice#slice'
# 
# Python possède un ensemble très riche de fonctions pour le traitement de texte. Voir par exemple 'http://docs.python.org/2/library/string.html' pour plus d'informations.
# 
# Parmi les fonctions les plus souvent utilisées sur les chaînes on cite :
# 
# | Méthode                | Rôle                         |
# |------------------------|------------------------------|
# | strip, lstrip, rstrip  | Élimine les espaces          |
# | split                  | Découpe (en une liste)       |
# | join                   | Recolle une liste de chaînes |
# | find                   | Recherche une sous-chaîne    |
# | replace                | Remplace une sous-chaîne     |
# | upper, lower           | Renvoie la chaine en majuscules, miniscules            |
# | capitalize             | Renvoie la chaine avec la première lettre en majuscule |
# | count                  | Renvoie le nombre d'occurence |
# 

# ### Méthodes sur les caractères

# #### find: Trouver l'index d'une sous-chaîne dans une chaîne de caractères
# 
# Si vous voulez trouver l'indice d'une sous-chaîne dans une chaîne de caractères, utilisez la méthode `find()`. 
# Cette méthode renvoie l'indice de la première occurrence de la sous-chaîne si elle est trouvée et `-1` sinon.

# In[13]:


phrase = "Bonjour je m'appelle Pierre"

# Trouver l'index de la première occurrence de la sous-chaîne.
phrase.find("Pierre")


# In[14]:


phrase.find("Marie")


# Vous pouvez également indiquer la position de départ et de fin de la recherche :
# 

# In[15]:


# Commence la cherche de la sous-chaîne à partir de l'index 7
phrase.find("Pierre", 7)


# In[16]:


# Commence la cherche de la sous-chaîne à partir de l'index 7 et prend fin à l'index 18
phrase.find("P", 5, 10)


# #### re.sub : Remplacer une chaîne par une autre chaîne à l'aide d'une expression régulière
# 
# Si vous souhaitez remplacer une chaîne de caractères par une autre ou modifier l'ordre des caractères dans une chaîne, utilisez `re.sub`.
# 
# `re.sub` vous permet d'utiliser une expression régulière pour spécifier le motif de la chaîne de caractères que vous souhaitez remplacer.
# 
# Dans le code ci-dessous, on remplace **12/12/2018** par `mardi` et on remplace **12/12/2018** par `2018/12/12`.

# In[17]:


import re

texte = "Cours d'anglais le 12/12/2018"
concordance = r"(\d+)/(\d+)/(\d+)"

re.sub(concordance, "mardi", texte)


# In[18]:


re.sub(concordance, r"\3/\1/\2", texte)


# #### difflib.SequenceMatcher : Détecter les articles "presque similaires
# 
# Lors de l'analyse d'articles, différents articles peuvent être presque similaires mais pas identiques à 100 %, peut-être à cause de la grammaire ou du changement de deux ou trois mots (comme le postage croisé). 
# 
# Comment détecter les articles "presque similaires" et en éliminer un ? C'est là que `difflib.SequenceMatcher` peut s'avérer utile.

# In[19]:


from difflib import SequenceMatcher

text1 = 'Je rentre du travail'
text2 = 'Je rentre du boulot'
print(SequenceMatcher(a=text1, b=text2).ratio())


# #### difflib.get_close_matches : Obtenir une liste des meilleures correspondances pour un certain mot
# 
# Si vous voulez obtenir une liste des meilleures correspondances pour un certain mot, utilisez difflib.get_close_matches.

# In[20]:


from difflib import get_close_matches

Fruits = ['Pomme', 'Poire','Peche','Pruneau','Fraise']
get_close_matches('Prune', Fruits)


# Pour obtenir des correspondances plus proches, augmentez la valeur de l'argument cutoff (par défaut 0.6).

# In[21]:


get_close_matches('Prune', Fruits, cutoff=0.8)

