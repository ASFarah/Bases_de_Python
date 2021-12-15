#!/usr/bin/env python
# coding: utf-8

# ## Les dictionnaires

# Les dictionnaires sont également des listes, sauf que chaque élément est une paire clé-valeur.
#     - La syntaxe des dictionnaires est {key1: value1, ...}:
#     - chaque clé est unique
#     - mutables

# In[1]:


mois ={ 'Jan ':31 , 'Fev ':28 , 'Mar ':31}
print(type(mois))
print(mois)


# In[2]:


mois.keys()


# In[3]:


mois.values()


# In[4]:


mois.items()


# Suppression d'une clé :

# In[5]:


del mois['Mar ']  # supprimer une clé 
print(mois)


# Test de présence d'une clé

# In[6]:


"Fev " in mois # Test de présence d’une clé


# ### Fonctions pratiques

# #### Update : mise à jour d'un dictionnaire avec les éléments d'un autre dictionnaire
# 
# Si vous souhaitez mettre à jour un dictionnaire avec des éléments provenant d'un autre dictionnaire ou d'un itérable de paires clé/valeur, utilisez la méthode `update`.

# In[7]:


Notes = {"Pierre": 6}
Nouvelle_notes = {"Marie": 16, 'Jean': 10}
Notes.update(Nouvelle_notes)


# In[8]:


Notes.update(Julie=13, Michel=9)


# In[9]:


Notes


# #### Paramètre clé dans Max() : Trouver la clé ayant la plus grande valeur
# 
# Appliquer `max` sur un dictionnaire Python vous donnera la clé la plus grande, et non la clé avec la plus grande valeur. Si vous voulez trouver la clé ayant la plus grande valeur, spécifiez-la en utilisant le paramètre `key` dans la méthode `max`.

# In[10]:


Notes = {'Pierre': 6, 'Marie': 16, 'Jean': 10, 'Julie': 13, 'Michel': 9}

max(Notes)


# In[11]:


max_val = max(Notes, key=lambda k: Notes[k])
max_val


# #### dict.get : Obtenir la valeur par défaut d'un dictionnaire si une clé n'existe pas
# 
# Si vous voulez obtenir la valeur par défaut lorsqu'une clé n'existe pas dans un dictionnaire, utilisez `dict.get`. Dans le code ci-dessous, puisqu'il n'y a pas de clé `Cours3`, la valeur par défaut `Anglais` est retournée.

# In[12]:


Cours = {'Cours1': 'Français', 'Cours2': 'Mathématique'}


# In[13]:


Cours.get('Cours1', 'Anglais')


# In[14]:


Cours.get('Cours3', 'Anglais')


# #### dict.fromkeys: Obtenir un dictionnaire à partir d'une liste et d'une valeur
# 
# Si vous voulez obtenir un dictionnaire à partir d'une liste et d'une valeur, essayez `dict.fromkeys`.

# In[15]:


Informatique = ['Clavier', 'Ecran', 'Souris', 'Disque dur']
Courses = ['Pâtes', 'Jus', 'Salade']
Lieu1 = 'Fnac'
Lieu2 = 'Auchan'


# On peut utiliser `dict.fromkeys` pour créer un dictionnaire des lieux pour les produits Informatique :

# In[16]:


Informatique_lieu = dict.fromkeys(Informatique, Lieu1)
Informatique_lieu


# dictionnaire pour lieu des courses :

# In[17]:


Courses_lieu = dict.fromkeys(Courses, Lieu2)
Courses_lieu


# Ces 2 résultats peuvent être combinés dans un dictionnaire commun :

# In[18]:


Lieux = {**Informatique_lieu, **Courses_lieu}
Lieux

