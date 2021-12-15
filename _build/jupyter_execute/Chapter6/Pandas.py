#!/usr/bin/env python
# coding: utf-8

# ## Pandas en 10 minutes

# In[1]:


#Pour intégrer les graphes à votre notebook, il suffit de faire
get_ipython().run_line_magic('matplotlib', 'inline')

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# On importe générallement les librairies suivantes

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Création d'objets

# On créé une 'Series' en lui passant une liste de valeurs, en laissant pandas créer un index d'entiers

# In[3]:


s = pd.Series([1,3,5,np.nan,6,8])
s


# On créé un DataFrame en passant un array numpy, avec un index sur sur une date et des colonnes labellisées

# In[4]:


dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df


# On peut également créer un DataFrame en passant un dictionnaire d'objets qui peut être converti en sorte de série.

# In[5]:


df2 = pd.DataFrame({ 'A' : 1.,
'B' : pd.Timestamp('20130102'),
'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
'D' : np.array([3] * 4,dtype='int32'),
'E' : pd.Categorical(["test","train","test","train"]),
'F' : 'foo' })

df2


# Chaque colonne a son propre dtypes

# In[6]:


df2.dtypes


# On peut afficher les premières lignes et les dernières

# In[7]:


print(df.head())
print(df.tail())


# On peut afficher l'index, les colonnes et les données numpy

# In[8]:


print(df.index)
print(df.columns)
print(df.values)


# La méthode describe permet d'afficher un résumé des données

# In[9]:


df.describe()


# On peut faire la transposée, trier en fonction d'un axe ou des valeurs

# In[10]:


print(df.T)


# In[11]:


df.sort_index(axis=1, ascending=False)


# In[12]:


df.sort_values(by='B')


# ### Selection des données

# #### Getting

# Selection d'une colonne (équivalent à df.A)

# In[13]:


print(df['A'])
print(df[0:3])
print(df['20130102':'20130104'])


# #### Selection par Label

# En utilisant un label

# In[14]:


df.loc[dates[0]]


# Selection de plusieurs axes par label

# In[15]:


df.loc[:,['A','B']]


# Avec le label slicing, les deux points de terminaisons sont INCLUS

# In[16]:


df.loc['20130102':'20130104', ['A','B']]


# Obtenir une valeur scalaire

# In[17]:


df.loc[dates[0],'A']


# Acces plus rapide (méthode équivalente à la précédente)

# In[18]:


df.at[dates[0],'A']


# #### Selection par position

# Integer : 

# In[19]:


df.iloc[3]


# Tranches d'entiers, similaire à numpy

# In[20]:


df.iloc[3:5,0:2]


# Par liste d'entiers

# In[21]:


df.iloc[[1,2,4],[0,2]]


# Découpage de ligne explicite

# In[22]:


df.iloc[1:3,:]


# Obtenir une valeur explicitement

# In[23]:


df.iloc[1,1]


# Acces rapide au scalaire

# In[24]:


df.iat[1,1]


# #### Indexation booléenne

# En utilisant une valeur sur une colonne : 

# In[25]:


df[df.A > 0]


# Opérateur where : 

# In[26]:


df[df > 0]


# Pour filter, on utilise la méthode isin()

# In[27]:


df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
print(df2)

df2[df2['E'].isin(['two','four'])]


# #### Ajouter / modifier valeurs / colonnes

# Ajouter une nouvelle colonne automatiquement aligne les données par index.

# In[28]:


s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
print(s1)
df['F'] = s1


# Modifier une valeur par label

# In[29]:


df.at[dates[0],'A'] = 0


# In[30]:


df.loc[dates[0],'A']


# Modifier une valeur par position

# In[31]:


df.iat[0,1] = 0


# Modifier une valeur en assignant un tableau numpy

# In[32]:


df.loc[:,'D'] = np.array([5] * len(df))


# In[33]:


df


# #### Gérer les données manquantes

# Pandas utilise le type np.nan pour représenter les valeurs manquantes. Ce n'est pas codé pour faire des calculs.

# Reindex permet de changer/ajouter/supprimer les index d'un axe. Cette fonction retourne une copie des données

# In[34]:


df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
df1


# Pour supprimer les lignes contenant des NaN :

# In[35]:


df1.dropna(how='any')


# Remplacement des valeurs manquantes

# In[36]:


df1.fillna(value=5)


# Obtenir le masque de booléen de l'emplacement des nan

# In[37]:


pd.isnull(df1)


# ### Opérations

# #### Stats

# Les opérations excluent généralement les données manquantes.

# In[38]:


print(df.mean())
print(df.mean(1)) #Autre axe


# Situation avec des objets de dimmension différentes. En plus, pandas va automatiquement étendre la donnée sur la dimension spécifiée

# In[39]:


s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)

print(s)

df.sub(s, axis='index')


# In[40]:


help(df.sub)


# In[41]:


df.sub(s, axis='index')


# #### Apply

# Appliquer des foncitons aux données

# In[42]:


df.apply(np.cumsum)


# In[43]:


df.apply((lambda x: x.max() - x.min()))


# #### Histogramme

# In[44]:


s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())


# #### Methodes String

# Les séries sont équipées de méthodes pour traiter les strings avec l'attribut str qui rend facile la manipulation de chaque élémen d'un tableau. On utilise régulièrement des expressions régulières.

# In[45]:


s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()


# ### Regrouper

# #### Concaténation

# Pandas fournit des methodes pour facilement combiner des Series, DatFrame et des Panel objets avec des types variés de set logique pour les indexes et des fonctionnalités d'algèbre dans le cas de jointure / regroupement

# On peut concaténer des objets pandas avec concat()

# In[46]:


df = pd.DataFrame(np.random.randn(10, 4))
print(df)


# In[47]:


pieces = [df[:3], df[3:7], df[7:]]


# In[48]:


pd.concat(pieces)


# #### Jointures

# On peut merger à la manière de requete SQL.

# In[49]:


left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

print(left)
print(right)

print(pd.merge(left, right, on='key'))


# #### Append

# In[50]:


df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
df


# In[51]:


s = df.iloc[3]
df.append(s, ignore_index=True)


# #### Groupement

# Le regroupement comprend les étapes suivantes :
#    * Séparation de la donnée en groupes
#    * Appliquer une fonction a chaque group indépendamment
#    * Combiner les resultats dans une structure de données

# In[52]:


df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
'foo', 'bar', 'foo', 'foo'],
'B' : ['one', 'one', 'two', 'three',
'two', 'two', 'one', 'three'],
'C' : np.random.randn(8),
'D' : np.random.randn(8)})


# In[53]:


df


# Groupement et somme des groupes

# In[54]:


df.groupby('A').sum()


# Groupement de multiple colonnes

# In[55]:


df.groupby(['A','B']).sum()


# #### Reformation

# Stack

# In[56]:


tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two',
'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
df2


# La méthode stack() compresses un level dans les colonnes du dataframe
# 

# In[57]:


stacked = df2.stack()
stacked


# Avec une 'stacked' dataframe ou série, l'opération inverse est unstack()
