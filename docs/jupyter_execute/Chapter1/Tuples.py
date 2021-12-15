#!/usr/bin/env python
# coding: utf-8

# ## Les Tuples

# Les tuples sont comme des listes, sauf qu'ils ne peuvent pas être modifiés une fois créés, c'est-à-dire qu'ils sont *immuables*.
# En Python, les tuples sont créés en utilisant la syntaxe  `(..., ..., ...)`, ou  `..., ...`:

# In[1]:


Mon_Tuple =('Carmen ', 'Georges Bizet ', 1875)
print (type ( Mon_Tuple ))


# In[2]:


point = 24, 17

print(point, type(point))


# On peut décompresser un tuple en l'affectant à une liste de variables séparées par des virgules :

# In[3]:


x, y = point

print("x =", x)
print("y =", y)


# On peut accéder aux élément d'un tuples en précisant les index

# In[4]:


Mon_Tuple [1]


# Si on essaye d'assigner une nouvelle valeur à un élément dans un tuple, nous obtenons une erreur:

# In[5]:


Mon_Tuple [1]=10

