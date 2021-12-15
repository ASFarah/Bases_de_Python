#!/usr/bin/env python
# coding: utf-8

# ## Structures de contrôle

# Présentation des structures de contrôle : Conditions, branchements et boucles

# In[1]:



# Afficher la table des matières

from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ### Test : if - elif - else

# La syntaxe Python pour l'exécution conditionnelle du code utilise les mots-clés `if`, `elif` (else if), `else` :

# In[2]:


Instruction1 = False
Instruction2 = False

if Instruction1:
    print("Instruction1 est vrai")
    
elif Instruction2:
    print("Instruction2 est vrai")
    
else:
    print("Instruction1 et Instruction2 sont toutes les deux False")


# **Remarque :** être très attentif sur la gestion des `indentations` car la fin
# d’indentation signifie la fin d’un bloc de commandes

# L'étendue d'un bloc de code est définie par le niveau d'indentation (habituellement une tabulation ou quatre espaces blancs). Cela signifie qu'il faut faire attention à indenter votre code correctement, sinon vous obtiendrez des erreurs de syntaxe.

# In[3]:


Instruction1 = Instruction2 = True

if Instruction1:
    if Instruction2:
        print("l'Instruction1 et l'Instruction2 sont toutes les deux True")


# In[4]:


# Mauvaise indentation!
if Instruction1:
    if Instruction2:
    print("l'Instruction1 et l'Instruction2 sont toutes les deux True")  # Cette ligne n'est pas correctement indentée


# In[84]:


Instruction1 = False 

if Instruction1:
    print("Afficher si Instruction1 est True")
    
    print("Encore à l'intérieur du bloc if")


# In[85]:


if Instruction1:
    print("Afficher si Instruction1 est True")
    
print("Maintenant à l'extérieur du bloc if")


# In[86]:


##### Syntaxe compacte d'une assignation conditionnelle


# Exemple :

# In[87]:


x,y = 10, 6
if x < y:
    minimum = x
else:
    minimum = y

minimum


# Python offre une syntaxe abrégée (inspirée du C) pour faire ceci :

# In[88]:


minimum = x if x < y else y
minimum


# ### Boucles

# En Python, les boucles peuvent être programmées de différentes façons. Il y a deux types de boucles, la boucle **for** parcourt un ensemble, la boucle **while** continue tant qu’une condition est vraie.
# 
# La plus courante est la boucle for, qui est utilisée avec des objets itérables, comme des listes. La syntaxe de base est :

# #### Boucle for :

# In[89]:


for x in [1,2,3]:
    print(x)


# In[90]:


for i in range (4) :    # Par défaut range commence à 0
    print(i, end =" ")


# **Note :** 
#     - range (4) génère une liste de 0 à n-1 donc, ne comprend pas 4!
#             Syntaxe générale de range : range(start, stop, step)
#     - la propriété end = " " dans la fonction `print` permet de rester sur la même ligne lors de l'affichage

# Pour itérer sur les paires clé-valeur d'un dictionnaire:

# In[91]:


for cle, valeur in mois.items():
    print(cle + " = " + str(valeur))


# Parfois, il est utile d'avoir accès aux indices des valeurs lors de l'itération sur une liste. Nous pouvons utiliser la fonction enumerate pour cela :

# In[ ]:


for idx, x in enumerate(range(-3,3)):
    print(idx, x)


# ##### Listes en Compréhension

# La liste en compréhension permet d’éviter une écriture en boucle explicite et rend l’exécution plus rapide

# In[ ]:


L = [x ** 2 for x in range (0 ,5)]
print (L)


# est la version courte de :

# In[ ]:


L =list()
for x in range (0, 5):
    L.append (x ** 2)
print (L)


# #### Boucle while :

# Parfois, on ne sait pas à l'avance combien de fois on veut exécuter un bloc d'instructions. Dans ce cas, il vaut mieux utiliser une boucle `while` dont la syntaxe est:

# In[ ]:


'''
while CONDITION:
    INSTRUCTION 1
    INSTRUCTION 2
    ...
    INSTRUCTION n
'''


# Le bloc d'instruction est exécuté (au complet) tant que la condition est satisfaite. La condition est testée avant l'exécution du bloc, mais pas pendant. C'est donc toutes les instructions du bloc qui sont exécutées si la condition est vraie. Par exemple, on peut afficher les puissances de 5 inférieures à un million avec une boucle `while` :

# In[ ]:


a = 1
while a < 1000000:
    print(a)
    a = a * 5


# In[ ]:


ct =2
while ct <= 8:
    print(ct , end =" ")
    ct = ct + 2


# ### Interruptions de boucles avec break et continue

# * La commande `break` permet d'interrompre une boucle for ou while en cours:

# In[ ]:


for i in range(10):
    if i == 5:
        break
    print(i)


# On remarque que les valeurs plus grandes que 4 n'ont pas été affichées par la fonction `print`.

# * La commande `continue` permet de continuer le parcours d'une boucle à la valeur suivante :

# In[ ]:


for i in range(10):
    if i == 5:
        continue
    print(i)


# On remarque que la valeur 5 n'a pas été affichées par la fonction `print`.
