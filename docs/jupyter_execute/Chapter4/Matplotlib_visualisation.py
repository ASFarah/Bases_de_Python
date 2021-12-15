#!/usr/bin/env python
# coding: utf-8

# ## Matplotlib (visualisation en 2D et 3D pour Python)

# Librairie pour les représentations graphiques

# In[1]:


#Pour intégrer les graphes à votre notebook, il suffit de faire
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# *Aparté*
# 
# Les librairies de visualisation en python se sont beaucoup développées ([10 plotting librairies](http://www.xavierdupre.fr/app/jupytalk/helpsphinx/2016/pydata2016.html)). 
# 
# La référence reste [matplotlib](http://matplotlib.org/), et la plupart sont pensées pour être intégrées à ses objets (c'est par exemple le cas de [seaborn](https://stanford.edu/~mwaskom/software/seaborn/introduction.html), [mpld3](http://mpld3.github.io/), [plotly](https://plot.ly/) et [bokeh](http://bokeh.pydata.org/en/latest/)). Il est donc utile de commencer par se familiariser avec matplotlib.

# <center>
# <img src="images/images.png" width="300" hight="500">
# </center>

# ### Introduction
# 
#  

# Matplotlib est un module destiné à produire des graphiques de toute sorte (voir http://matplotlib.org/gallery.html pour une gallerie d’images produites avec Matplotlib).  Il est l’outil complémentaire de *numpy* et *scipy* lorsqu’on veut faire de l’analyse de données. Un guide d’utilisation se trouve à l’adresse http://matplotlib.org/users/index.html .

# On importe le module matplotlib.pyplot sour le nom plot afin d’avoir accès aux fonctions de façon plus simple

# In[3]:


import matplotlib.pyplot as plt


# ### Graphes simples

# la fonction la plus simple est la fonction **plot** sour la forme plot(x,y) où *x* est un tableau d’abscisses et *y* le tableau
# des ordonnées associées

# In[4]:


import numpy as np
x=np.linspace(0,1,21)
y=x*x
plt.plot(x,y)
plt.show()


# #### Plusieurs courbes en même temps
# 
# Il suffit de les ajouter les unes après les autres avant la commande plt.show().

# In[5]:


x=np.linspace(0,2*np.pi,101)
plt.plot(x,np.sin(x))
plt.plot(x,np.sin(2*x))
plt.plot(x,np.sin(3*x))
plt.show()


# De façon plus condensée, on peut le faire en une seule instruction

# In[6]:


x=np.linspace(0,2*np.pi,101)
plt.plot(x,np.sin(x),x,np.sin(2*x),x,np.sin(3*x))
plt.show()


# #### Couleurs, Marqueurs et styles de ligne

# MatplotLib offre la possibilité d'adopter deux types d'écriture : chaîne de caractère condensée ou paramétrage explicite via un système clé-valeur.

# In[7]:


x=np.linspace(0,2*np.pi,20)
plt.plot(x,np.sin(x), color='red', linewidth=3, linestyle="--", marker="o", markersize=5,
markeredgecolor="b", markeredgewidth=2)
plt.show()


# In[8]:


x=np.linspace(0,1,21)
y=x*x
plt.plot(x,y,color='g',marker='o',linestyle='dashed')
# plt.show()


# In[9]:


x=np.linspace(0,1,21)
y=x*x
plt.plot(x,y,'og--') #l'ordre des paramètres n'importe pas


# Plus de détails dans la documentation sur l'API de matplotlib pour paramétrer la
# <a href="http://matplotlib.org/api/colors_api.html">
# couleur
# </a>
# , les
# <a href="http://matplotlib.org/api/markers_api.html">
# markers
# </a>
# , et le
# <a href="http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D.set_linestyle">
# style des lignes
# </a>
# . MatplotLib est compatible avec plusieurs standards de couleur :
# - sous forme d'une lettre : 'b' = blue (bleu), 'g' = green (vert), 'r' = red (rouge), 'c' = cyan (cyan), 'm' = magenta (magenta), 'y' = yellow (jaune), 'k' = black (noir), 'w' = white (blanc).
# - sous forme d'un nombre entre 0 et 1 entre quotes qui indique le niveau de gris : par exemple '0.70' ('1' = blanc, '0' = noir).
# - sous forme d'un nom : par exemple 'red'.
# - sous forme html avec les niveaux respectifs de rouge (R), vert (G) et bleu (B) : '#ffee00'. Voici un site pratique pour récupérer une couleur en [RGB hexadécimal](http://www.proftnj.com/RGB3.htm). 
# - sous forme d'un triplet de valeurs entre 0 et 1 avec les niveaux de R, G et B : (0.2, 0.9, 0.1).

# In[10]:


x=np.linspace(0,1,21)
y=x*x

#avec la norme RGB
plt.plot(x,y,color='#D0BBFF',marker='o',linestyle='-.')
plt.plot(x*2,y,color=(0.156862745098039, 0.7333333333333333, 1.0),marker='o',linestyle='-.')


# #### Titre, légendes et labels

# On peut également améliorer notre graphique en inserant :
#     - des étiquettes (abscisses et ordonnées) : $xlabel(texte), ylabel(texte)$
#     - une légende : $legend()$
#     - la grille : $grid()$
#     - titre du graphique : $title(texte) $
#     - un texte sur le graphique : $text(x,y,texte) $
#     - une grille en filligrane : $grid()$
# 
#         

# Exemple :

# In[11]:


x = np.linspace(0, 2, 10)

plt.plot(x, x, 'o-', label='linear')
plt.plot(x, x ** 2, 'x-', label='quadratic')

plt.legend(loc='best')
plt.title('Linear vs Quadratic progression')
plt.xlabel('Input')
plt.ylabel('Output');
plt.show()


# #### Ticks et axes

# 3 méthodes clés : 
# - xlim() : pour délimiter l'étendue des valeurs de l'axe
# - xticks() : pour passer les graduations sur l'axe
# - xticklabels() : pour passer les labels
# 
# Pour l'axe des ordonnées c'est ylim, yticks, yticklabels.
# 
# Pour récupérer les valeurs fixées : 
# - plt.xlim() ou plt.get_xlim()
# - plt.xticks() ou plt.get_xticks()
# - plt.xticklabels() ou plt.get_xticklabels()
#     
# Pour fixer ces valeurs :
# - plt.xlim([start,end]) ou plt.set_xlim([start,end])
# - plt.xticks(my_ticks_list) ou plt.get_xticks(my_ticks_list)
# - plt.xticklabels(my_labels_list) ou plt.get_xticklabels(my_labels_list)
# 
# Si vous voulez customiser les axes de plusieurs sous graphiques, passez par une [instance de axis](http://matplotlib.org/users/artists.html) et non subplot.

# In[12]:


from numpy.random import randn

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)

serie1=randn(50).cumsum()
serie2=randn(50).cumsum()
serie3=randn(50).cumsum()
ax1.plot(serie1,color='#33CCFF',marker='o',linestyle='-.',label='un')
ax1.plot(serie2,color='#FF33CC',marker='o',linestyle='-.',label='deux')
ax1.plot(serie3,color='#FFCC99',marker='o',linestyle='-.',label='trois')

#sur le graphe précédent, pour raccourcir le range
ax1.set_xlim([0,21])
ax1.set_ylim([-20,20])

#faire un ticks avec un pas de 2 (au lieu de 5)
ax1.set_xticks(range(0,21,2))
#changer le label sur la graduation
ax1.set_xticklabels(["j +" + str(l) for l in range(0,21,2)])
ax1.set_xlabel('Durée après le traitement')

ax1.legend()
#permet de choisir l'endroit le plus vide


# #### Inclusion d'annotation et de texte, titre et libellé des axes 

# In[13]:


from numpy.random import randn

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(serie1,color='#33CCFF',marker='o',linestyle='-.',label='un')
ax1.plot(serie2,color='#FF33CC',marker='o',linestyle='-.',label='deux')
ax1.plot(serie3,color='#FFCC99',marker='o',linestyle='-.',label='trois')

ax1.set_xlim([0,21])
ax1.set_ylim([-20,20])
ax1.set_xticks(range(0,21,2))
ax1.set_xticklabels(["j +" + str(l) for l in range(0,21,2)])
ax1.set_xlabel('Durée après le traitement')

ax1.annotate("vous êtes ici ", xy=(7, 7), #point de départ de la flèche
             xytext=(10, 10),          #position du texte
            arrowprops=dict(facecolor='#000000', shrink=0.10),
            )

ax1.legend(loc='best')

plt.xlabel("Libellé de l'axe des abscisses")
plt.ylabel("Libellé de l'axe des ordonnées")
plt.title("Une idée de titre ?")
plt.text(5, -10, r'$\mu=100,\ \sigma=15$')

plt.show()


# #### matplotlib et le style

# Il est possible de définir son propre style. Cette possibilité est intéressante si vous faîtes régulièrement les mêmes graphes et voulez définir des templates (plutôt que de copier/coller toujours les mêmes lignes de code). Tout est décrit [ici](http://matplotlib.org/users/style_sheets.html).
# 
# Une alternative consiste à utiliser des feuilles de styles pré-définies. Par exemple [ggplot](http://blog.yhat.com/posts/ggplot-for-python.html), librairie très utilisée sous R.

# In[14]:


from numpy.random import randn

#pour que la définition du style soit seulement dans cette cellule notebook
with plt.style.context('ggplot'):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(serie1,color='#33CCFF',marker='o',linestyle='-.',label='un')
    ax1.plot(serie2,color='#FF33CC',marker='o',linestyle='-.',label='deux')
    ax1.plot(serie3,color='#FFCC99',marker='o',linestyle='-.',label='trois')

    ax1.set_xlim([0,21])
    ax1.set_ylim([-20,20])
    ax1.set_xticks(range(0,21,2))
    ax1.set_xticklabels(["j +" + str(l) for l in range(0,21,2)])
    ax1.set_xlabel('Durée après le traitement')

    ax1.annotate("You're here", xy=(7, 7), #point de départ de la flèche
                 xytext=(10, 10),          #position du texte
                arrowprops=dict(facecolor='#000000', shrink=0.10),
                )

    ax1.legend(loc='best')

    plt.xlabel("Libellé de l'axe des abscisses")
    plt.ylabel("Libellé de l'axe des ordonnées")
    plt.title("Une idée de titre ?")
    plt.text(5, -10, r'$\mu=100,\ \sigma=15$')

    #plt.show()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

print("De nombreux autres styles sont disponibles, pick up your choice! ", plt.style.available)
with plt.style.context('dark_background'):
    plt.plot(serie1, 'r-o')

# plt.show()


# ### Histogrammes

# Un exemple est plus explicite

# In[16]:


# Gaussian, mean 1, stddev .5, 1000 elements
# on crée un tableau avec des valeurs aléatoires obtenues
# avec une loi normale
samples = np.random.normal(loc=1.0, scale=0.5, size=1000)
print(samples.shape)
print(samples.dtype)
print(samples[:30])
plt.hist(samples, bins=50);
plt.show()


# On peut modifier quelques options intéressantes, par exemple
# 

# In[17]:


# on crée un tableau avec des valeurs aléatoires obtenues
# avec une loi normale
valeurs=np.random.randn(1000)
plt.hist(valeurs ,
        25, # nombre de barres
        cumulative=True, # histogramme cumulatif
        color='magenta', # couleur
        histtype='stepfilled' # type : bar,barstacked ,step,stepfilled
        )
plt.show()


# #### Deux histogramme sur la même figure

# In[18]:



samples_1 = np.random.normal(loc=1, scale=.5, size=10000)
samples_2 = np.random.standard_t(df=10, size=10000)
bins = np.linspace(-3, 3, 50)

# Définir un alpha et utiliser le même bins vu qu'on représente deux histogrammes
plt.hist(samples_1, bins=bins, alpha=0.5, label='samples 1')
plt.hist(samples_2, bins=bins, alpha=0.5, label='samples 2')
plt.legend(loc='upper left');
plt.show()


# ### Nuage de points

# In[19]:


plt.scatter(samples_1, samples_2, alpha=0.1);
plt.show()


# In[20]:


# Create sample data, add some noise
x = np.random.uniform(1, 100, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)

plt.scatter(x, y)
plt.show()


# ### Graphes en coordonnées polaires

# On utilise la commande polar :
# 

# In[21]:


theta=np.linspace(0,2*np.pi,500)
r=np.cos(3*theta)
plt.polar(theta,r)
plt.show()


# ### Figures et subplots

# L’affichage
# avec **plt.show()** ouvrira une fenêtre pour chaque figure créée. Plus intéressant, on peut dans une même figure, créer
# plusieurs graphiques. On utilise pour cela la commande **subplot** sous la forme subplot(p,q,n) où (p,q) représente la
# taille du tableau des sous-graphiques et n la position dans ce tableau : si p = 2 et q = 3, alors on crée une grille de 6
# graphiques (2 lignes et 3 colonnes) numérotés de 1 à 6.
# 

# In[22]:


x=np.linspace(0,np.pi,31)
plt.suptitle("Différents graphes") # le titre général

plt.subplot(2,2,1) # une grille 2x2 - premier graphique
plt.axis((0,np.pi,0,1))
y=np.sin(x)
plt.title('sinus')
plt.xticks([0,1,2,3])
plt.plot(x,y,"b-")

plt.subplot(2,2,2) # second graphique
plt.axis((0,np.pi,0,1))
plt.title('sinus points')
plt.plot(x,y,"go")

plt.subplot(2,2,3) # troisième graphique
y=np.cos(x)
plt.axis((0,np.pi,-1,1))
plt.title('cosinus')
plt.plot(x,y,"b-")

plt.subplot(2,2,4) # et le dernier
plt.axis((0,np.pi,-1,1))
plt.title('cosinus points')
plt.plot(x,y,"go")

plt.show() # on affiche le tout


# On peut également utiliser la syntaxe suivante :

# In[23]:


from numpy.random import randn

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)

# On peut compléter les instances de sous graphiques par leur contenu.
# Au passage, quelques autres exemples de graphes
ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))
ax3.plot(randn(50).cumsum(),'k--')


# **Faire attention :** Si aucune instance d'axes n'est précisée, la méthode plot est appliquée à la dernière instance créée.

# In[24]:


from numpy.random import randn

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
plt.plot(randn(50).cumsum(),'k--')
plt.show()


# ### Affichage d’une matrice

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
def f(x,y):
    return (2*x-y**2+y)*np.exp(-x**2-y**2)
x=np.linspace(-2,2,100) # discrétisation de [-2,2]
y=np.linspace(-2,2,100) # idem
X,Y = np.meshgrid(x,y) # création de la grille complète
Z=f(X,Y) # on évalue f sur la grille
plt.matshow(Z,cmap='hot') # image avec les valeurs , style de couleurs 'hot'
plt.show()


# ### Pour aller plus loin

# Pour explorer l'ensemble des catégories de graphiques possibles : [Gallery](http://matplotlib.org/gallery.html). Les plus utiles pour l'analyse de données : [scatter](http://matplotlib.org/examples/lines_bars_and_markers/scatter_with_legend.html), [scatterhist](http://matplotlib.org/examples/axes_grid/scatter_hist.html), [barchart](http://matplotlib.org/examples/pylab_examples/barchart_demo.html), [stackplot](http://matplotlib.org/examples/pylab_examples/stackplot_demo.html), [histogram](http://matplotlib.org/examples/statistics/histogram_demo_features.html), [cumulative distribution function](http://matplotlib.org/examples/statistics/histogram_demo_cumulative.html), [boxplot](http://matplotlib.org/examples/statistics/boxplot_vs_violin_demo.html), , [radarchart](http://matplotlib.org/examples/api/radar_chart.html).
