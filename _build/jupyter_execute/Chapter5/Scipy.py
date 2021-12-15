#!/usr/bin/env python
# coding: utf-8

# ## SciPy - Librairie d'algorithmes pour le calcul scientifique en Python

# Librairie de calcul numérique : intégration numérique, résolution d’équations différentielles, algèbre
# linéaire, traitement du signal, optimisation…

# In[1]:


#Pour intégrer les graphes à votre notebook, il suffit de faire
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from jyquickhelper import add_notebook_menu
add_notebook_menu()


# ### Introduction
# 
# SciPy s'appuie sur NumPy.
# 
# SciPy fournit des implémentations efficaces d'algorithmes standards.
# 
# Certains des sujets couverts par SciPy:
# 
# * Fonctions Spéciales ([scipy.special](http://docs.scipy.org/doc/scipy/reference/special.html))
# * Intégration ([scipy.integrate](http://docs.scipy.org/doc/scipy/reference/integrate.html))
# * Optimisation ([scipy.optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html))
# * Interpolation ([scipy.interpolate](http://docs.scipy.org/doc/scipy/reference/interpolate.html))
# * Transformées de Fourier ([scipy.fftpack](http://docs.scipy.org/doc/scipy/reference/fftpack.html))
# * Traitement du Signal ([scipy.signal](http://docs.scipy.org/doc/scipy/reference/signal.html))
# * Algèbre Linéaire ([scipy.linalg](http://docs.scipy.org/doc/scipy/reference/linalg.html))
# * Matrices *Sparses* et Algèbre Linéaire Sparse ([scipy.sparse](http://docs.scipy.org/doc/scipy/reference/sparse.html))
# * Statistiques ([scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html))
# * Traitement d'images N-dimensionelles ([scipy.ndimage](http://docs.scipy.org/doc/scipy/reference/ndimage.html))
# * Lecture/Ecriture Fichiers IO ([scipy.io](http://docs.scipy.org/doc/scipy/reference/io.html))
# 
# Durant ce cours on abordera certains de ces modules.
# 
# Pour utiliser un module de SciPy dans un programme Python il faut commencer par l'importer.
# 
# Voici un exemple avec le module *linalg*

# In[3]:


from scipy import linalg


# On aura besoin de NumPy:

# In[4]:


import numpy as np


# Et de matplotlib/pylab:

# In[5]:


# et JUSTE POUR MOI (pour avoir les figures dans le notebook)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ### Fonctions Spéciales
# 
# Un grand nombre de fonctions importantes, notamment en physique, sont disponibles dans le module *scipy.special*
# 
# Pour plus de détails: http://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special. 
# 
# Un exemple avec les fonctions de Bessel:

# In[6]:


# jn : Bessel de premier type
# yn : Bessel de deuxième type
from scipy.special import jn, yn


# In[7]:


get_ipython().run_line_magic('pinfo', 'jn')


# In[8]:


n = 0    # ordre
x = 0.0

# Bessel de premier type
print ("J_%d(%s) = %f" % (n, x, jn(n, x)))

x = 1.0
# Bessel de deuxième type
print("Y_%d(%s) = %f" % (n, x, yn(n, x)))


# In[9]:


x = np.linspace(0, 10, 100)

for n in range(4):
    plt.plot(x, jn(n, x), label=r"$J_%d(x)$" % n)
plt.legend()


# In[10]:


from scipy import special
get_ipython().run_line_magic('pinfo', 'special')


# ### Intégration
# 
# #### intégration numerique
# 
# L'évaluation numérique de:
# 
# $\displaystyle \int_a^b f(x) dx$
# 
# est nommée *quadrature* (abbr. quad). SciPy fournit différentes fonctions: par exemple `quad`, `dblquad` et `tplquad` pour les intégrales simples, doubles ou triples.

# In[11]:


from scipy.integrate import quad, dblquad, tplquad


# In[12]:


get_ipython().run_line_magic('pinfo', 'quad')


# L'usage de base:

# In[13]:


# soit une fonction f
def f(x):
    return x


# In[14]:


a, b = 1, 2 # intégrale entre a et b

val, abserr = quad(f, a, b)

print ("intégrale =", val, ", erreur =", abserr )


# Exemple intégrale double:

# $\int_{x=1}^{2} \int_{y=1}^{x} (x + y^2) dx dy$

# In[15]:


get_ipython().run_line_magic('pinfo', 'dblquad')


# In[16]:


def f(y, x):
    return x + y**2

def gfun(x):
    return 1

def hfun(x):
    return x

print(dblquad(f, 1, 2, gfun, hfun))


# #### Equations différentielles ordinaires (EDO)
# 
# SciPy fournit deux façons de résoudre les EDO: Une API basée sur la fonction `odeint`, et une API orientée-objet basée sur la classe `ode`.
# 
# `odeint` est plus simple pour commencer.
# 
# Commençons par l'importer:

# In[17]:


from scipy.integrate import odeint


# Un système d'EDO se formule de la façon standard:
# 
# $y' = f(y, t)$
# 
# avec 
# 
# $y = [y_1(t), y_2(t), ..., y_n(t)]$ 
# 
# et $f$ est une fonction qui fournit les dérivées des fonctions $y_i(t)$. Pour résoudre une EDO il faut spécifier $f$ et les conditions initiales, $y(0)$.
# 
# Une fois définies, on peut utiliser `odeint`:
# 
#     y_t = odeint(f, y_0, t)
# 
# où `t` est un NumPy *array* des coordonnées en temps où résoudre l'EDO. `y_t` est un array avec une ligne pour chaque point du temps `t`, et chaque colonne correspond à la solution `y_i(t)` à chaque point du temps. 

# ##### Exemple: double pendule

# Description: http://en.wikipedia.org/wiki/Double_pendulum

# In[18]:


from IPython.core.display import Image
Image(url='http://upload.wikimedia.org/wikipedia/commons/c/c9/Double-compound-pendulum-dimensioned.svg')


# Les équations du mouvement du pendule sont données sur la page wikipedia:
# 
# ${\dot \theta_1} = \frac{6}{m\ell^2} \frac{ 2 p_{\theta_1} - 3 \cos(\theta_1-\theta_2) p_{\theta_2}}{16 - 9 \cos^2(\theta_1-\theta_2)}$
# 
# ${\dot \theta_2} = \frac{6}{m\ell^2} \frac{ 8 p_{\theta_2} - 3 \cos(\theta_1-\theta_2) p_{\theta_1}}{16 - 9 \cos^2(\theta_1-\theta_2)}.$
# 
# ${\dot p_{\theta_1}} = -\frac{1}{2} m \ell^2 \left [ {\dot \theta_1} {\dot \theta_2} \sin (\theta_1-\theta_2) + 3 \frac{g}{\ell} \sin \theta_1 \right ]$
# 
# ${\dot p_{\theta_2}} = -\frac{1}{2} m \ell^2 \left [ -{\dot \theta_1} {\dot \theta_2} \sin (\theta_1-\theta_2) +  \frac{g}{\ell} \sin \theta_2 \right]$
# 
# où les $p_{\theta_i}$ sont les moments d'inertie. Pour simplifier le code Python, on peut introduire la variable $x = [\theta_1, \theta_2, p_{\theta_1}, p_{\theta_2}]$
# 
# ${\dot x_1} = \frac{6}{m\ell^2} \frac{ 2 x_3 - 3 \cos(x_1-x_2) x_4}{16 - 9 \cos^2(x_1-x_2)}$
# 
# ${\dot x_2} = \frac{6}{m\ell^2} \frac{ 8 x_4 - 3 \cos(x_1-x_2) x_3}{16 - 9 \cos^2(x_1-x_2)}$
# 
# ${\dot x_3} = -\frac{1}{2} m \ell^2 \left [ {\dot x_1} {\dot x_2} \sin (x_1-x_2) + 3 \frac{g}{\ell} \sin x_1 \right ]$
# 
# ${\dot x_4} = -\frac{1}{2} m \ell^2 \left [ -{\dot x_1} {\dot x_2} \sin (x_1-x_2) +  \frac{g}{\ell} \sin x_2 \right]$

# In[19]:


g = 9.82
L = 0.5
m = 0.1

def dx(x, t):
    """The right-hand side of the pendulum ODE"""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    dx1 = 6.0/(m*L**2) * (2 * x3 - 3 * np.cos(x1-x2) * x4)/(16 - 9 * np.cos(x1-x2)**2)
    dx2 = 6.0/(m*L**2) * (8 * x4 - 3 * np.cos(x1-x2) * x3)/(16 - 9 * np.cos(x1-x2)**2)
    dx3 = -0.5 * m * L**2 * ( dx1 * dx2 * np.sin(x1-x2) + 3 * (g/L) * np.sin(x1))
    dx4 = -0.5 * m * L**2 * (-dx1 * dx2 * np.sin(x1-x2) + (g/L) * np.sin(x2))
    
    return [dx1, dx2, dx3, dx4]


# In[20]:


# on choisit une condition initiale
x0 = [np.pi/4, np.pi/2, 0, 0]


# In[21]:


# les instants du temps: de 0 à 10 secondes
t = np.linspace(0, 10, 250)


# In[22]:


# On résout
x = odeint(dx, x0, t)
print(x.shape)


# In[23]:


# affichage des angles en fonction du temps
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].plot(t, x[:, 0], 'r', label="theta1")
axes[0].plot(t, x[:, 1], 'b', label="theta2")

x1 = + L * np.sin(x[:, 0])
y1 = - L * np.cos(x[:, 0])
x2 = x1 + L * np.sin(x[:, 1])
y2 = y1 - L * np.cos(x[:, 1])
    
axes[1].plot(x1, y1, 'r', label="pendulum1")
axes[1].plot(x2, y2, 'b', label="pendulum2")
axes[1].set_ylim([-1, 0])
axes[1].set_xlim([1, -1])
plt.legend()


# ### Transformées de Fourier
# 
# SciPy utilise la librairie [FFTPACK](http://www.netlib.org/fftpack/) écrite en FORTRAN.
# 
# Commençons par l'import:

# In[24]:


from scipy import fftpack


# Nous allons calculer les transformées de Fourier discrètes de fonctions spéciales:

# In[25]:


from scipy.signal import gausspulse

t = np.linspace(-1, 1, 1000)
x = gausspulse(t, fc=20, bw=0.5)

# Calcul de la TFD
F = fftpack.fft(x)

# calcul des fréquences en Hz si on suppose un échantillonage à 1000Hz
freqs = fftpack.fftfreq(len(x), 1. / 1000.)
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(t, x) # plot du signal
axes[0].set_ylim([-2, 2])

axes[1].plot(freqs, np.abs(F)) # plot du module de la TFD
axes[1].set_xlim([0, 200])
# mask = (freqs > 0) & (freqs < 200)
# axes[0].plot(freqs[mask], abs(F[mask])) # plot du module de la TFD
axes[1].set_xlabel('Freq (Hz)')
plt.show()


# ### Algèbre linéaire
# 
# Le module de SciPy pour l'algèbre linéaire est `linalg`. Il inclut des routines pour la résolution des systèmes linéaires, recherche de vecteur/valeurs propres, SVD, Pivot de Gauss (LU, cholesky), calcul de déterminant etc.
# 
# Documentation : http://docs.scipy.org/doc/scipy/reference/linalg.html

# ##### Résolution d'equations linéaires
# 
# Trouver x tel que:
# 
# $A x = b$
# 
# avec $A$ une matrice et $x,b$ des vecteurs.

# In[26]:


A = np.array([[1,0,3], [4,5,12], [7,8,9]], dtype=np.float)
b = np.array([[1,2,3]], dtype=np.float).T
print (A)
print (b)


# In[27]:


from scipy import linalg
x = linalg.solve(A, b)
print (x)


# In[28]:


print (x.shape)
print (b.shape)


# ##### Valeurs propres et vecteurs propres

# $\displaystyle A v_n = \lambda_n v_n$
# 
# avec $v_n$ le $n$ème vecteur propre et $\lambda_n$ la $n$ème valeur propre.
# 
# Les fonctions sont: `eigvals` et `eig`

# In[29]:


A = np.random.randn(3, 3)


# In[30]:


evals, evecs = linalg.eig(A)


# In[31]:


evals


# In[32]:


evecs


# Si A est symmétrique

# In[33]:


A = A + A.T
# A += A.T  # ATTENTION MARCHE PAS !!!!
evals = linalg.eigvalsh(A)
print (evals)


# In[34]:


print (linalg.eigh(A))


# ##### Opérations matricielles

# In[35]:


# inversion
linalg.inv(A)


# In[36]:


# vérifier


# In[37]:


# déterminant
linalg.det(A)


# In[38]:


# normes
print (linalg.norm(A, ord='fro'))  # frobenius
print (linalg.norm(A, ord=2))
print (linalg.norm(A, ord=np.inf))


# ### Optimisation
# 
# **Objectif**: trouver les minima ou maxima d'une fonction
# 
# Doc : http://scipy-lectures.github.com/advanced/mathematical_optimization/index.html
# 
# On commence par l'import

# In[39]:


from scipy import optimize


# #### Trouver un minimum

# In[40]:


def f(x):
    return 4*x**3 + (x-2)**2 + x**4


# In[41]:


x = np.linspace(-5, 3, 100)
plt.plot(x, f(x))


# Nous allons utiliser la fonction `fmin_bfgs`:

# In[42]:


x_min = optimize.fmin_bfgs(f, x0=-3)
x_min


# #### Trouver les zéros d'une fonction
# 
# Trouver $x$ tel que $f(x) = 0$. On va utiliser `fsolve`.

# In[43]:


omega_c = 3.0
def f(omega):
    return np.tan(2*np.pi*omega) - omega_c/omega


# In[44]:


x = np.linspace(0, 3, 1000)
y = f(x)
mask = np.where(abs(y) > 50)
x[mask] = y[mask] = np.nan # get rid of vertical line when the function flip sign
plt.plot(x, y)
plt.plot([0, 3], [0, 0], 'k')
plt.ylim(-5,5)


# In[45]:


np.unique(
    (optimize.fsolve(f, np.linspace(0.2, 3, 40))*1000).astype(int)
) / 1000.


# In[46]:


optimize.fsolve(f, 0.72)


# In[47]:


optimize.fsolve(f, 1.1)


# ##### Estimation de paramètres de fonctions

# In[48]:


from scipy.optimize import curve_fit

def f(x, a, b, c):
    """
    f(x) = a exp(-bx) + c
    """
    return a*np.exp(-b*x) + c

x = np.linspace(0, 4, 50)
y = f(x, 2.5, 1.3, 0.5)
yn = y + 0.2*np.random.randn(len(x))  # ajout de bruit


# In[49]:


plt.plot(x, yn)
plt.plot(x, y, 'r')


# In[50]:


(a, b, c), _ = curve_fit(f, x, yn)
print (a, b, c)


# In[51]:


get_ipython().run_line_magic('pinfo', 'curve_fit')


# On affiche la fonction estimée:

# In[52]:


plt.plot(x, yn)
plt.plot(x, y, 'r')
plt.plot(x, f(x, a, b, c))


# Dans le cas de polynôme on peut le faire directement avec NumPy

# In[53]:


x = np.linspace(0,1,10)
y = np.sin(x * np.pi / 2.)
line = np.polyfit(x, y, deg=10)
plt.plot(x, y, '.')
plt.plot(x, np.polyval(line,x), 'r')
# xx = np.linspace(-5,4,100)
# plt.plot(xx, np.polyval(line,xx), 'g')


# ### Interpolation

# In[54]:


from scipy.interpolate import interp1d


# In[55]:


def f(x):
    return np.sin(x)


# In[56]:


n = np.arange(0, 10)  
x = np.linspace(0, 9, 100)

y_meas = f(n) + 0.1 * np.random.randn(len(n)) # ajout de bruit
y_real = f(x)

linear_interpolation = interp1d(n, y_meas)
y_interp1 = linear_interpolation(x)

cubic_interpolation = interp1d(n, y_meas, kind='cubic')
y_interp2 = cubic_interpolation(x)


# In[57]:


from scipy.interpolate import barycentric_interpolate, BarycentricInterpolator
get_ipython().run_line_magic('pinfo2', 'BarycentricInterpolator')


# In[58]:


plt.plot(n, y_meas, 'bs', label='noisy data')
plt.plot(x, y_real, 'k', lw=2, label='true function')
plt.plot(x, y_interp1, 'r', label='linear interp')
plt.plot(x, y_interp2, 'g', label='cubic interp')
plt.legend(loc=3);


# #### Images

# In[59]:


from scipy import ndimage
from scipy import misc
img = misc.ascent()
print (img)
type(img), img.dtype, img.ndim, img.shape


# In[60]:


plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()


# In[61]:


_ = plt.hist(img.reshape(img.size),200)


# In[62]:


img[img < 70] = 50
img[(img >= 70) & (img < 110)] = 100
img[(img >= 110) & (img < 180)] = 150
img[(img >= 180)] = 200
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()


# Ajout d'un flou

# In[63]:


img_flou = ndimage.gaussian_filter(img, sigma=2)
plt.imshow(img_flou, cmap=plt.cm.gray)


# Application d'un filtre

# In[64]:


img_sobel = ndimage.filters.sobel(img)
plt.imshow(np.abs(img_sobel) > 200, cmap=plt.cm.gray)
plt.colorbar()


# Accéder aux couches RGB d'une image:

# In[65]:


import imageio
img = imageio.imread('china.jpg')
print (img.shape)
plt.imshow(img[:,:,0], cmap=plt.cm.Reds)


# In[66]:


plt.imshow(img[:,:,1], cmap=plt.cm.Greens)


# Conversion de l'image en niveaux de gris et affichage:

# In[67]:


plt.imshow(np.mean(img, axis=2), cmap=plt.cm.gray)


# ### Pour aller plus loin
# 
# * http://www.scipy.org - The official web page for the SciPy project.
# * http://docs.scipy.org/doc/scipy/reference/tutorial/index.html - A tutorial on how to get started using SciPy. 
# * http://scipy-lectures.github.io
# 
