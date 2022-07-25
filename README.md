_[En](https://github.com/kaugrv/models_words/blob/main/README_EN.md)_

-----------------------------------

Des modèles et des mots
========================

**Des Modèles et des Mots est une série de publications dédiée à un ensemble de conversations sur le Computer Art et l'intelligence artificielle éditées par Gaëtan Robillard. La série entend présenter des entretiens avec Margit Rosen, Frieder Nake, Jérôme Nika, Véra Molnar, et Kazushi Mukaiyama. Les textes s'accompagnent d'une série d'images et de la diffusion d'un modèle d'apprentissage profond.**

Edition ESIPE – formation IMAC (Université Gustave Eiffel).

-----------------------------------

**Modèle**

# Trois lignes dans un espace latent
  
Gaëtan Robillard et Wendy Gervais.

## Avant-propos

_Des Modèles et des Mots_ s'inscrit dans un projet de recherche en art et en deep learning génératif. Le code présenté ici soutient un travail exploratoire sur les GANs (Generative Adversarial Networks, ou Réseaux antagonistes génératifs)[^21]. Ces modèles d'architecture sont établis depuis quelques années dans la recherche en informatique, employés pour produire des médias de synthèse à partir de large bases de données d'images. Ce support est pensé en tant qu'environnement de travail ou *framework*, base à compléter et à explorer, pour quiconque souhaiterait découvrir, façonner ou critiquer ce type de modèle dans un contexte de recherche et d'expérimentation visuelle.

Le plus souvent, les GANs sont entraînés à partir de grandes quantités de photographies ou de dessins produits à la main – des données observées dans le réel. Les images sont figuratives. Qu'en est-il de l'abstraction ? Et qu'en est-il des images déjâ produites par un code ? Ici, nous nous intéressons à des données d'entrainement qui sont des données de synthèse – des données générées par un algorithme et par le truchement de variables aléatoires – en référence au champ du Computer Art[^1]. À ceci s'ajoute une recherche sur l'image en mouvement, telle qu'elle apparaît en potentiel dans l'exploration de l'espace latent[^2] d'un GAN. L'une des perspectives du modèle est bien la création de séquences d'images nouvelles, issues de fonctions de parcour dans l'espace ou la « vision » du modèle entraîné.

Principalement écrit en Python, décomposé en trois Notebooks distincts : Training, Inference, Inference+ -- le code est commenté de façon à guider le profane à travers les différentes partie du *framework*. Les trois Notebooks présentés plus bas ont été configurés sur [Google Colaboratory](https://colab.research.google.com/drive/12WCzKlR--V8E7HMZHJ89nobVDCknCKmE#scrollTo=C7vmECVpwSZm), (aucune installation requise), et sont également exploitables dans un environnement local [^22].

D'une façon générale, l'apprentissage profond est un sujet technique complexe – notamment du fait de la très grande dimension des réseaux de neurones (le terme de boîte noire est adapté pour désigner ce problème). Tout en réfléchissant à la lisibilité de ce type de modèle, c'est ici la recherche artistique et pédagogique qui doit être mise en avant. Autant que possible, les aspects visuels de la démarche sont présentés : les données d'entraînement, l'architecture du GAN et les résultats.

[^21]: Le modèle de GAN présenté ici a été adapté à partir de l'ouvrage _Generative Deep Learning_ de David Foster (O’Reilly).

[^1]: Le Computer Art est entendu ici comme un champ iconographique à part entière, caractérisable par de nombreuses références à l'abstraction géométrique, à l'art cinétique et à l'art conceptuel. Bien qu'il soit difficile de circonscrire ce champs, on peut caractériser le Computer Art par son inscription dans l'esthétique infromationnelle et générative issu de théorciens comme Max Bense et Abraham Moles. Sans s'y réduire totalement, l'emploi d'algorithmes générateurs de nombres pseudo-aléatoires est trait marquant de l'époque pionnière.

[^2]: L'espace latent est l'espace théorique contenant les "points", ou vecteurs, support des motifs récurrents interprétables par un réseau de type apprentissage profond génératif. Ces vecteurs ou inputs sont traités par le GAN afin de générer les images finales ou outputs.

[^22]: Par exemple dans environnement comme Anaconda. Pour l'installation des dépendances, Cf. [Generative Deep Drawing](https://github.com/leogenot/DeepDrawing)

### Fondations

Cette démarche se fonde en partie sur une expérience pédagogique menée avec des étudiants ingénieurs en art et science qui a consisté à étudier le Computer Art à travers des artistes pionniers comme Frieder Nake, Véra Molnar, Georg Nees et Manfred Mohr. Après avoir choisi une oeuvre dans un corpus, les étudiants ont été amenés à penser et re-coder le répèrtoire visuel et algorithmique d'une oeuvre, de façon à constituer des données d'entraînement. Chaque groupe a alors entraîné un modèle avec ces données.

Dans cette démarche, deux processus génératifs sont enchassés : le premier pour générer des données de façon synthétique, le second pour explorer l'espace de représentation (l'espace latent) du réseau une fois celui-ci entrainé. Les données d'entraînements présentées ici – les « Trois lignes » – et leur méthode, sont directement issus des supports ayant guidé les étudiants dans la démarche.

De plus, la base algorithmique de notre modèle a été explorée lors du projet tuteuré _Generative Deep Drawing_ mené par des étudiants de deuxième année de la même formation. Deep Drawing est un projet d'exploration du dessins par l'intéraction avec un GAN (Cf. ressources).

-----------------------------------

## Données d'entraînement

« Trois lignes dans un espace latent » repose sur un ensemble de 10 000 images élémentaires et semblables, appelé « dataset », généré à l'aide d'un code Java écrit et exécuté dans l'environnement de programmation Processing. Le code permettant de générer ces images est disponible dans le [dossier « lines »](https://github.com/kaugrv/models_words/blob/main/lines/lines.pde).

Cet ensemble d'images constitue une seule et même classe. Chacune d'elles suit les mêmes règles de construction – elles sont détaillées ci-après, et est strictement dictée par l'execution du code. Cela en fait un dataset synthétique et algorithmique, comportant des images simples, chacune différente mais appartenant à un ensemble. Cette homogénéité relative sera utile pour entraîner notre réseau.

Chaque image de notre dataset est au format 128 par 128 pixels. Sur fond noir, on génère 3 lignes d'épaisseur 5 pixels. On distingue une ligne verticale et deux horizontales. Leur position dans l'image est définie de façon aléatoire (loi uniforme ou loi gaussienne, selon). Leurs couleurs sont également aléatoires (loi uniforme sur [0,3]). Chacune des 3 lignes peut prendre la couleur suivante[^3] :

- blanc
- jaune
- rouge
- bleu

![Extrait du dataset](https://user-images.githubusercontent.com/103901906/179547588-a322ef87-3ee4-4d77-b573-b6f613bf541a.png)

Par calcul, on peut approcher le nombre d'images contenue dans la classe. Chacune des 3 lignes comporte une abscisse (si verticale) ou une ordonnée (si horizontale) comprise entre 0 et 128, et possède une couleur choisie aléatoirement parmi 4. Le nombre de lignes verticales et horizontales étant fixé, on a donc :

$$ \prod_{i=1}^{3} (4 \times 128) = (4 \times 128)^3 = 134217728 $$  

c'est-à-dire 134 millions d'images possibles. En générant 10 000 d'entre elles, on couvre 0.007% d'images de la classe – autant dire que nous sommes certains de générer 10 000 images uniques. Ce dataset sera chargé lors de l'entraînement, à l'aide de le la fonction `load_dataset` de [loaders.py](https://github.com/kaugrv/models_words/blob/main/utils/loaders.py).

[^3]: Les couleurs exactes choisies sont tirées du travail de Piet Mondrian (Cf. _Trafalgar Square_, Piet Mondrian, 1943 ; reproduction in _[Museum of Modern Art](https://www.moma.org/collection/works/79879)_). Voir également _Projet Mondrian_ de Frieder Nake : _"I had scanned all of the neoplastic images of Mondrian's, and had arranged them into a time sequence of Mondrian's painting. The idea then was to analyze them year by year, in all sorts of statistical directions. Out of this, I wanted to develop a predictory system that would predict and create the New York Boogie-Woogie! Outrageously optimistic!"._ Frieder Nake, Margit Rosen, « The Art of Being Precise: Frieder Nake in Conversation [With Margit Rosen] », ZKM, 2022. [https://www.youtube.com/watch?v=Z_pOiHX6HYE](https://www.youtube.com/watch?v=Z_pOiHX6HYE)

## Architecture

Le réseau *Trois lignes* est un GAN de type WGAN-GP, codé en Python à l'aide des librairies _TensorFlow_ et _Keras_.

Rappelons qu'un GAN se constitue d'un générateur et d'un discriminateur (que l'on appelle critique). Le GAN repose sur la concurrence de ces deux réseaux : le générateur crée des échantillons visuels et les soumet au critique, qui à son tour leur donne un score. Ce score détermine si l'image analysée est une image observée (issue des données d'entraînement) ou une image produite par le générateur). Le générateur alors progresse pour engendrer des images de plus en plus enclines à tromper l'appréciation du critique.   

La classe de cette architecure est définie dans le code [WGANGP.py](https://github.com/kaugrv/models_words/blob/main/models/WGANGP.py). Lors de l'entraînement, le modèle est instancié pour travailler sur des images de 128 x 128 ; à l'entrée du générateur le vecteur (Z) est de dimension 100. Voici également les paramètres choisis pour les différentes _layers_ (critique, générateur) constitués principalement convolutions. Les paramètres des matrices de convolutions sont données par _filters, kernel_ et _strides_) :

```
gan = WGANGP(input_dim=(IMAGE_SIZE, IMAGE_SIZE, 3), 
             critic_conv_filters=[128, 256, 512, 1024],
             critic_conv_kernel_size=[5,5,5,5], 
             critic_conv_strides=[2, 2, 2, 2], 
             critic_batch_norm_momentum=None, 
             critic_activation='leaky_relu', 
             critic_dropout_rate=None, 
             critic_learning_rate=0.0002, 
             generator_initial_dense_layer_size=(8, 8, 512), 
             generator_upsample=[1, 1, 1, 1], 
             generator_conv_filters=[512, 256, 128, 3], 
             generator_conv_kernel_size=[10,10,10,10], 
             generator_conv_strides=[2, 2, 2, 2], 
             generator_batch_norm_momentum=0.9, 
             generator_activation='leaky_relu', 
             generator_dropout_rate=None, 
             generator_learning_rate=0.0002, 
             optimiser='adam', 
             grad_weight=10, 
             z_dim=100, 
             batch_size=BATCH_SIZE
             )
 ```
 
La matrice de convolution choisie pour le générateur est de taille 10 x 10 : `generator_conv_kernel_size=[10,10,10,10]`. La convolution est ainsi relativement grande, de façon à ce que le générateur réponde à la qualité abstraites des images en considérant à une échelle globale. En revanche, le critique est associé à une matrice de convolution de taille inférieure, 5 x 5, avec `critic_conv_kernel_size=[5,5,5,5]`. Les images sont scorées par le critique via une convolution de niveau plus locale que le générateur. En somme, si le générateur tend à l'abstraction visuelle – la ligne, le critique porte son analyse dans les détails. Cette asymétrie micro-macro des matrices de convolution entre générateur et le critique résulte d'une interrogation forte quant à la capacité du réseau en matière d'abstraction visuelle.

D'autres dimensions de la convolution pourraient être explorées en fonction de nouveaux datasets. Pour plus de détails sur la convolution, voir _[Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)_.

![Générateur](https://user-images.githubusercontent.com/103901906/179565557-e17c1acc-e9a2-48c1-aa5e-973d7caf03d2.png)

![Critique](https://user-images.githubusercontent.com/103901906/179565848-3a8334e6-1680-4085-9bad-cbcb81c8d142.png)

_Représentations graphiques des layers constituant le générateur et le critique, réalisées avec l'outil NN-SVG d'[Alex LeNail](https://github.com/alexlenail/NN-SVG). On observe en 3D l'articulation des layers et leurs dimensions successives, en proportions réelles en haut à droite et arangées pour une meilleure lisibilité au centre._

## Entraînement

La partie effective de l'entraînement est exécutable dans le Notebook [Training](https://colab.research.google.com/drive/12WCzKlR--V8E7HMZHJ89nobVDCknCKmE#scrollTo=C7vmECVpwSZm) sur Google Colab.

Après avoir cloné ce dépôt, on travaillera à sa racine, /models_words. Il est possible d'utiliser le dataset _lines_ décrit plus haut ou bien d'en utiliser un nouveau (en l'uploadant directement dans l'environnement Colab via une cellule ou en remplaçant le fichier data.zip). Au moins 10 000 images (format 128x128) sont conseillées, à placer dans un fichier data.zip_, mais selon la complexité des images, davantage seront peut-être nécessaires.

Les premières cellules servent à charger le modèle et les bibliothèques, puis à charger le dataset. Il est possible de configurer les paramètres de l'entraînement, notamment le numéro de l'entraînement que l'on va lancer (`RUN_ID`), ou le `BATCH_SIZE`, c'est-à-dire le nombre d'images du dataset présentées au réseau lors d'une itération. Par défaut, il est paramétré à 16, l'augmenter rendra le temps de calcul plus long, mais peut permettre d'améliorer la précision de l'entraînement (garder une puissance de 2, comme 32, 64 ou 128). Après avoir chargé le dataset (celui-ci devrait bien renvoyer `Found ... images belonging to 1 classes.`) et paramétré le GAN (comme expliqué plus haut dans _Architecture_), d'autres paramètres de l'entraînement sont modifiables :

* `EPOCHS` : le nombre d'expositions de l'entiereté du dataset au réseau neuronal. Il est généralement de l'ordre de 1000 époques, mais selon la complexité des images à analyser, il peut être inférieur ou supérieur.
* `PRINT_EVERY_N_BATCHES` : au lieu de générer une image à toutes les époques, on peut choisir la fréquence d'enregistrement. Les images générées se trouveront dans run/gan, dans le dossier correspondant au numéro de l'entraînement. Par exemple, pour `EPOCHS = 1201` et `PRINT_EVERY_N_BATCHES = 100`) on obtiendra :

![Exemple samples](https://user-images.githubusercontent.com/103901906/176936405-c4fbce75-1ece-419f-a47e-5bd0e4547ef4.png)

* `rows` et `columns` : chaque sample généré ne contient pas forcément une seule image générée par le réseau mais plutôt une grille d'images, afin d'observer l'évolution de l'apprentissage de manière globale. Par exemple, respectivement pour `rows = 5`, `columns = 5` et `rows = 2`, `columns = 2` :

![Exemple 5x5](https://user-images.githubusercontent.com/103901906/176937177-ef705ff3-603f-4fb1-9243-b15b1783b3a0.png) ![Exemple 2x2](https://user-images.githubusercontent.com/103901906/176937301-ef2df143-63bb-4dad-8ce9-86d777c364f6.png)

Il est possible d'observer l'évolution de la perte (_loss_, c'est-à-dire la note attribuée par le critique aux images générées) textuellement au cours des passages :

```
...
452 (5, 1) [D loss: (-92.2)(R -71.9, F -58.8, G 3.8)] [G loss: 41.4]
453 (5, 1) [D loss: (-116.2)(R -179.5, F 4.4, G 5.9)] [G loss: 1.4]
454 (5, 1) [D loss: (-95.4)(R -113.7, F -34.5, G 5.3)] [G loss: 46.7]
455 (5, 1) [D loss: (-104.6)(R -114.9, F -40.9, G 5.1)] [G loss: 51.2]
456 (5, 1) [D loss: (-108.6)(R -120.9, F -28.5, G 4.1)] [G loss: 5.9]
457 (5, 1) [D loss: (-82.6)(R -87.7, F -38.1, G 4.3)] [G loss: 40.1]
458 (5, 1) [D loss: (-92.0)(R -173.4, F 16.8, G 6.5)] [G loss: -37.0]
459 (5, 1) [D loss: (-120.8)(R -118.1, F -61.4, G 5.9)] [G loss: 61.5]
460 (5, 1) [D loss: (-101.2)(R -94.1, F -60.1, G 5.3)] [G loss: 84.7]
461 (5, 1) [D loss: (-86.1)(R -167.7, F 22.9, G 5.9)] [G loss: -37.0]
462 (5, 1) [D loss: (-92.4)(R -127.9, F -16.1, G 5.2)] [G loss: 26.5]
463 (5, 1) [D loss: (-109.2)(R -116.9, F -36.9, G 4.5)] [G loss: 7.6]
464 (5, 1) [D loss: (-103.0)(R -105.2, F -52.5, G 5.5)] [G loss: 30.6]
465 (5, 1) [D loss: (-76.8)(R -90.0, F -34.5, G 4.8)] [G loss: 58.8]
466 (5, 1) [D loss: (-107.0)(R -145.8, F -23.6, G 6.2)] [G loss: 13.6]
467 (5, 1) [D loss: (-106.2)(R -191.6, F 15.8, G 7.0)] [G loss: -42.8]
...
```

Nous pouvons aussi voir ces données de manière graphique, en générant une représentation de la convergence de l'entraînement :


<img src="https://user-images.githubusercontent.com/103901906/177615266-41ff49a0-16de-42bf-be11-5ed55a44dcc8.png" alt="Graphe" width="400"></img>
<img src="https://user-images.githubusercontent.com/103901906/177614854-75bcd83a-f65f-429c-b229-ffcb1738e8f7.png" alt="Légende" width="200"></img>



On peut ainsi observer l'évolution de l'entraînement, et la convergence des _loss_ : ici par exemple, la différence en moyenne entre les images réelles (_real_, issues du _batch_) et fausses (_fake_, générées par le générateur) tend vers 0 à partir d'environ 1100 époques. C'est donc à ce stade que l'entraînement est devenu assez satisfaisant pour l'arrêter.

L'image de ce graphe sera enregistrée avec les _samples_ en tant que _converge.png_. Finalement des cellules permettent pour un environnement Google Colab de télécharger premièrement toutes les images, et enfin le fichier generator.h5, le fichier du générateur du réseau lié à cet entraînement, qui sera indispensable pour la partie suivante : l'_Inference_.

## Inference

La partie _Inference_ se donne pour objectif d'explorer _l'espace latent_, l'espace contenant les données interprétables par le réseau qui lui sera capable de générer les images, une fois que l'entraînement a été effectué avec le Notebook précédent. Il faudra pour cela se munir de fichier du générateur _generator.h5_ et utiliser le [second Notebook, "Inference"](https://colab.research.google.com/drive/13g3rX2zgyxT5YKTZILBrISybmLJ4_pXi), depuis Google Colaboratory.

L'idée générale de cette exploration est d'étudier les images que le réseau est désormais capable de générer à l'issue de l'entraînement. En utilisant le fichier du générateur datant de la dernière époque de celui-ci, il va être possible de générer à l'envi de nouvelles images relevant de la même classe que les images générées par le réseau lors de la dernière époque. Il sera aussi possible de les étudier, et les _interpoler_ entre elles (c'est-à-dire introduire des vecteurs intermédiaires entre des vecteurs générés).

Une première fonction `generate_latent_points` génère un ou plusieurs vecteurs 100 (on choisit le nombre de vecteurs en paramétrant `nb_vec`), aléatoires (ce vecteur contiendra 100 valeurs _float_ aléatoires comprises entre -3 et 3). On utilisera donc `latent_dim = 100` et `nb_vec`, le nombre de vecteurs choisi, passé en argument. Nous utilisons ici `nb_vec = 2`, car par la suite nous étudierons l'interpolation de deux images générées et d'éventuels vecteurs 100 supplémentaires seraient inutiles.

La seconde fonction `interpolate_points` permet l'interpolation, c'est-à-dire de créer de nouveaux vecteurs se plaçant entre les deux vecteurs précédemment générés par le générateur. Si l'on interpole deux vecteurs (`nb_vec = 2`) avec une vingtaine d'autres (`nb_img = 20`) par exemple, il sera possible de créer une animation, en accolant les 20 images associées. Pour cela, la fonction `plot_generated` génère, enregistre et affiche toutes ces images.

![Animation 1](https://user-images.githubusercontent.com/103901906/177205129-48acd30e-9a0b-4f2b-8450-ea58f21e3d83.gif)

Pour résumer, pour explorer et interpoler les points de l'espace latent et les images associées :

* on génère un tableau `pts` de format (2,100) qui contient 2 vecteurs 100 avec la fonction `generate_latent_points`. On choisit le nombre de vecteurs avec de `nb_vec` -`nb_vec = 2` ici.
* on interpole les deux premiers tableaux de `pts` avec `interpolate_points`, ce qui donne le nouveau tableau `interpolated`
* on génère les images associées en passant le tableau `interpolated` en argument de `plot_generated`
* sur Colab, on peut télécharger l'ensemble de ces images au format zip, à l'aide de la dernière cellule.

## Inference +

Pour aller plus loin, une seconde version du [Notebook, "Inference +"](https://colab.research.google.com/drive/14oww73GEQrECNtgaj8iK78jSw8GtHIiE), ajoute quelques fonctionnalités supplémentaires :

* afficher le vecteur Z (vecteur 100 ici) sous la forme d'un images de 10x10 carrés, avec la fonction `printZ`. Exemples d'images générées avec la représentation du vecteur Z associé :

<img src="https://user-images.githubusercontent.com/103901906/178315771-00b5226d-bb5b-4c88-a091-59c70135336a.png" width="85"></img>
<img src="https://user-images.githubusercontent.com/103901906/178315053-f114c50f-4e11-4750-9846-19002c30553a.png" width="128"></img>

<img src="https://user-images.githubusercontent.com/103901906/178315953-689183b3-15ca-497a-ba4b-5456342083c2.png" width="85"></img>
<img src="https://user-images.githubusercontent.com/103901906/178315944-5dab8c66-eaad-4872-b45b-38311e176ec4.png" width="128"></img>

<img src="https://user-images.githubusercontent.com/103901906/178316212-068b2a10-08f3-4601-a69c-dc89c242755b.png" width="85"></img>
<img src="https://user-images.githubusercontent.com/103901906/178316192-73627c6d-faf6-4a06-9f5a-6600c61e43e7.png" width="128"></img>

Attention cependant, l'échelle des couleurs est relative (un niveau de gris ne représentera pas la même valeur d'une représentation à une autre).

* de là, étudier le vecteur Z et son influence sur la génération en n'utilisant plus `generate_latent_point` mais en fabriquant le tableau `pts` avec d'autres méthodes. Des exemples, certains animés, sont visibles dans [le dossier Z-anim](https://github.com/kaugrv/models_words/tree/main/Z-anim). Ici les valeurs ont été affichées pour donner une grille sous la forme d'une _heatmap_, on remarque d'ailleurs la relativité de l'échelle des couleurs : 

![anim1](https://user-images.githubusercontent.com/103901906/179784656-8ac40368-075b-42d3-b950-da614857c4dc.gif)


* générer en nombre des images par paires _non interpolées_. Le nombre de paires est paramétrable avec `nb_inf`
* exporter le vecteur Z associé à une paire de deux images dans un fichier texte. Ainsi, il sera possible d'importer les coordonnées d'un vecteur dans l'_Inference_ et donc d'obtenir de nouveau l'image associée (éventuellement les interpoler, les étudier...) en utilisant les _arrays_ exportés dans _Vec##.txt_


## Super-résolution

À l'issue de l'inférence, nous avons choisi d'améliorer la résolution de nos images interpolées, toujours grâce à un algorithme (écrit par Adrish Dey[^4]). La partie dédiée _Super Resolution_ se trouve dans le Notebook de l'Inference (et Inference +), à la suite de l'interpolation, pour être appliquée directement sur les images obtenues.

On passe ainsi d'une image de 128x128 à une image de 512x512 (on multiplie les dimensions par 4) :

![Image originale](https://user-images.githubusercontent.com/103901906/177224935-3e7ec9c7-af83-490f-a78c-91088bfbff76.png) ![Super-résolution](https://user-images.githubusercontent.com/103901906/177224950-3936d167-81d9-44ca-9824-932b2aabdeb5.jpg)

L'algorithme s'applique sur toutes les images obtenues par l'interpolation. La dernière cellule permet de télécharger ces images, en grande résolution, dans un fichier zip.

![Animation Super-résolution](https://user-images.githubusercontent.com/103901906/177619339-3cf28dfd-ff00-4761-a659-66a02d5e5abe.gif)

[^4]: Voir sur Tfhub : [esrgan by captain-pool](https://tfhub.dev/captain-pool/esrgan-tf2/1)

## Résultats

Pour conclure sur cette étude, nous pouvons observer les différences esthétiques entre les images de notre dataset synthétique _lines_ contenant 10 000 images, et les images et animations finales obtenues à travers l'oeil du GAN utilisé.

Si les images d'origine contenaient des lignes nettes, unicolores, toujours au nombre de 3 (deux horizontales et une verticale), les images obtenues au cours de l'entraînement du GAN font apparaître plus ou moins de lignes (d'une seule, à parfois 4 ou 5) - il y a donc introduction d'une nouveauté géométrique. Une impression de profondeur, de différents plans superposés, absente dans le dataset, semble également apparaître dans l'entraînement : des lignes semblent cachées dans l'obscurité et d'autres se trouver au premier plan.

Certaines lignes semblent briller, évoquant des raies spectrales, et leur texture, légèrement bruitée et caractéristique des images générées par IA, apparaît, renforcée par la super-résolution. De nouvelles couleurs apparaissent (du orange, du bleu clair, du beige) ainsi que des dégradés, mais "tous" les mélanges ne se font pas (le bleu et le jaune originaux auraient pû donner du vert, ce qui n'a jamais été observé dans nos entraînements). Malgré ces innovations esthétiques itnroduites par le réseau, certaines demeurent très proches de celles du dataset.

![image\_0\_00](https://user-images.githubusercontent.com/103901906/177868869-e3375d52-a76c-4ecc-9f12-1cf25d79e345.png) ![image\_39\_00](https://user-images.githubusercontent.com/103901906/177869223-8f200f12-21a9-4187-a36a-fd495b5e185f.png) ![image\_40\_00](https://user-images.githubusercontent.com/103901906/177869230-faa33f57-f0f3-4f3a-b70b-552fc22f6b82.png) ![image\_47\_01](https://user-images.githubusercontent.com/103901906/177869257-728d5fd4-2fc4-4b77-84ec-822620a7bbef.png) ![image\_16\_00](https://user-images.githubusercontent.com/103901906/177869431-e225bcbe-8788-4d3b-bbb6-e09d68466b80.png) ![image\_0\_01](https://user-images.githubusercontent.com/103901906/177869460-96cbca56-69e7-48df-a570-9a0904b2825e.png)

D'un point de vue esthétique, ces observations sont très intéressantes, de même que les résultats obtenus par interpolation et les animations créées. Des modifications sur notre dataset (couleurs, géométries, répartitions en probabilités...) pourraient donner d'autres résultats intéressants, de même que d'autres datasets synthétiques que nous engageons désormais vivement l'utilisateur à créer et utiliser !

## Ressources

Basé sur [davidADSP/GDL\_code](https://github.com/davidADSP/GDL\_code)\
Forked depuis [leogenot/DeepDrawing](https://github.com/leogenot/DeepDrawing)

* [_Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play,_ David Foster](https://www.amazon.fr/Generative-Deep-Learning-Teaching-Machines/dp/1492041947)
* [Keras](https://keras.io/api/)
* [Tensorflow](https://www.tensorflow.org)
