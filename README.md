-----------------------------------

Des modèles et des mots
========================

***Des Modèles et des Mots* est une série de publications dédiée à un ensemble de conversations sur le Computer Art et l'intelligence artificielle éditées par Gaëtan Robillard. La série entend présenter des entretiens avec Margit Rosen, Frieder Nake, Jérôme Nika, Véra Molnar, et Kazushi Mukaiyama. Les textes s'accompagnent d'une série d'images et de la diffusion d'un modèle d'apprentissage profond.**

Edition ESIPE – formation IMAC (Université Gustave Eiffel).  
[Fr]|[En](https://github.com/robillardstudio/three-lines-in-latent-space/blob/main/README_EN.md)

-----------------------------------

**Modèle**

# Trois lignes dans un espace latent
  
Gaëtan Robillard et Wendy Gervais.

## Avant-propos

Le code présenté ici soutient un travail exploratoire sur les GANs (Generative Adversarial Networks, ou Réseaux antagonistes génératifs)[^21]. Ces modèles d'architecture sont établis depuis quelques années dans la recherche en informatique, employés pour produire des médias de synthèse à partir de large bases de données d'images. Ce répértoire github est pensé en tant qu'environnement de travail ou *framework*, base à compléter et à explorer, pour quiconque souhaiterait découvrir, façonner ou critiquer ce type de modèle dans un contexte de recherche et d'expérimentation visuelle.

Le plus souvent, les GANs sont entraînés à partir de grandes quantités de photographies ou de dessins produits à la main – des données observées dans le réel. Les images sont figuratives. Qu'en est-il de l'abstraction géométrique [^22]? Et qu'en est-il des images déjâ produites par un code ? Ici, nous nous intéressons à des données d'entrainement qui sont des données de synthèse – des données générées par un algorithme et par le truchement de variables aléatoires – en référence au champ du Computer Art[^1]. À ceci s'ajoute une recherche sur l'image en mouvement, telle qu'elle apparaît en potentiel dans l'exploration de l'espace latent[^2] d'un GAN. L'une des perspectives du modèle est bien la création de séquences d'images nouvelles, issues de fonctions de parcour dans l'espace ou la « vision » du modèle entraîné.

Principalement écrit en Python, décomposé en trois Notebooks distincts : Training, Inference, Inference+ -- le code est commenté de façon à guider le profane à travers les différentes partie du *framework*. Les trois Notebooks présentés plus bas ont été configurés sur [Google Colaboratory](https://colab.research.google.com/drive/12WCzKlR--V8E7HMZHJ89nobVDCknCKmE#scrollTo=C7vmECVpwSZm), (aucune installation requise), et sont également exploitables dans un environnement local [^23].

D'une façon générale, l'apprentissage profond est un sujet technique complexe – notamment du fait de la très grande dimension des réseaux de neurones (le terme de boîte noire est adapté pour désigner ce problème). Tout en réfléchissant à la lisibilité de ce type de modèle, c'est ici la recherche artistique et pédagogique qui doit être mise en avant. Autant que possible, les aspects visuels de la démarche sont présentés : les données d'entraînement, l'architecture du GAN et les résultats.

[^21]: Le modèle de GAN présenté ici a été adapté à partir de l'ouvrage _Generative Deep Learning_ de David Foster (O’Reilly).

[^1]: Le Computer Art est entendu ici comme un champ iconographique à part entière, caractérisable par de nombreuses références à l'abstraction géométrique, à l'art cinétique et à l'art conceptuel. Bien qu'il soit difficile de circonscrire ce champs, on peut caractériser le Computer Art par son inscription dans l'esthétique infromationnelle et générative issu de théorciens comme Max Bense et Abraham Moles. Sans s'y réduire totalement, l'emploi d'algorithmes générateurs de nombres pseudo-aléatoires est trait marquant de l'époque pionnière.

[^2]: L'espace latent est l'espace théorique contenant les "points", ou vecteurs, support des motifs récurrents interprétables par un réseau de type apprentissage profond génératif. Ces vecteurs ou inputs sont traités par le GAN afin de générer les images finales ou outputs.

[^22]: En faisant référence ici au _Projet Mondrian_ de Frieder Nake : « I had scanned all of the neoplastic images of Mondrian's, and had arranged them into a time sequence of Mondrian's painting. The idea then was to analyze them year by year, in all sorts of statistical directions. Out of this, I wanted to develop a predictory system that would predict and create the New York Boogie-Woogie! Outrageously optimistic! » Frieder Nake, « The Art of Being Precise: Frieder Nake in Conversation [With Margit Rosen] », ZKM, 2022. [https://www.youtube.com/watch?v=Z_pOiHX6HYE](https://www.youtube.com/watch?v=Z_pOiHX6HYE)

[^23]: Par exemple dans environnement comme Anaconda. Pour l'installation des dépendances, Cf. [Generative Deep Drawing](https://github.com/leogenot/DeepDrawing)

### Fondations

Cette démarche se fonde en partie sur une expérience pédagogique menée avec des étudiants ingénieurs en art et science qui a consisté à étudier le Computer Art à travers des artistes pionniers comme Frieder Nake, Véra Molnar, Georg Nees et Manfred Mohr. Après avoir choisi une oeuvre dans un corpus, les étudiants ont été amenés à penser et re-coder le répèrtoire visuel et algorithmique d'une oeuvre, de façon à constituer des données d'entraînement. Chaque groupe a alors entraîné un modèle avec ces données.

Dans cette démarche, deux processus génératifs sont enchassés : le premier pour générer des données de façon synthétique, le second pour explorer l'espace de représentation (l'espace latent) du réseau une fois celui-ci entrainé. Les données d'entraînements présentées ici – les « Trois lignes » – et leur méthode, sont directement issus des supports ayant guidé les étudiants dans la démarche.

De plus, la base algorithmique de notre modèle a été explorée lors du projet tuteuré _Generative Deep Drawing_ mené par des étudiants de deuxième année de la même formation. Deep Drawing est un projet d'exploration du dessins par l'intéraction avec un GAN (Cf. ressources).

-----------------------------------

## Données d'entraînement

« Trois lignes dans un espace latent » repose sur un ensemble de 10 000 images élémentaires et semblables, appelé « dataset », généré à l'aide d'un code Java écrit et exécuté dans l'environnement de programmation Processing. Le code permettant de générer ces images est disponible dans le [dossier « lines »](https://github.com/kaugrv/models_words/blob/main/lines/lines.pde).

Cet ensemble d'images constitue une seule et même classe. Chacune d'elles suit les mêmes règles de construction – elles sont détaillées ci-après, et est strictement dictée par l'execution du code. Cela en fait un dataset synthétique et algorithmique, comportant des images simples, chacune différente mais appartenant à un ensemble. Cette homogénéité relative sera utile pour entraîner notre réseau.

Chaque image de notre dataset est au format 128 par 128 pixels. Sur fond noir, on génère 3 lignes d'épaisseur 5 pixels. On distingue une ligne verticale et deux horizontales. Leur position dans l'image est définie de façon aléatoire (loi uniforme ou loi gaussienne, selon). Leurs couleurs sont également aléatoires (loi uniforme sur [0,3]). Chacune des 3 lignes peut prendre l'une des couleurs suivantes [^3] :

- blanc
- jaune
- rouge
- bleu

![Extrait du dataset](https://user-images.githubusercontent.com/103901906/179547588-a322ef87-3ee4-4d77-b573-b6f613bf541a.png)

Par calcul, on peut approcher le nombre d'images contenue dans la classe. Chacune des 3 lignes comporte une abscisse (si verticale) ou une ordonnée (si horizontale) comprise entre 0 et 128, et possède une couleur choisie aléatoirement parmi 4. Le nombre de lignes verticales et horizontales étant fixé, on a donc :

$$ \prod_{i=1}^{3} (4 \times 128) = (4 \times 128)^3 = 134217728 $$  

c'est-à-dire 134 millions d'images possibles. En générant 10 000 d'entre elles, on couvre 0.007% d'images de la classe – autant dire que nous sommes certains de générer 10 000 images uniques. Ce dataset sera chargé lors de l'entraînement, à l'aide de le la fonction `load_dataset` de [loaders.py](https://github.com/kaugrv/models_words/blob/main/utils/loaders.py).

[^3]: Les trois premières couleurs sont les couleurs primaires. Leurs valeurs numériques sont tirées du travail de Piet Mondrian (Cf. _Trafalgar Square_, Piet Mondrian, 1943 ; reproduction in _[Museum of Modern Art](https://www.moma.org/collection/works/79879)_).

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
 
La matrice de convolution choisie pour le générateur est de taille 10 x 10 :

`generator_conv_kernel_size=[10,10,10,10]`

La convolution est ainsi relativement grande, de façon à ce que le générateur réponde à la qualité abstraites des images en considérant à une échelle globale. En revanche, le critique est associé à une matrice de convolution de taille inférieure, en taille 5 x 5 :

`critic_conv_kernel_size=[5,5,5,5]`

Les images sont scorées par le critique via une convolution de niveau plus locale que celle du générateur. En somme, si le générateur tend à l'abstraction visuelle – la ligne, le critique porte son analyse dans les détails. Cette asymétrie macro-micro des matrices de convolution entre générateur et critique résulte d'une interrogation forte quant à la capacité du réseau en matière d'abstraction visuelle.

D'autres dimensions de la convolution pourraient être explorées en fonction de nouveaux datasets. Pour plus de détails sur la convolution, voir _[Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)_.

![Générateur](https://user-images.githubusercontent.com/103901906/179565557-e17c1acc-e9a2-48c1-aa5e-973d7caf03d2.png)

![Critique](https://user-images.githubusercontent.com/103901906/179565848-3a8334e6-1680-4085-9bad-cbcb81c8d142.png)

Ci-dessus : Représentations graphiques des couches constituant le générateur et le critique. Dans cette perspective, on observe l'articulation des couches et leurs dimensions successives, en condénsé (au centre) et en proportions réelles (en haut à droite). Rendu réalisé avec [NN-SVG](https://github.com/alexlenail/NN-SVG).

## Entraînement

------------------------------------

Notebook [Training](https://colab.research.google.com/drive/12WCzKlR--V8E7HMZHJ89nobVDCknCKmE#scrollTo=C7vmECVpwSZm) sur Google Colab.

------------------------------------

Après avoir cloné ce dépôt, on travaillera à sa racine : `/models_words`. Il est possible d'utiliser le dataset _lines_ décrit plus haut ou bien d'en utiliser un nouveau (en le chargeant directement via une cellule  dans l'environnement Colab ou en remplaçant le fichier `data.zip`). Au moins 10 000 images (format 128x128) sont conseillées. Selon la complexité des images, davantage de données peuvent être nécessaires.

Les premières cellules du Notebook servent à charger le modèle et les bibliothèques, puis à charger le dataset. Il est possible de configurer les paramètres de l'entraînement, notamment le numéro associé à l'entraînement (`RUN_ID`), ou le `BATCH_SIZE`, c'est-à-dire le nombre d'images du dataset présentées au réseau lors d'une itération de l'entraînement. Par défaut, le `BATCH_SIZE` est paramétré à 16. Augmenter cette taille rendra le temps de calcul plus long, mais permettra d'améliorer la précision du réseau (garder une puissance de 2, comme 32, 64 ou 128).

D'autres paramètres de l'entraînement sont modifiables :

- `EPOCHS` : le nombre de cycles d'expositions du réseau au dataset complet (un cycle est composé de n batchs). Pour ce modèle, le paramètre par défaut est de 1201 époques, mais selon la complexité des images à analyser, il peut être inférieur ou supérieur (jusqu'à 60 000 époques et plus).
- `PRINT_EVERY_N_BATCHES` : au lieu de rendre et d'enregistrer une image ou output à chaque époque, on peut choisir une autre fréquence. Les images générées se trouveront dans le dossier `run/gan`, dans un sous dossier correspondant au numéro de l'entraînement. Par exemple, pour `EPOCHS = 1201` et `PRINT_EVERY_N_BATCHES = 100`) on obtiendra :

![Exemple samples](https://user-images.githubusercontent.com/103901906/176936405-c4fbce75-1ece-419f-a47e-5bd0e4547ef4.png)

- `rows` et `columns` : au cours de l'entraînement le rendu est paramétré pour présenter une grille d'images générées par le réseau. Ces deux variables permettent de paramétrer la densité de la grille. Voici un exemple avec les valeur `rows = 5`, `columns = 5` et `rows = 2`, `columns = 2` :

![Exemple 5x5](https://user-images.githubusercontent.com/103901906/176937177-ef705ff3-603f-4fb1-9243-b15b1783b3a0.png) ![Exemple 2x2](https://user-images.githubusercontent.com/103901906/176937301-ef2df143-63bb-4dad-8ce9-86d777c364f6.png)

En sortie de l'entraînement, il est possible d'observer l'évolution de fonction de perte (*Loss*) du critique D et du générateur :

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

La progression de ces valeurs à travers l'entraînement peut également être observée sous forme graphique :

<img src="https://user-images.githubusercontent.com/103901906/177615266-41ff49a0-16de-42bf-be11-5ed55a44dcc8.png" alt="Graphe" width="400"></img>
<img src="https://user-images.githubusercontent.com/103901906/177614854-75bcd83a-f65f-429c-b229-ffcb1738e8f7.png" alt="Légende" width="200"></img>

On peut ainsi analyser la convergence des deux _loss_ : ici par exemple, la différence moyenne entre les images observées (_real_, issues des données d'entraînement) et synthétiques (_fake_, générées par le générateur) tend vers 0 à partir d'environ 1100 époques. À ce stade de l'entraînement les images générées et les données d'entraînement sont évaluées à égalité par le critique. L'image de ce graphe est enregistrée dans `RUN_FOLDER+"/images/Converge.png"`.

Dans le Notebook Colab, la cellule **Results** permet de télécharger l'ensemble des images issues de l'entraînement ; et le fichier generator.h5 – le modèle entraîné du générateur, indispensable pour la partie _Inference_. Notons qu'à la suite de l'entraînement, le critique n'a plus d'utilité [^31].

[^31]: Aussi curieux que cela puisse paraître, le critique n'a en effet de rôle que lors de l'entraînement. Il serait intéressant d'imaginer un nouvel emploi pour le critique qui est alors délaissé. Une piste de réflexion : dans *Artificial Aesthetics: a critical guide to media and design*, Lev Manovich et Emanuele Arielli proposent de considérer à égale importance la fonction analytique et générative d'un réseau de deep learning, la fonction analytique étant attribuée à une fonction d'évaluation esthétique des artefacts culturels.

## Inference

------------------------------------

Notebook [Inference](https://colab.research.google.com/drive/13g3rX2zgyxT5YKTZILBrISybmLJ4_pXi) sur Google Colab.

Pour utiliser le réseau entraîné, il faut se munir du fichier _generator.h5_, voir partie précédente.

------------------------------------

La partie _Inference_ se donne pour objectif de générer de nouvelles images par l'utilisation du modèle entraîné et par l'exploration de _l'espace latent_. Cet espace est un espace vectoriel (ici en dimension 100) représentant l'ensemble des informations interprétables par le modèle. L'idée générale de l'exploration est d'étudier les images que le réseau est désormais capable de générer.

En utilisant le générateur entraîné, il va être possible de générer de nouvelles images relevant de la même catégorie que les images générées par le réseau lors de la dernière époque de son entraînement. Il sera possible de les étudier et de les _interpoler_ entre elles (c'est-à-dire d'introduire des vecteurs intermédiaires entre des vecteurs choisis).

La première des fonctions, `generate_latent_points`, génère un ou plusieurs vecteurs 100 (on choisit le nombre de vecteurs à générer en paramétrant `nb_vec`). Chaque vecteur est déterminé par une fonction alétoire qui détermine 100 valeurs flottantes aléatoires comprises entre environ -3 et 3 (la distribution répond à une loi normale centrée ou *standard normal distribution*). Par défaut, on utilisera `latent_dim = 100` et `nb_vec`, et `nb_vec = 2`. Ensuite nous étudions l'interpolation entre deux vecteurs de façon à générer une séquence d'images.

La seconde fonction `interpolate_points` permet l'interpolation, c'est-à-dire de créer de nouveaux vecteurs situés entre les deux vecteurs crées dans la fonction précédente. Si par exemple on interpole deux vecteurs avec 20 autres (`nb_img = 20`), il sera possible de créer alors une animation continue avec les 22 images correspondantes à chacun de ces vecteurs. Pour cela, la fonction `plot_generated` génère, enregistre et affiche toutes ces images.

![Animation 1](https://user-images.githubusercontent.com/103901906/177205129-48acd30e-9a0b-4f2b-8450-ea58f21e3d83.gif)

En résumé, voici les trois étapes principales pour explorer et interpoler les points de l'espace latent et leurs images associées :

1. avec la fonction `generate_latent_points`, on génère un tableau `pts` de format (2,100) qui contient 2 vecteurs 100 ; le nombre de vecteurs est choisit avec `nb_vec`
2. avec la fonction `interpolate_points`, on interpole les deux premiers tableaux de `pts`, ce qui donne le nouveau tableau `interpolated`
3. avec la fonction `plot_generated`, on génère les images associées en passant le tableau `interpolated` en argument

On peut ensuite télécharger l'ensemble de ces images au format zip, à l'aide de la dernière cellule de la partie **Function**. La partie **Super Resolution** sera présentée plus bas.

## Inference +

------------------------------------

Notebook [Inference +](https://colab.research.google.com/drive/13g3rX2zgyxT5YKTZILBrISybmLJ4_pXi) sur Google Colab.

------------------------------------

Pour aller plus loin, le Notebook Inference +, propose quelques fonctionnalités supplémentaires :

- visualiser le vecteur Z (vecteur 100 ici) sous la forme d'un images de 10x10 carrés, avec la fonction `printZ` ; voici quelques exemples d'images produites par le générateur, accompagnées de la visualisation du vecteur Z associée :

<img src="https://user-images.githubusercontent.com/103901906/178315771-00b5226d-bb5b-4c88-a091-59c70135336a.png" width="85"></img>
<img src="https://user-images.githubusercontent.com/103901906/178315053-f114c50f-4e11-4750-9846-19002c30553a.png" width="128"></img>

<img src="https://user-images.githubusercontent.com/103901906/178315953-689183b3-15ca-497a-ba4b-5456342083c2.png" width="85"></img>
<img src="https://user-images.githubusercontent.com/103901906/178315944-5dab8c66-eaad-4872-b45b-38311e176ec4.png" width="128"></img>

<img src="https://user-images.githubusercontent.com/103901906/178316212-068b2a10-08f3-4601-a69c-dc89c242755b.png" width="85"></img>
<img src="https://user-images.githubusercontent.com/103901906/178316192-73627c6d-faf6-4a06-9f5a-6600c61e43e7.png" width="128"></img>

Attention, dans cette visualisation, l'échelle des couleurs est relative (un niveau de gris ne représentera pas la même valeur d'une visualisation à une autre).

- de là, étudier le vecteur Z et son influence sur les images produites en sortie, en laissant de côté la fonction initiale `generate_latent_point` (Cf. Inférence) mais en fabriquant un tableau (`pts`) avec d'autres méthodes ; les résultats d'une première étude en image fixe et en image animée sont disponibles dans le dossier `Z-anim`. Ici les valeurs ont été affichées pour donner une grille de proportion rectangulaire, accompagnées des valeurs numériques du vecteur (on remarque la relativité de l'échelle de gris) : 

![anim1](https://user-images.githubusercontent.com/103901906/179784656-8ac40368-075b-42d3-b950-da614857c4dc.gif)

Deux autres fonctionnalités ont été ajoutées pour :

- générer des images en quantité, et par paires _non interpolées_ ; le nombre de paires est paramétrable avec `nb_inf`
- exporter dans un fichier texte les deux vecteurs Z associés à une paire de deux images ; il est alors possible d'importer les coordonnées d'un vecteur en tant qu'input du modèle, obtenir ainsi de nouveau l'image associée au vecteur donné
  
Ces deux fonctionnalité sont pensées pour de futures recherches.

## Super-résolution

------------------------------------

Notebooks [Inference](https://colab.research.google.com/drive/13g3rX2zgyxT5YKTZILBrISybmLJ4_pXi) et [Inference +](https://colab.research.google.com/drive/13g3rX2zgyxT5YKTZILBrISybmLJ4_pXi)

------------------------------------

Pour augmenter la résolution de images interpolées, nous proposon d'utiliser un modèle de deep learning entraîné et distribué par Tensorflow [^4]. Dans les deux notebooks en question, la partie **Super Resolution** s'applique directement sur les images obtenues en sortie du modèle.

On passe ainsi d'une image de 128x128 à une image de 512x512 (la définition de l'image est multipliée par 4).

![Image originale](https://user-images.githubusercontent.com/103901906/177224935-3e7ec9c7-af83-490f-a78c-91088bfbff76.png)
![Super-résolution](https://user-images.githubusercontent.com/103901906/177224950-3936d167-81d9-44ca-9824-932b2aabdeb5.jpg)

L'algorithme s'applique sur toutes les images issue de l'interpolation. La dernière cellule permet de télécharger les super-résolutions, dans un fichier zip.

![Animation Super-résolution](https://user-images.githubusercontent.com/103901906/177619339-3cf28dfd-ff00-4761-a659-66a02d5e5abe.gif)

[^4]: Voir sur Tfhub : [esrgan by captain-pool](https://tfhub.dev/captain-pool/esrgan-tf2/1)

## Résultats

Il est intéressant de comparer les images issues des données d'entraînement et les images générées par le modèle. Cette comparaison révèle des différences esthétiques importantes. Si les images d'origine contenaient des lignes nettes, unicolores, et toujours au nombre de 3 (deux horizontales et une verticale), les images générées augmentent ou diminuent ce nombre pour parfois faire apparaître jusqu'à 4 ou 5 lignes. Il y a une nouveauté géométrique et topologique. De plus, se dégage des résultats une impression de profondeur, dûe notamment à des contrastes clairs/obcsurs, pourtant absents des données d'entraînement.

Certaines traces semblent plus lumineuses, évoquant des raies spectrales. Leur texture apparraît légèrement bruitée, une marque des processus de convolution. Ce grain est renforcé par la super-résolution. De nouvelles couleurs apparaissent (du orange, du bleu clair, du beige) ainsi que des dégradés, mais tous les mélanges possibles ne sont sont pas visibles (le bleu et le jaune originaux auraient pû donner du vert, ce qui n'a jamais été observé dans nos entraînements). Ces « innovations esthétiques » sont inhérentes à la forme et aux propriétés du réseau. Pour autant elles préservent des qualités dont le rapprochement avec les images d'origine est évident (verticalité, horizontalité, couleurs primaires, épaisseurs moyennes, ...).

![image\_0\_00](https://user-images.githubusercontent.com/103901906/177868869-e3375d52-a76c-4ecc-9f12-1cf25d79e345.png) ![image\_39\_00](https://user-images.githubusercontent.com/103901906/177869223-8f200f12-21a9-4187-a36a-fd495b5e185f.png) ![image\_40\_00](https://user-images.githubusercontent.com/103901906/177869230-faa33f57-f0f3-4f3a-b70b-552fc22f6b82.png) ![image\_47\_01](https://user-images.githubusercontent.com/103901906/177869257-728d5fd4-2fc4-4b77-84ec-822620a7bbef.png) ![image\_16\_00](https://user-images.githubusercontent.com/103901906/177869431-e225bcbe-8788-4d3b-bbb6-e09d68466b80.png) ![image\_0\_01](https://user-images.githubusercontent.com/103901906/177869460-96cbca56-69e7-48df-a570-9a0904b2825e.png)

Le rapport entre entre les premières images, celle des données d'entraînement et les secondes – les images générées par le modèle pose un certain nombre de questions sur la nature des images numériques, entre réprésentation et abstraction, entre originalité et reconstruction. L'espace latent du modèle de deep learning ouvre une voie pour formuler ces questions et explorer ce rapport par le travail de l'image en mouvement. 

## Références

David Foster, *Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play*, 2019. Code [(https://github.com/davidADSP/GDL\_code)](https://github.com/davidADSP/GDL\_code)

<!-- François Chollet, *Deep Learning with Python*, Manning Publications, Co. Shelter Island, NY, 2021. Code [https://github.com/fchollet/deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks) -->
