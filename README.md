# Clustering-Hybride-Distribue
Implémentation en Python d’un algorithme de classification non supervisée combinant K-means et K-medoids dans un environnement distribué, alliant les avantages des deux méthodes.

# Fonctionnalités
- Possibilité d'exécuter trois types d'algorithmes de clustering : K-means, K-medoids ainsi qu'un algorithme hybride distribué.
- Comparaison des algorithmes en termes de temps d’exécution et de précision (accuracy).

# Structure du projet
- data/ : Contient les fichiers de données CSV à utiliser pour le clustering.
- main.py : Point d’entrée du projet avec l’interface graphique.
- clustering.py : Contient l'implémentation des fonctions de clustering K-means, K-medoids et des algorithmes hybrides distribués.

# Prérequis
- Python version 3.x
- Les blibliothèques pandas, numpy, matplotlib et sklearn.

# Note
- Pour exécuter le projet, saisissez la commande python `main.py` dans votre terminal.
- La première méthode hybride divise les données en partitions, utilise K-means pour obtenir des centroïdes locaux, sélectionne les points réels les plus proches (pseudo-médoïdes) puis regroupe ces médoïdes et applique K-médoïde final pour un clustering précis et optimisé.
- La deuxième méthode hybride divise d'abord les données en plusieurs partitions puis applique l'algorithme K-Medoids sur chaque partition pour obtenir des médianes. Ensuite, ces médianes sont regroupées et un K-Means est exécuté pour réaffecter les points aux clusters finaux, permettant ainsi d'obtenir une étiquette de clustering hybride pour chaque élément du jeu de données.