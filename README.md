# system-d-indexation-d-images

![image](https://github.com/user-attachments/assets/5bc38821-2c66-469b-83cb-c225e907af4b)
![image](https://github.com/user-attachments/assets/7a01ecbc-5b19-4ed0-9294-2e8cca734bd2)
![image](https://github.com/user-attachments/assets/3248dde4-a726-4db0-a6be-cbcb7157aa18)

Système d’indexation d’images par similarité sur un dataset de fruits. 5 descripteurs (histogrammes, corrélogramme, VGG16, AlexNet) au choix. Interface Jupyter interactive. Prédit les 5 images les plus proches d’une image requête via modèles TensorFlow
---
##Fonctionnalités Principales
🖼️ Méthodes d'Indexation

1/Histogramme couleur (3D RGB)

```
def indexer_images_couleur(chemin_repertoire):
    index = {}
    for nom_fichier in os.listdir(chemin_repertoire):
        chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)
        if chemin_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(chemin_fichier)
            histogramme = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            index[nom_fichier] = histogramme.flatten()
    return index
```
![image](https://github.com/user-attachments/assets/97767d9c-ac48-4262-a387-8b2dd742b9b1)

2/Histogramme niveaux de gris

```
def indexer_images_gris(chemin_repertoire):
    index = {}
    for nom_fichier in os.listdir(chemin_repertoire):
        chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)
        if chemin_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(chemin_fichier, cv2.IMREAD_GRAYSCALE)
            histogramme = cv2.calcHist([image], [0], None, [8], [0, 256])
            index[nom_fichier] = histogramme.flatten()
    return index
```
![image](https://github.com/user-attachments/assets/08df65dd-7792-4911-bd54-0f838d91ea6a)

3/Corrélogramme (texture)

```
def calculer_correlogramme(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return correlation
```
![image](https://github.com/user-attachments/assets/a18eecd2-25db-497b-a4dc-98906645529e)

4/Features profondes (VGG16)

```
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
# Charger le modèle VGG16 pré-entraîné sans la couche de classification (top)
base_model = VGG16(weights='imagenet', include_top=False)
model_vgg16 = Model(inputs=base_model.input, outputs=base_model.output)
# Fonction pour extraire les caractéristiques d'une image avec le modèle VGG16
def extraire_features_vgg16(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model_vgg16.predict(img_array) 
    return features.flatten() 

```
![image](https://github.com/user-attachments/assets/452144c5-df3a-4e8b-b035-59ac8d342e1a)
![image](https://github.com/user-attachments/assets/fece9c73-876f-4c03-bd62-390c8845ea34)

5/Features profondes (AlexNet)

```
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
# Charger le modèle AlexNet pré-entraîné sans la dernière couche de classification
model_alexnet = models.alexnet(pretrained=True)
model_alexnet.classifier = torch.nn.Sequential(*list(model_alexnet.classifier.children())[:-1])  # Retirer la dernière couche
model_alexnet.eval()  # Mode évaluation pour désactiver Dropout etc.
# Fonction pour extraire les features avec AlexNet
def extraire_features_alexnet(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # AlexNet attend 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Ajouter dimension batch

    with torch.no_grad():
        features = model_alexnet(img_tensor)
    return features.numpy().flatten()
```
![image](https://github.com/user-attachments/assets/ec291842-1a0c-4bac-a8c4-86ed5f5c9533)

---
##🔍 Recherche d'Images Similaires
Comparaison par distance euclidienne et Retour des 5 images les plus similaires
```
#prenons le cas dans l'histogramme de couleur
def rechercher_images_similaires_couleurs(index, histogramme_requete, nombre_resultats=5):
    distances = []
    for nom_fichier, histogramme_indexe in index.items():
        distance = np.linalg.norm(histogramme_requete - histogramme_indexe)
        distances.append((nom_fichier, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:nombre_resultats]

```
## 🛠Technologies Utilisées

🔧Backend

OpenCV - Traitement d'images
NumPy/SciPy - Calculs scientifiques
scikit-image - Analyse de texture
TensorFlow/Keras - Modèles de deep learning
PyTorch - Modèles pré-entrainés

💻Interface

Matplotlib - Visualisation des résultats
Google Colab - Interface interactive

![deepseek_mermaid_20250504_c483f0](https://github.com/user-attachments/assets/1936984f-223c-4b4f-97d4-717247850c57)

---
## 📁 Structure du projet 

```
📁 system-d-indexation/
│
├── 📂 fruits/                  # Dossier des catégories de fruits
│   ├── 🍎 Apple_Red_1/        # Sous-dossier pour les pommes
│   ├── 🥑 avocado/            # Sous-dossier pour les avocats  
│   └── 🍌 banana/             # Sous-dossier pour les bananes
│
├── 🖼️ Images/                 # Dossier principal des images
│   ├── 🍎 apple.jpg           # Image de pomme
│   ├── 🥑 avocadoo.jpg        # Image d'avocat (à renommer)
│   ├── 🍌 banana.jpg          # Image de banane
│   ├── 🍌 banana2.jpg         # Variante de banane
│   └── 🔍 request.jpg         # Image test pour requêtes
│
├── 📊 Notebooks/
│   └── 🐍 proj.ipynb          # Notebook Jupyter principal
│
├── 📝 Documentation/
│   └── 📄 README.md           # Documentation du projet
│
└── 🗑️ ipynb_checkpoints/      # Dossier cache Jupyter (à ignorer)

```
---
## 📦 Installation et Utilisation du Projet
⚙️ Installation

Prérequis:
Python 3.8 ou supérieur (python --version)
pip (gestionnaire de paquets Python)
Git (pour cloner le dépôt)

1. Cloner le dépôt
```
#bash
git clone https://github.com/ayoubjemaa/system-d-indexation.git
cd system-d-indexation
```
2. Créer un environnement virtuel (recommandé)
```
#bash
python -m venv venv
```
Activation :
Linux/Mac :
```
#bash
source venv/bin/activate
```
Windows :
```
#cmd
venv\Scripts\activate
```
3. Installer les dépendances
```
#bash
pip install opencv-python numpy scikit-image tensorflow torch matplotlib jupyter
````
🚀 Utilisation

1. Lancer Jupyter Notebook
```
#bash
jupyter notebook
```
Ouvrir proj.ipynb pour exécuter le système d'indexation.
2. Charger des images
Placer vos images dans :
📁 system-d-indexation/
└── 📂 Images/
    ├── 🍎 apple.jpg
    ├── 🥑 avocado.jpg
    └── 🍌 banana.jpg
3. Exécuter une recherche
Dans le notebook :
1.Choisir un descripteur (couleur, texture, deep learning).
2.Indexer un dossier :
```
#python
index = indexer_images_par_type("chemin/vers/images", "vgg16")
```
3.Rechercher des images similaires :
```
#python
resultats = rechercher_images_similaires_par_type("chemin/image.jpg", "vgg16", index)
````

🖥️ Interface (Option Google Colab)
Ouvrir Google Colab.
Importer proj.ipynb.
Monter Google Drive (pour accéder aux images) :
#python
```
from google.colab import drive
drive.mount('/content/drive')
````

📌 Exemple de Commande Complète
````
bash
# Dans le terminal
git clone https://github.com/ayoubjemaa/system-d-indexation.git
cd system-d-indexation
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install opencv-python numpy tensorflow
jupyter notebook
````
---
## 📜 Licence
MIT © Ayoub Jemaa
License




