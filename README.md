# projet-sda-machine-learning
## Installation 
### Prerequisites
- Python 3.8 or higher 
- pip (Python package installer)

### Clone the Repository
    git clone https://github.com/mriusero/projet-sda-machine-learning
    cd projet-sda-machine-learning

## Create and Activate Virtual Environment (Optional but Recommended)

### On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

### On Windows
    python -m venv venv
    venv\Scripts\activate

## Install Dependencies
    pip install -r requirements.txt

## Running the Application
To start the Streamlit application, run the following command from the root directory of the project:
 
    cd app 
    streamlit run app/app.py

## Project Details

- **app.py:** The main entry point of the Streamlit application.
- **dashboard/:** Contains the components, layout, and content for the dashboard.
- **data/:** Contains the input data files, exploratory notebooks, and preprocessed data outputs. 


#### Tree
    .
    ├── testing_data
    │   └── sent_to_student
    │       ├── group_0
    │       │   ├── Sample_submission.csv
    │       │   ├── testing_data.rar
    │       │   └── testing_item_(n).csv  // n = 50 files
    │       ├── scenario_0
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_1
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_2
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_3
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_4
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_5
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_6
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_7
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_8
    │       │   └── item_(n).csv        // n = 11 files
    │       ├── scenario_9
    │       │   └── item_(n).csv        // n = 11 files
    │       └── testing_data_phase_2.rar
    │
    └── training_data
        ├── create_a_pseudo_testing_dataset.ipynb
        ├── degradation_data
        │   └──item_(n).csv            // n = 50 files
        ├── failure_data.csv
        ├── pseudo_testing_data
        │   └── item_(n).csv            // n = 50 files
        └── pseudo_testing_data_with_truth
            ├── Solution.csv
            └── item_(n).csv            // n = 50 files


    18 directories, 326 files


### **2. Données d'Entraînement**

- **URL :** [training_data]. 

- **Nombre d'Échantillons :** 50 échantillons au total.
- **`failure_data.csv` :**
  - Contient un résumé des temps jusqu'à la panne pour les 50 échantillons.
  - Indique le mode de défaillance pour chaque échantillon (Infant Mortality, Fatigue Crack, ou Control Board Failure).

- **Dossier `degradation_data` :**
  - Contient les mesures de la longueur de fissure pour les 50 échantillons.
  - Chaque fichier CSV dans ce dossier correspond à un échantillon spécifique.
  - Le nom du fichier `item_X` correspond à l'identifiant de l'échantillon (`item_id`) dans le fichier `failure_data.csv`.


- **Notebook `create_a_testing_dataset.ipynb` :** 
  - Un notebook pour générer un "pseudo testing dataset" à partir des données d'entraînement.
  - Ce notebook montre comment les données sont générées et peut être utilisé pour développer votre modèle en utilisant des données pseudo-test.

### **3. Données de Test et Évaluation**
- **Lien vers les données de test : `testing_data/sent_to_student/group_0`** 
 

- **Fichier `Sample_submission.csv` :**
 fichier modèle pour soumettre les résultats.


### **Résumé**

1. **Données d'Entraînement :** des fichiers de mesure de dégradation et des informations sur les pannes.

2. **Notebook :** Outil pour générer des données pseudo-test pour aider à la préparation de votre modèle.

3. **Données de Test :** Disponibles pour évaluation et un modèle de soumission est fourni pour structurer les résultats.

