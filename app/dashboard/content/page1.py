import streamlit as st
from ..functions import load_training_data, plot_crack_length_by_item_with_failures
def page_1():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#1 Input Analysis_</div>', unsafe_allow_html=True)    
    tree = """
### # Input data_  
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
"""
    st.markdown(tree)      
    
    training_description = """
### # Training data_
    * failure_data.csv :  résumé des temps jusqu'à la panne pour les 50 échantillons.

        - Type int    --> item_id
        - Type int    --> Time to failure (months)
        - Type string --> Failure mode
        
        // Indique le mode de défaillance pour chaque échantillon :
        // 'Infant Mortality', 'Fatigue Crack', ou 'Control Board Failure'
    
    * Degradation data (a folder with x50 .csv files):  mesures de la longueur de fissure pour les 50 échantillons.  

        - Type int   --> time (months)  
        - Type float --> crack length (arbitary unit)  
        - Type int   --> rul (months)  

        // Chaque fichier CSV dans ce dossier correspond à un échantillon spécifique.
        // Le nom du fichier `item_X` correspond à l'identifiant de l'échantillon (`item_id`) dans le fichier `failure_data.csv`.

### # Testing data_
Pour évaluer la performance du modèle, un jeu de données de "pseudo-test" basé sur le jeu de données de training permet d'évaluer la performance de prédiction.

`training_data/pseudo_testing_data`  
    --> un jeu de données spécifiquement conçu pour évaluer les performances du modèle de manière similaire à un test. 

`training_data/pseudo_testing_data_with_truth`  
    --> un jeu de données similaire à pseudo_testing_data, mais inclut également les valeurs réelles pour évaluer les prédictions.  
    --> les fichiers dans ce dossier incluent Solution.csv, qui contient les vérités de terrain pour les prévisions de RUL.

`testing_data`  
    --> contient des données de test qui sont utilisées pour évaluer la performance du modèle dans un contexte plus général.  
    --> Il y a différents sous-dossiers pour chaque scénario de test (scenario_0 à scenario_9) 

Le jeu de données de test dans le dossier `testing_data/group_0` est créé à partir des séquences complètes de fonctionnement,
jusqu'à la défaillance en les tronquant aléatoirement à un moment donné $t_end$. L'objectif est de prédire la RUL : Remaining Useful Life à partir de ce point $t_end$. 

La troncature est effectuée de la manière suivante :
- si le temps jusqu'à la défaillance est inférieur ou égal à $6$, nous conservons la séquence telle quelle.
- si le temps jusqu'à la défaillance est supérieur à $6$, elle est tronquée à un point temporel aléatoire $t_end$, généré à partir d'une distribution uniforme de [1, ttf-1].
    
"""
    st.markdown(training_description)

    training_data = load_training_data()
    df_to_plot = training_data['combined_data']
    # Exemple d'utilisation
    fig = plot_crack_length_by_item_with_failures(df_to_plot)
    st.pyplot(fig)