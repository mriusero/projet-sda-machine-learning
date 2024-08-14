import streamlit as st


def page_0():
    context = """ 
# Phase I : Remaining Useful Life (RUL)_

## I.1) Context
The purpose consist to predict remaining useful life (RUL) of an industrial robot based on monitored data from three failure modes.
The test dataset consisting of 50 robots, each measured for crack growth at a monthly frequency. The purpose consist to predict the remaining useful life (RUL) of each robot in the test set and decide whether the robot will survive the next 6 months.

Based on historical knowledge, the robot has three main failure modes:

- **Infant Mortality** : *Failures due to manufacturing defects that occur very early in the robot’s life.*
- **Failure of Control Boards** : *Random failures described by a probabilistic distribution.*
- **Fatigue Crack Growth** : *Failures due to the growth of cracks over time. (measured every month).*

**Target:** Predict if each robot can survive the next 6 months
"""
    st.markdown(context)

    knowledges = """
## I.2) Knowledges
### a) Physical model for the crack growth process

On suppose que le processus de croissance des fissures suit une fonction logistique :

$$
y(t) = {β2} / {1 + e^{-(β0 + β1 t)}}
$$

où $y(t)$ est la longueur de la fissure en unités arbitraires, $t$ est le temps en mois et $β0$, $β1$, $β2$ sont des paramètres liés aux propriétés du matériau.

Une défaillance se produit lorsque

$$
y(t) > y_{th}
$$

où
$y_{th} = 0.85$.
"""
    st.markdown(knowledges)

    process_noise = """
### b) Process noise, observation noise, and state space models


En raison du bruit de processus, les paramètres $β0$, $β1$, $β2$ peuvent légèrement varier au fil du temps. De plus, en raison des limitations de l'équipement de mesure, la longueur de fissure mesurée est affectée par un bruit de mesure important.

La longueur de fissure est mesurée tous les $1$ mois. Le modèle d'espace d'état suivant est utilisé pour capturer l'incertitude du processus et de l'observation :

$$
z_k = y_k + \epsilon_k
$$

where

$$
y_k = {β{2,k}} / {1 + e^{-(β{0,k} + β{1,k} t_k)}}
$$

and

$$
β{2,k} = β{2,k-1} + \omega_{2,k-1}
$$

$$
β{1,k} = β{1,k-1} + \omega_{1,k-1}
$$

$$
β{0,k} = β{0,k-1} + \omega_{0,k-1}
$$

#### Observation Noise:

$$
\epsilon_k \sim \text{Normal}(0, 0.05)
$$

#### Process Noise:

$$
\omega_{2,k-1} \sim \text{Normal}(0, 0.01)
$$

$$
\omega_{1,k-1} \sim \text{Normal}(0, 0.001)
$$

$$
\omega_{0,k-1} \sim \text{Normal}(0, 0.01)
$$
"""
    st.markdown(process_noise)

    prediction = """
### c) Prediction and metrics

Le participant doit prédire si le RUL d'un élément est inférieur à $6$ mois :

- Étiquette = $1$, si RUL ≤ 6  (signifie que le robot échouera dans les $6$ mois suivants)
- Étiquette = $0$, sinon. (signifie le contraire)  

Si l'étiquette prédite correspond à la vérité terrain, une récompense de $2$ sera attribuée.

Si elle ne correspond pas, alors :

- Une pénalité de -$4$ sera attribuée, si la vérité est $1$ et la prédiction est $0$ ;
- Une pénalité de -$1/60 \times \text{true\_rul}$ sera attribuée, si la vérité est $0$ et la prédiction est $1$.

La métrique d'évaluation est calculée comme suit :

$$
\text{perf} = \sum_{i=1}^{n} \text{Reward}_i
$$

où $\text{Reward}_i$ est calculé comme mentionné précédemment.

"""
    st.markdown(prediction)

    col1, col2 = st.columns(2)

    with col1:
        input_data = """
### # Input data_  
    .
    ├── testing
    │   └── sent_to_student
    │       ├── group_0
    │       │   ├── Sample_submission.csv
    │       │   ├── testing.rar
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
    └── training
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
        st.markdown(input_data)

    with col2:
        train_test = """
### # Training_
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
    
    """
        st.markdown(train_test)

        testing = """
    ### # Testing_
        Pour évaluer la performance du modèle, un jeu de données de "pseudo-test" basé sur le jeu de données de training permet d'évaluer la performance de prédiction.
        
        `training/pseudo_testing_data`  
            --> un jeu de données spécifiquement conçu pour évaluer les performances du modèle de manière similaire à un test. 
        
        `training/pseudo_testing_data_with_truth`  
            --> un jeu de données similaire à pseudo_testing_data, mais inclut également les valeurs réelles pour évaluer les prédictions.  
            --> les fichiers dans ce dossier incluent Solution.csv, qui contient les vérités de terrain pour les prévisions de RUL.
        
        `testing`  
            --> contient des données de test qui sont utilisées pour évaluer la performance du modèle dans un contexte plus général.  
            --> Il y a différents sous-dossiers pour chaque scénario de test (scenario_0 à scenario_9) 
    """
        st.markdown(testing)

    synthesis = """\n
Le jeu de données de test dans le dossier `testing/group_0` est créé à partir des séquences complètes de fonctionnement,
jusqu'à la défaillance en les tronquant aléatoirement à un moment donné $t_end$. L'objectif est de prédire la RUL : Remaining Useful Life à partir de ce point $t_end$. 

La troncature est effectuée de la manière suivante :
- si le temps jusqu'à la défaillance est inférieur ou égal à $6$, nous conservons la séquence telle quelle.
- si le temps jusqu'à la défaillance est supérieur à $6$, elle est tronquée à un point temporel aléatoire $t_end$, généré à partir d'une distribution uniforme de [1, ttf-1].
        """
    st.markdown(synthesis)



    phase2 ="""
# Phase II : _

    """
    st.markdown(phase2)

