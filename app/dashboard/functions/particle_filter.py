import numpy as np
import pandas as pd

class ParticleFilter:
    def __init__(self, num_particles=1000, process_noise=None, measurement_noise=0.05):
        self.num_particles = num_particles
        self.process_noise = process_noise or [0.01, 0.001, 0.01]
        self.measurement_noise = measurement_noise

    def initialize_particles(self, beta0_range, beta1_range, beta2_range):
        particles = np.zeros((self.num_particles, 3))
        particles[:, 0] = np.random.uniform(*beta0_range, self.num_particles)
        particles[:, 1] = np.random.uniform(*beta1_range, self.num_particles)
        particles[:, 2] = np.random.uniform(*beta2_range, self.num_particles)
        return particles

    def propagate_particles(self, particles):
        noise = np.random.normal(0, self.process_noise, particles.shape)
        particles += noise
        return particles

    def compute_weights(self, particles, observation, time):
        weights = np.zeros(self.num_particles)
        for i, particle in enumerate(particles):
            beta0, beta1, beta2 = particle
            predicted_observation = beta2 / (1 + np.exp(-(beta0 + beta1 * time)))
            weights[i] = np.exp(-0.5 * ((observation - predicted_observation) ** 2) / self.measurement_noise ** 2)
        weights /= np.sum(weights)  # Normaliser les poids
        return weights

    def resample_particles(self, particles, weights):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        return particles[indices]

    def filter(self, df, beta0_range, beta1_range, beta2_range):

        df = df.rename(columns={'crack length (arbitary unit)': 'length_measured'})

        filtered_data = []

        # Traiter chaque 'item_index' individuellement
        for item_index in df['item_index'].unique():
            df_item = df[df['item_index'] == item_index]
            particles = self.initialize_particles(beta0_range, beta1_range, beta2_range)

            for index, row in df_item.iterrows():
                time = row['time (months)']
                observation = row['length_measured']

                particles = self.propagate_particles(particles)
                weights = self.compute_weights(particles, observation, time)
                particles = self.resample_particles(particles, weights)

                # Estimation de l'état
                estimated_state = np.mean(particles, axis=0)
                estimated_crack_length = estimated_state[2] / (1 + np.exp(-(estimated_state[0] + estimated_state[1] * time)))

                # Ajouter les données filtrées à la liste
                filtered_data.append(row.tolist() + [estimated_crack_length] + list(estimated_state))

        # Créer un DataFrame avec les résultats filtrés
        column_names = df.columns.tolist() + ['length_filtered', 'beta0', 'beta1', 'beta2']
        filtered_df = pd.DataFrame(filtered_data, columns=column_names)
        print('Noise filtered !')
        return filtered_df