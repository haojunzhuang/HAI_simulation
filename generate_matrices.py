import numpy as np
import os
from simulation.simulation import Simulation
from tqdm import tqdm

if __name__ == "__main__":

    movement_folder_path = "movement_generation/generated_movements"
    output_folder_path = "matrix_generation/"
    matrix_per_movement = 10

    for filename in os.listdir(movement_folder_path):
        if filename.endswith('.pkl'):
            movement_path = os.path.join(movement_folder_path, filename)
            
            # first round simulation to gather data
            simulation = Simulation(movement_data_path=movement_path, 
                                cleaned=True, initial_info=None, initial_patients=None,
                                uniform_alpha=0.20, uniform_beta=0.02, uniform_gamma=0.3,
                                uniform_delta=0.05, uniform_zeta=0.01, uniform_eta=0.005,
                                test=False,
                                )
            simulation.simulate()
            
            print(f'simulating {matrix_per_movement} rounds for {filename}...')
            # second round to generate matrix
            for i in tqdm(1, range(matrix_per_movement)+1):
                simulation.init_condensed_matrix()
                simulation.simulate()

                # save the matrix
                full_matrix_path = output_folder_path + filename + '_full_' + i
                np.save(full_matrix_path, simulation.real_CD)
                partial_matrix_path =  output_folder_path + filename + '_partial_'+ i
                np.save(partial_matrix_path, simulation.observed_CD)
        else:
            raise NotImplementedError()
