import numpy as np
import pandas as pd
import os
from simulation.simulation import Simulation

if __name__ == "__main__":

    movement_folder_path = "movement_generation/deid_data/"
    output_folder_path = "matrix_generation/"

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

            # second round to generate matrix
            simulation.init_condensed_matrix()
            simulation.simulate()

            # save the matrix
            full_matrix_path = output_folder_path + filename + '_full'
            np.save(full_matrix_path, simulation.real_CD)
            partial_matrix_path =  output_folder_path + filename + '_partial'
            np.save(partial_matrix_path, simulation.observed_CD)
