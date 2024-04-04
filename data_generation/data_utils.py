import pandas as pd
import numpy as np

def remove_empty_rows(data:np.array):
    return data[~(data == 0).all(axis=1)]

def health_level_transform(data:np.array):
    status_mapping = {
        0.0: 0.0,    # Not in hospital
        1.0: -0.5,   # In hospital but not tested
        2.0: -1.0,   # Tested negative
        3.0: 0.5,    # Tested colonized
        4.0: 1.0   # Tested infected
    }
    transformed_data = np.vectorize(status_mapping.get)(data)
    return transformed_data


# data = np.load("/Users/richardzhuang/Desktop/UCSF/HAI_simulation/data_generation/generated_matrices/sw_3_1_full_1.npy")
# print(data.shape)
# print(remove_empty_rows(data).shape)

# print(data[0:5, 0:5])
# print(health_level_transform(data)[0:5, 0:5])