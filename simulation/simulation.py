import pandas as pd
from .utils.data_utils import read_movement_data, compress_by_day, compress_self_loop, keep_departments_of_interest

class Simulation:
    """
    Network ABM simulation for HAI transmission.
    """

    def __init__(self, movement_data_path: str, cleaned: bool, initial_status: list | None) -> None:
        """_summary_

        Parameters
        ----------
        movement_data_path : str
            Path to raw movement data csv.
        initial_status : list | None
            _description_
        """
        
        self.movements   = self._import_data(movement_data_path, cleaned=cleaned)
        self.nodes       = self.movements.from_department.unique()


    def _import_data(self, movement_data_path: str, cleaned: bool) -> pd.DataFrame:

        if cleaned:
            mvt = pd.read_csv(movement_data_path)
        else:
            # preprocessing steps
            mvt = read_movement_data(movement_data_path)
            mvt = compress_by_day(mvt)
            mvt = compress_self_loop(mvt)
            mvt = keep_departments_of_interest(mvt)
            mvt = mvt.sort_values(by='date')

            mvt.to_csv(movement_data_path[:-4]+'_cleaned.csv')

        return mvt

    def export_checkpoint(self):
        pass