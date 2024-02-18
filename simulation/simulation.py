import pandas as pd
from .utils.data_utils import (
    read_movement_data,
    compress_by_day,
    compress_self_loop,
    keep_departments_of_interest,
)
from .patient import Patient
from .department import Department


class Simulation:
    """
    Network ABM simulation for HAI transmission.
    """

    def __init__(
        self,
        movement_data_path: str,
        cleaned: bool,
        initial_patients: dict[str, list[Patient]] | None,
        initial_info: dict[str, dict] | None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        movement_data_path : str
            Path to raw movement data csv.
        cleaned : bool
            _description_
        initial_patients : dict[str, list[Patient]]
            _description_
        initial_info : dict[str, dict]
            _description_
        """

        self.movements  = self.import_data(self, movement_data_path, cleaned=cleaned)

        self.node_names = self.movements.from_department.unique()
        self.node_names = [name for name in self.node_names if name != 'ADMISSION']

        if initial_patients and initial_info:
            assert set(self.node_names) == set(initial_patients.keys())
            assert set(self.node_names) == set(initial_info.keys())
            self.nodes = {name: Department(name, initial_patients[name], initial_info[name]) for name in self.node_names}
        else:
            self.nodes = {name: Department(name, [], {'β': 0.}) for name in self.node_names}

    @staticmethod
    def import_data(self, movement_data_path: str, cleaned: bool) -> pd.DataFrame:
        """
        If the data is not cleaned, preprocess data and save it in a cleaned version.
        If the data is cleaned, import data directly.
        Can be called static to clean data.

        Parameters
        ----------
        movement_data_path : str
            _description_
        cleaned : bool
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """

        if cleaned:
            mvt = pd.read_csv(movement_data_path)
        else:
            # preprocessing steps
            mvt = read_movement_data(movement_data_path)
            mvt = compress_by_day(mvt)
            mvt = compress_self_loop(mvt)
            mvt = keep_departments_of_interest(mvt)
            mvt = mvt.sort_values(by="date")

            mvt.to_csv(movement_data_path[:-4] + "_cleaned.csv")

        return mvt

    def export_checkpoint(self):
        patients = {name: self.nodes[name].patients for name in self.node_names}
        info     = {name: self.nodes[name].info     for name in self.node_names}
        return patients, info
