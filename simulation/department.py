from typing import Any
from patient import Patient


class Department:
    """
    A department object.
    """

    def __init__(self, name: str, patients: set[Patient], info: dict[str, Any]) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            Department name.
        patients : set[Patient]
            Set of patient instances.
        info : dict[str, Any]
            Department level informations, including parameters.
        """

        self.name = name
        self.patients = patients
        self.into = info
        self.records = []

    def get_num_pos(self) -> int:
        return sum([p.infected for p in self.patients])

    def get_num_neg(self) -> int:
        return sum([not p.infected for p in self.patients])

    def record(self, incidence: int) -> None:
        """
        Record prevalence, incidence, and num_patients of the department on a day.
        Incidence(number of new infections) needs to be provided and the others will be inferred.

        Parameters
        ----------
        incidence : int
            Number of new infections that day.
        """
        prevalence = self.get_num_pos()
        total = prevalence + self.get_num_neg()

        self.records.append(
            {"total": total, "prevalence": prevalence, "incidence": incidence}
        )
