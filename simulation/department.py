from typing import Any
from patient import Patient


class Department:
    """
    A department object.
    """

    def __init__(self, name: str, patients: list[Patient], info: dict[str, Any]) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            Department name.
        patients : list[Patient]
            Set of patient instances.
        info : dict[str, Any]
            Department level informations, including parameters.
        """

        self.name = name
        self.patients = patients
        self.info = info
        self.records = []
    
    def __str__(self) -> str:
        info_str = ', '.join(f"{key}: {value}" for key, value in self.info.items())
        return (
            f"Department Name: {self.name}\n"
            f"Number of Patients: {len(self.patients)}\n"
            f"Infected: {self.get_num_pos()}, Not Infected: {self.get_num_neg()}\n"
            f"Info: [{info_str}]"
        )

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
    
    def verbose_info(self) -> str:
        """_summary_

        Returns
        -------
        str
            _description_
        """
        

        patient_details = '\n'.join([' ' + str(patient) for patient in self.patients])
        
        print (
            '+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n'
            f"[Department]\n"
            f"Name: {self.name}\n"
            f"Total Patients: {len(self.patients)}\n"
            f"Patient Details:\n{patient_details}\n"
            f"Department Info: {self.info}\n"
            '---------------------------------\n'
        )
