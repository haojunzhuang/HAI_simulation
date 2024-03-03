from typing import Any
from simulation.status import Status

class Patient:
    """
    A patient agent.
    """

    def __init__(self, id: str, info: dict[str, Any]) -> None:
        """_summary_

        Parameters
        ----------
            id (str): 
                Patient ID (e.g. deid_enc_id).
            info (dict[str, Any]): 
                Fixed patient informations like demographics.
        """

        self.id = id
        self.info = info
        self.status = Status.healthy
        self.symptom = 0
        # TODO: Just_Recovered?
        
    def __hash__(self):
        # Use the patient_id to generate a unique hash value for each Patient object
        return hash(self.id)

    def __eq__(self, other):
        # Two Patient objects are considered equal if their patient_id attributes are equal
        return self.id == other.id
    
    def __repr__(self) -> str:
        return self.verbose_info()
    
    def __str__(self) -> str:
        return self.verbose_info()

    def verbose_info(self) -> str:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'

        status_color = RED if self.infected else YELLOW if self.recovered else GREEN
        status = f"{status_color}{' infected' if self.infected else 'recovered' if self.recovered else '  healthy'}{RESET}"
        info_str = ', '.join(f"{key}: {value}" for key, value in self.info.items())
        return f"[Patient] - ID: {self.id}, Status: {status}, Info: [{info_str}]"
    
    def colonoize(self) -> None:
        """
        Colonize a patient.
        """

        self.status = Status.colonized

    def infect(self) -> None:
        """
        Infect a patient.
        """

        self.status = Status.infected

    def recover(self) -> None:
        """
        Recover a patient. Status changed to recovered and symptom to False.
        """

        self.status  = Status.recovered
        self.symptom = 0
    
    def test(self) -> Status:
        """
        Test the status of the patient.
        In the case of C. diff, either Toxin gene or protein can be found/not found.

        Returns
        -------
        Status
            The status of the patient. Can be healthy, colonized, or infected.
        """

        return self.status
    
    def develop_symptom(self) -> None:
        """
        Develop symptoms.
        """

        assert self.status == Status.infected, "Only infected patients can develop symptoms?"
        self.symptom += 1
        
    