import numpy as np
from src.modules.commun import Constant
from src.modules.Tools import Tools


class Service:
    """
    This class represents a service in the cloud / service composition
    """

    def __init__(
        self,
        id_ser: int,
        cloud_id: int,
        energy: float,
        cost: float,
        reliability: float,
        availability: float,
        response_time: float,
    ):

        self.__id_ser = id_ser
        self.__cloud_id = cloud_id
        self.__energy = energy
        self.__cost = cost
        self.__reliability = reliability
        self.__availability = availability
        self.__response_time = response_time

    def get_cloud_id(self) -> int:
        "getter for cloud id"
        return self.__cloud_id

    # A methode to give reward for agent for selectiong a specific service :
    def display_service(self) -> dict:
        print("SERVICE ID , TYPE :", self.__id_ser, type(self.__id_ser))
        service_data = {
            "service_id": self.__id_ser,
            "service_name": Constant.services_info[self.__id_ser]["name"],
            "cloud_id": self.__cloud_id,
            "energy": round(
                Tools.revert_value(self.__energy, str(self.__id_ser), "energy"), 2
            ),
            "cost": round(
                Tools.revert_value(self.__cost, str(self.__id_ser), "cost"), 2
            ),
            "response_time": round(
                Tools.revert_value(
                    self.__response_time, str(self.__id_ser), "response_time"
                ),
                2,
            ),
            "service_img": Constant.services_info[self.__id_ser]["photo"],
        }
        return service_data

    def get_serviceId(self):
        """
        getter that returns the id of the service
        """
        return self.__id_ser

    # A methode to give reward for agent for selectiong a specific service :
    def getrewardVect(self) -> np.ndarray[float]:
        """
        This function give reward for agent for selectiong a specific service in a specific cloud
        Returns:
            reward vector score for each QoS
        """
        return np.array(
            [
                round(self.__energy, 4),
                round(self.__cost, 4),
                round(self.__response_time, 4),
                round(self.__reliability, 4),
                round(self.__availability, 4),
                0,  # for number of clouds in composition QoS, which is not in this level yet, in level of service composition
            ]
        )
