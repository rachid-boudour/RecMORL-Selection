import xml.etree.ElementTree as ET
from src.modules.Service import Service
import numpy as np
from itertools import permutations
from src.modules.commun import Constant


class Composition:
    """
    This class represents a Composition that containes n services
    """

    def __init__(self, services: list[Service]):
        """
        constructor method of class Composition
        Args:
            id (int): representing the Composition's ID.
            servicesIds (list): A list of service IDs in the composition.
            servicesClouds (list): A list of cloud services, presumably corresponding to the service IDs.
        """
        self.__CompositionServices = services
        self.__number_services_compos = len(services)

    def calculate_score_script(self) -> np.ndarray[float]:
        """
        This function calculates the QoS score (accumulative reward) for a composition
        Warning : DONT USE IT IN THE ENV WHERE REWARD IS CALCULATED simultaneously
                only use it with scripts
        Return : the the QoS score (accumulative reward) for a composition
        """
        composition_score = np.zeros(Constant.number_objectives)
        for service in self.__CompositionServices:
            composition_score += service.getrewardVect()
        # Add the last QoS (number of clouds in a composition)
        self.calculate_reward_numberClouds(composition_score)
        return composition_score

    def calculate_reward_numberClouds(self, reward_vec: np.ndarray[float]) -> None:
        """
        This function return a reward for number of clouds objective
        if all services are from the same clouds, reward = number of services in the composition
        if all services are from diffrent clouds, reward = 0
        """
        # This scaling function is used : scaled_number = (number - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        reward_vec[-1] = (
            -1 * (self.numberClouds_composition()) + self.__number_services_compos + 1
        )

    def numberClouds_composition(self) -> int:
        """
        This function give the number of clouds existing in the composition
        Returns:
            number of clouds in the composition
        """
        clouds = []
        for service in self.__CompositionServices:
            clouds.append(service.get_cloud_id())
        return len(set(clouds))

    def display_composition(self) -> dict:
        """
        This function print the content (id + QoS scores) of the services in the composition:
        """
        i = 1
        composition_data = {}
        services = []
        for service in self.__CompositionServices:
            service_data = service.display_service()
            services.append(service_data)
            i += 1
        composition_data["services"] = services
        composition_data["nb_services"] = len(self.__CompositionServices)
        composition_data["nb_clouds"] = self.numberClouds_composition()
        return composition_data

    def calculate_sum_QoS_composition(self) -> dict:
        """
        This function print the content (id + QoS scores) of the services in the composition:
        """
        i = 1
        composition_sum = {}

        for service in self.__CompositionServices:
            service_data = service.display_service()
            for key, value in service_data:
                for QoS_elem, QoS_score in value:
                    composition_sum[f"QoS_elem"] += QoS_score

        composition_sum["nb_clouds"] = self.numberClouds_composition()

    def get_serviceInComp(self, service_id: int) -> Service | None:
        """
        This function returns a service from the composition that has a giving id
        Aergs:
            service_id: the id of the service we are looking for
        Returns : the service object tha has that id
        """
        for service in self.__CompositionServices:
            if service.get_serviceId == service_id:
                return service
        print(f"the is no service with id = {service_id}")
        return None
