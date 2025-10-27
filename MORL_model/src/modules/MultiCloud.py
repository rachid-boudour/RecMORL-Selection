import os
from src.modules.Cloud import Cloud
import xml.etree.ElementTree as ET
from src.modules.Service import Service
from src.modules.Compositon import Composition
from src.modules.commun import Constant


class MultiCloud:
    def __init__(
        self,
        serviceIds: list[int],
        multiCloud_dir: str,
    ):
        self.serviceIds = serviceIds
        self.multiCloud_dir = multiCloud_dir
        self.__clouds_list = self.get_Clouds_list()

    def getNumberClouds(self):
        """
        function that retgurns Number of clouds existing in the multiCloud
        """
        return len(self.__clouds_list)

    def get_Clouds_list(self) -> list[Cloud]:
        """
        This function takes a directory (the multicloud) of xml files (the clouds) to return the cloud instances .
        Args:
            directory (str): directory containing XML files
        Returns:
            List: A list with containing the clouds in the multicloud
        """
        cloud_instances = []
        print("\n\n", (self.multiCloud_dir), "\n\n")
        for filename in os.listdir(self.multiCloud_dir):
            if filename.lower().endswith(".xml"):
                file_path = os.path.join(self.multiCloud_dir, filename)
                cloud_instances.append(Cloud(file_path, serviceIds=self.serviceIds))
        return cloud_instances

    def get_CloudById(self, id: int) -> Cloud | None:
        """
        function that find a cloud in the multicloud by its id
        Return : the cloud object
        """
        for cloud in self.__clouds_list:
            if cloud.get_cloud_info()["id"] == id:
                return cloud
        print(f"No cloud with id {id} found")
        return None

    def printMultiCloud(self):
        for cloud in self.__clouds_list:
            cloud.printCloud()

    def get_service_by_id(self, Cloud_id: int, service_id: int) -> Service:
        """
        This function returns a service giving both cloud and service id
        Args :
            Cloud_id : id of the cloud where the service is
            service_id : id of the service we are looking for
        return: the Service object that is in cloud with Cloud_id and has id of service_id
        """
        return self.get_CloudById(Cloud_id).get_serviceById(service_id)

    def init_clouds_for_services(self) -> list[Cloud]:
        """
        This function initialise for each service its initial cloud, each service in the composition is associated
        with first cloud where it is availible
        Returns :
            List of clouds,
            exemple, [cloud0 , cloud1 , cloud0]:
                cloud of first service in composition has id = 0
                cloud of second service in composition has id = 1
                cloud of third service in composition has id = 0
        """
        clouds_init = []
        for i in range(len(self.serviceIds)):
            for j in range(self.getNumberClouds()):
                cloud = self.get_CloudById(j)
                Service_is_availible = cloud.service_is_dispo(self.serviceIds[i])
                if Service_is_availible:
                    break
            clouds_init.append(cloud)
        return clouds_init
