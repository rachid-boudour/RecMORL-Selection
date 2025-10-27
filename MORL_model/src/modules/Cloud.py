import xml.etree.ElementTree as ET
from src.modules.commun import Constant

# from src.modules.Compositon import Composition
from src.modules.Service import Service


class Cloud:
    """
    This class represents a Cloud
    """

    def __init__(self, xml_file: str, serviceIds: list[int]):
        self.xml_file = xml_file
        self.serviceIds = serviceIds
        self.__id = self.get_cloud_info()["id"]
        self.__name = self.get_cloud_info()["name"]
        self.__Services_list = self.get_cloud_Services_list()

    def get_cloud_info(self) -> dict:
        """
        This function parses the provided XML data and returns a dictionary containing the cloud ID and name.
        Args:
            xml_path (str): The XML path as a string.
        Returns:
            dict: A dictionary with keys 'id' and 'name' containing the extracted information.
        """
        try:
            # Parse the XML file
            tree = ET.parse(self.xml_file)
            root = tree.getroot()

            # Extract cloud ID and name attributes
            cloud_id = root.attrib["id"]
            cloud_name = root.attrib["name"]

            return {"id": int(cloud_id), "name": cloud_name}
        except FileNotFoundError:
            print(f"Error: File not found at {self.xml_file}")
            return {}

    def service_is_dispo(self, service_id: int) -> bool:
        """
        Check if the service passed in argument by it id is in this cloud or not
        Args:
            service_id (int): the id of the service we are looking for
        Returns:
            True if service exist in the cloud, else returns False
        """
        for service in self.__Services_list:
            if service.get_serviceId() == service_id:
                return True
        return False

    def get_serviceById(self, id) -> Service | None:
        """
        function that returns a service object that correspond to the giving id
        if the service doesnt exist it returns None
        """
        for service in self.__Services_list:
            if service.get_serviceId() == id:
                return service
        print("service", id, "doesnt exist in the cloud", self.__id)
        return None

    def get_cloud_Services_list(self) -> list[Service]:
        """
        This function parses the XML data from a file and returns a list containing all services of this cloud
        Args:
            xml_file_path: The path to the XML file.
        Returns:
            A list of service object containing service attributes
        """
        try:
            with open(self.xml_file, "r") as f:
                xml_data = f.read()
        except FileNotFoundError:
            print(f"Error: XML file not found at {self.xml_file}")
            return []

        root = ET.fromstring(xml_data)
        services = root.find("services")
        Cloud_services = []
        for service in services.findall("service"):
            if int(service.attrib["id"]) in self.serviceIds:
                service_obj = Service(
                    int(service.attrib["id"]),
                    self.__id,  # id of cloud
                    float(service.find("energy").text),
                    float(service.find("cost").text),
                    float(service.find("reliability").text),
                    float(service.find("availability").text),
                    float(service.find("response_time").text),
                )
                Cloud_services.append(service_obj)
        return Cloud_services

    def printCloud(self) -> None:
        print("Cloud id = ", self.__id)
        print("Cloud name = ", self.__name)
        print(":::::::::::::::::::::::::::::::::::::::::::")
        print("Services : ")
        for service in self.__Services_list:
            print(service.display_service())
        print("\n////////////////////////////////////////")
