from src.modules.MultiCloud import MultiCloud
from src.modules.commun import Constant
from src.modules.Compositon import Composition
import csv
from itertools import product
from src.modules.Tools import Tools


class AllCompositions:
    """
    This class gives all the compositions we can get from the multiCloud
    """

    def __init__(
        self,
        multicloud: MultiCloud,
    ):
        self.__multicloud = multicloud

    def All_Possible_compositions_Scores(self) -> list:
        """
        This function generate all possible service compositions with their QoS score
        Returns:
            List:a list that containes dictionaries with keys 'cloud_ids' and 'composition_score' gives for each possible service composition the corresponding QoS score.
        """
        clouds_ids = list(range(0, self.__multicloud.getNumberClouds()))
        # We generate all possible compositions first :
        all_compositions = product(clouds_ids, repeat=len(self.__multicloud.serviceIds))
        # print(list(all_compositions))
        # print(len(list(all_compositions)))
        # Now we get the score of these compositions :
        all_points = []
        for composition_ids in all_compositions:
            # print(composition_ids)
            services_comp = []
            for i in range(len(composition_ids)):
                cloud_id = composition_ids[i]
                service_id = self.__multicloud.serviceIds[i]
                cloud = self.__multicloud.get_CloudById(cloud_id)
                service = cloud.get_serviceById(service_id)
                services_comp.append(service)
            composition = Composition(services_comp)

            row = list(composition_ids) + list(composition.calculate_score_script())
            # Save the composition and it score in a csv file
            all_points.append(
                {
                    "cloud_ids": list(composition_ids),
                    "composition_score": list(composition.calculate_score_script()),
                }
            )
        return all_points

    def all_compositions_to_csv(self):
        """
        This function output a csv file with generated possible composition of services compositions and their QoS score
        Args:
            path: The path to the csv file
        """
        path_file = Tools.path_join_folder_and_int_list(
            Constant.allCompositionFolder, self.__multicloud.serviceIds
        )
        with open(path_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row_dict in self.All_Possible_compositions_Scores():
                writer.writerow(row_dict["cloud_ids"] + row_dict["composition_score"])
