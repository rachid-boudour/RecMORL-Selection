from src.modules.AllCompositions import AllCompositions
from src.modules.MultiCloud import MultiCloud
from src.modules.commun import Constant
import csv
import numpy as np
from src.modules.Tools import Tools
from src.scripts.nsga3 import calculate_pareto_nsga3


class ParetoFront:
    """
    This class helps to find the true pareto front points
    """

    def __init__() -> None:
        pass

    def calculate_Pareto_front_dominance_rule(multiCloud: MultiCloud) -> None:
        """
        This function takes a list of points (represented as tuples of (x, y, z ..) coordinates)
        and stores in a csv file the dominant points.

        A point p(x,y) dominates another point q(x,y) if p.x >= q.x and p.y >= q.y. and (p.x > q.x or p.y > q.y)

        Args:
            multiCloud
        """
        dominated = []
        dominant = []
        allCompositions = AllCompositions(multiCloud)
        all_service_combinations = allCompositions.All_Possible_compositions_Scores()

        for combinision_curnt in all_service_combinations:
            # Verifier si régle de domminance non verifier donc ajouter le point a la liste dominated
            point_curent = combinision_curnt["composition_score"]
            check_dominant = True
            for combinision in all_service_combinations:
                if check_dominant == True:
                    point = combinision["composition_score"]
                    if all(a >= b for a, b in zip(point, point_curent)) and any(
                        a > b for a, b in zip(point, point_curent)
                    ):
                        # point_curent is dominated
                        check_dominant = False
                        dominated.append(combinision_curnt)

        for combinision in all_service_combinations:
            if combinision not in dominated:
                dominant.append(combinision)
        print("number dominated:", len(dominated), "number dominant:", len(dominant))
        # Put the dominant points found in a CSV file :
        return dominant

    def calculate_Pareto_front(multiCloud: MultiCloud) -> list[np.ndarray]:
        pf = []
        if Tools.check_Pareto_calculated(
            multiCloud.serviceIds, multiCloud.getNumberClouds()
        ):  # We already have pareto
            return ParetoFront.get_Pareto_from_csv(
                service_query=multiCloud.serviceIds,
                number_clouds=multiCloud.getNumberClouds(),
            )
        else:
            return calculate_pareto_nsga3(
                services_querry=multiCloud.serviceIds,
                MultiCloud_data_dir=multiCloud.multiCloud_dir,
            )

    def pareto_to_csv(dominant: np.ndarray, multicloud: MultiCloud):
        path_csv = Tools.path_join_folder_and_int_list_csv(
            Constant.paretosFolder + f"{multicloud.getNumberClouds()}clouds",
            multicloud.serviceIds,
        )
        print("Saving calculated pâreto in the file :", path_csv)
        with open(path_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row_dict in dominant:
                writer.writerow(row_dict["cloud_ids"] + row_dict["composition_score"])

    def get_Pareto_from_csv(
        service_query: list[int], number_clouds: int
    ) -> list[np.ndarray]:
        """
        This function reads a CSV file that contains Pareto points and returns
        a list of np.arrays that contains the QoS scores.

        Args:
           service_query (list): A list of integers used to construct the CSV file path.

        Returns:
           list: A list of np.arrays containing the last n elements from each row in the CSV file.
        """
        paretoPoints = []
        csv_file_path: str = Tools.path_join_folder_and_int_list_csv(
            Constant.paretosFolder + f"{number_clouds}clouds", service_query
        )
        print("get pareto from the file :", csv_file_path)

        with open(csv_file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the last num_objectives elements and convert to float
                last_objectives = [
                    float(score) for score in row[-Constant.number_objectives :]
                ]
                paretoPoints.append(np.array(last_objectives))

        return paretoPoints
