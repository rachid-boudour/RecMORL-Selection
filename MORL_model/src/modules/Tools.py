from itertools import combinations
import csv
from src.modules.commun import Constant
import os
from collections import defaultdict
import xml.etree.ElementTree as ET


class Tools:

    def path_join_folder_and_int_list(folder_path, int_list) -> str:
        """
        Joins a folder path (with or without trailing slash) and a list of integers,
        separated by 's' characters, to create a new file path.

        Args:
            folder_path: The path to the folder (may or may not have trailing slash).
            int_list: A list of integers.

        Returns:
            A string representing the new file path.
        """

        # Remove trailing slash from folder path if it exists
        folder_path = folder_path.rstrip("/")

        joined_path = folder_path + "/" + "s" + "s".join(str(x) for x in int_list)
        return joined_path

    def path_join_folder_and_int_list_csv(folder_path, int_list) -> str:
        """
        Joins a folder path (with or without trailing slash) and a list of integers,
        separated by 's' characters, to create a new file path.

        Args:
            folder_path: The path to the folder (may or may not have trailing slash).
            int_list: A list of integers.

        Returns:
            A string representing the new file path.
        """

        # Remove trailing slash from folder path if it exists
        folder_path = folder_path.rstrip("/")

        joined_path = (
            folder_path + "/" + "s" + "s".join(str(x) for x in int_list) + ".csv"
        )
        return joined_path

    def all_combinations(start, end, length):
        """
         This function generates all possible combinations of 'length' numbers
        from the range 'start' to 'end' (inclusive).
        """

        return [list(combo) for combo in combinations(range(start, end + 1), length)]

    def list_to_string(num_list) -> str:
        """Converts a list of numbers to a string with underscores.

        Args:
        num_list: A list of integers.

        Returns:
            A string with the numbers joined by underscores.
        """

        return "_" + "_".join(map(str, num_list)) + "_"

    def string_to_list(num_string) -> list[int]:
        """Converts a string with underscores to a list of numbers.

        Args:
        num_string: A string with numbers separated by underscores.

        Returns:
         A list of integers.
        """

        # Remove the leading and trailing underscores
        num_string = num_string[1:-1]
        return list(map(int, num_string.split("_")))

    def map_strings_to_integers(string_list: list[str]) -> list[int]:
        """
        Maps a list of string identifiers (e.g., "s0", "s1") to their corresponding integers
        based on a predefined mapping, and returns the sorted list of integers.

        Parameters:
        string_list (list[str]): A list of strings to be mapped to integers.

        Returns:
        list[int]: A sorted list of integers corresponding to the input strings.
        """
        # Define the mapping using a dictionary comprehension
        mapping = {f"s{i}": i for i in range(Constant.number_services)}

        # Map and sort the integers, only including valid keys
        result = sorted(mapping[s] for s in string_list if s in mapping)

        return result

    def revert_value(normalized_value, service_id, attr_name):
        def compute_max_min_data():
            # Data structure to hold service data
            services_data = defaultdict(lambda: defaultdict(list))

            # Function to parse each XML file
            def parse_xml(file_path):
                tree = ET.parse(file_path)
                root = tree.getroot()
                for service in root.find("services").findall("service"):
                    service_id = service.get("id")
                    for attribute in service:
                        services_data[service_id][attribute.tag].append(
                            float(attribute.text)
                        )

            # Loop through all XML files in the folder
            folder_data_original = Constant.originalDataFolder
            for file_name in os.listdir(folder_data_original):
                if file_name.endswith(".xml"):
                    parse_xml(os.path.join(folder_data_original, file_name))

            # Calculate max and min values for each attribute of each service id
            result = {}
            for service_id, attributes in services_data.items():
                result[service_id] = {}
                for attr, values in attributes.items():
                    result[service_id][attr] = {"max": max(values), "min": min(values)}

            return result

        max_val = compute_max_min_data()[service_id][attr_name]["max"]
        min_val = compute_max_min_data()[service_id][attr_name]["min"]
        original_value = max_val * (1 - normalized_value) + min_val
        return original_value

    def check_Pareto_calculated(service_querry: list, number_clouds: int):
        """
        Checks if a given CSV file is in the pareto folder folder. So we dont do useless calclules to find pareto

        Parameters:
        file_name (str): The name of the CSV file to check.

        Returns:
        bool: True if the file is found, False otherwise.
        """
        pareto_folder = Tools.path_join_folder_and_int_list_csv(
            Constant.paretosFolder + f"{number_clouds}clouds", service_querry
        )

        # Check if the file exists in the folder
        print(
            "Pareto already exist for",
            service_querry,
            "in multi cloud of",
            number_clouds,
            "cloud :",
            os.path.isfile(pareto_folder),
        )
        return os.path.isfile(pareto_folder)
