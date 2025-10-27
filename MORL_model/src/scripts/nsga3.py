import numpy as np
from pymoo.core.problem import ElementwiseProblem
from src.modules.MultiCloud import MultiCloud

from src.modules.commun import Constant
from src.modules.Tools import Tools
from src.modules.Compositon import Composition
import pandas as pd
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter


def calculate_pareto_nsga3(services_querry: list, MultiCloud_data_dir: str):
    number_Services_Compos = len(services_querry)

    multicloud = MultiCloud(
        serviceIds=services_querry, multiCloud_dir=MultiCloud_data_dir
    )
    numberClouds = multicloud.getNumberClouds()

    number_clouds = multicloud.getNumberClouds()
    numberVar = number_clouds**number_Services_Compos
    pareto_dir = Tools.path_join_folder_and_int_list_csv(
        Constant.paretosFolder, services_querry
    )

    class SelectService(ElementwiseProblem):

        def __init__(self):
            super().__init__(
                n_var=number_Services_Compos,
                n_obj=Constant.number_objectives,
                n_ieq_constr=0,
                xl=np.full(number_Services_Compos, 0),
                xu=np.full(number_Services_Compos, number_clouds - 1),
                vtype=int,
            )
            self.multicloud = multicloud

        def _evaluate(self, x, out, *args, **kwargs):
            services = []
            valid = True
            for i in range(number_Services_Compos):
                cloud = self.multicloud.get_CloudById(int(x[i]))
                service = cloud.get_serviceById(services_querry[i])
                if service is None:  # If the service is not available
                    valid = False
                    print("here nsga3 service", service, "dosnt exist in cloud", x[i])
                    break
                print("append service", service)
                services.append(service)

            if valid:
                # Perform your evaluation and set the objective values
                composition = Composition(services)
                out["F"] = [
                    -service_score
                    for service_score in composition.calculate_score_script()
                ]
            else:
                # Assign a high penalty for invalid solutions
                out["F"] = [float("1")] * Constant.number_objectives

        def _evaluate(self, x, out, *args, **kwargs):
            services = []
            for i in range(number_Services_Compos):
                cloud = self.multicloud.get_CloudById(int(x[i]))
                service = cloud.get_serviceById(services_querry[i])
                services.append(service)
            composition = Composition(services)
            if None in services:  # one of the services dosnt exist in the chosen cloud
                out["F"] = [1] * Constant.number_objectives
            else:
                out["F"] = [
                    -service_score
                    for service_score in composition.calculate_score_script()
                ]

    problem = SelectService()

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions(
        "das-dennis", Constant.number_objectives, n_partitions=8
    )

    # create the algorithm object
    algorithm = NSGA3(
        pop_size=2000,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=30.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.7, eta=20.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
        ref_dirs=ref_dirs,
    )

    # execute the optimization
    res = minimize(
        problem,
        algorithm,
        seed=1,
        termination=("n_gen", 15),
        save_history=True,
        verbose=True,
    )

    print("The solutions are : ")
    # Access the Pareto front solutions
    pareto_front = res.X
    pareto_objectives = -res.F

    print("Pareto front solutions:")
    print(pareto_front)

    print("Objective values of the Pareto front solutions:")
    print(pareto_objectives)

    # Create DataFrames for pareto front and objectives
    pareto_front_df = pd.DataFrame(
        pareto_front, columns=[f"var_{i}" for i in range(pareto_front.shape[1])]
    )
    pareto_objectives_df = pd.DataFrame(
        pareto_objectives,
        columns=[f"obj_{i}" for i in range(pareto_objectives.shape[1])],
    )

    # Concatenate decision variables and objectives
    solutions_df = pd.concat([pareto_front_df, pareto_objectives_df], axis=1)

    # Save to CSV
    path_pareto = Tools.path_join_folder_and_int_list_csv(
        Constant.paretosFolder + f"{multicloud.getNumberClouds()}clouds",
        int_list=services_querry,
    )
    solutions_df.to_csv(path_pareto, index=False)

    print("Pareto front solutions saved to", path_pareto)

    # Convert pareto_objectives_df to a list of numpy arrays
    pareto_objectives_list = pareto_objectives_df.to_numpy().tolist()

    return pareto_objectives_list
