import numpy as np

from pymoo.optimize import minimize


from pymoo.optimize import minimize


import sys

from pymoo.factory import get_problem
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import autograd.numpy as anp
from pymoo.core.problem import Problem
import time
from glob import glob
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
import os
import random

from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_problem, get_reference_directions


import numpy as np
from pymoo.core.problem import ElementwiseProblem
from src.modules.MultiCloud import MultiCloud
from src.modules.commun import Constant
from src.modules.Tools import Tools
from src.modules.Compositon import Composition
import numpy as np
import pandas as pd


def calculate_pareto_nsga2(services_querry: list, MultiCloud_data_dir: str):
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
            for i in range(number_Services_Compos):
                cloud = self.multicloud.get_CloudById(int(x[i]))
                service = cloud.get_serviceById(services_querry[i])
                services.append(service)

            composition = Composition(services)
            out["F"] = [
                -service_score for service_score in composition.calculate_score_script()
            ]
            # out["G"] = 0.1 - out["F"]

        """ THIS IS TO USE PARETO FOR IGD? DOESNT WORK FOR NOW, HAVE TO FIND A WAY TO GET TRUE PARETO
            def _calc_pareto_front(self, n_pareto_points=130):
            data = np.loadtxt(pareto_dir, delimiter=",") * -1
            # Select the last 4 elements from each row
            last_four = data[:, -Constant.number_objectives:]
            return (last_four)"""

    problem = SelectService()

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions(
        "das-dennis", Constant.number_objectives, n_partitions=14
    )

    # create the algorithm object
    algorithm = get_algorithm(
        "nsga2",
        pop_size=2000,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx"),
        mutation=get_mutation("int_pm", eta=3.0),
        eliminate_duplicates=False,
        pf=None,
        save_history=False,
        verbose=True,
    )

    # execute the optimization
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 15),
        seed=1,
        save_history=True,
        disp=False,
        verbose=True,
    )

    from pymoo.visualization.scatter import Scatter

    labels = ["Enrg", "Cost", "reliability", "availability", "RT", "nb_cld"]
    plot = Scatter(labels=labels)  # Pass labels as an argument
    plot.add(res.F)
    plot.show()

    print("The solutions are : ")
    # Access the Pareto front solutions
    pareto_front = res.X
    pareto_objectives = -res.F

    print("Pareto front solutions:")
    print(pareto_front)

    print("Objective values of the Pareto front solutions:")
    print(pareto_objectives)

    # Save the solutions to a CSV file
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
        folder_path="./src/data/paretos", int_list=services_querry
    )
    solutions_df.to_csv(path_pareto, index=False)

    print("Pareto front solutions saved to nsga2_solutions.csv")
    return res.F
