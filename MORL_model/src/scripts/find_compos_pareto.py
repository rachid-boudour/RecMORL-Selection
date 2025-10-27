from src.modules.MultiCloud import MultiCloud
from src.modules.AllCompositions import AllCompositions
from src.modules.Pareto import ParetoFront


"""
This script shoud be executed from PFE_Project directory
Used to get allCompositions_4obj.csv and ParetoPoints_4obj.csv
Used for evaluation + plotting 
"""


multiCloud = MultiCloud(directory="src/data/preprocessedData/filtred_normalized")
allComposition = AllCompositions(multiCloud)
allComposition.all_compositions_to_csv(
    "src/data/compositionPoints/allCompositions_4obj.csv"
)
listPareto = ParetoFront.calculate_Pareto_front(
    path_csv="src/data/compositionPoints/ParetoPoints_4obj.csv",
    multiCloud=multiCloud,
)
for row in listPareto:
    print(row)

print(len(listPareto))
