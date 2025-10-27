class Constant:
    """
    This class containes the commun constants that the diffrent modules need
    """

    number_objectives: int = 6

    # Folder when pareto frontier is saved , we use it the in the algo to calculate IGD
    paretosFolder = "./src/data/paretos/"
    trainedFolder = "./trained/"
    allCompositionFolder = "./src/data/AllCompositions/"
    originalDataFolder = "./src/data/data_set/"
