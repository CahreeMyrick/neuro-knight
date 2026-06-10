from experiment import Experiment
from linear_experiment import LinearExperiment
from neural_experiment import NeuralExperiment

def ExperimentFactory(cfg: dict) -> Experiment:
    if (cfg["experiment"]["type"] == "linear"):
        return LinearExperiment(cfg)
    elif (cfg["experiment"]["type"] == "neural"):
        return NeuralExperiment(cfg)


