"""This file contains the utils for handling the experiments data and models""" 


from typing import Dict, Literal, List, Union, Any
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Model, load_model
import pickle
from pydantic import BaseModel
import os


class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class ExperimentSettings(CustomBaseModel):
    """Experiment settings used within the Experiment Class."""

    look_back_window: int = None
    input_features: List = None
    target_features: List = ["lobith_q"]
    model_time_step: str = "hourly"
    pre_processing_info: Dict = {}
    data_source: Literal["bfg", "matroos"] = None


class ExperimentDataFrames(CustomBaseModel):
    """Experiment dataframes used within the Experiment Class."""

    train: pd.DataFrame = None
    val: pd.DataFrame = None
    test: pd.DataFrame = None
    pred: pd.DataFrame = None

    def filter_pred(self, start: pd.Timestamp, end: pd.Timestamp):
        return self.pred.loc[(slice(start, end), slice(None))]

    def get_pred_time_range(self):
        return (self.pred.index[0][0], self.pred.index[-1][0])


class Experiment(CustomBaseModel):
    """Experiment class to store experiments."""

    name: str
    settings: ExperimentSettings
    scaler: Any = None
    dfs: ExperimentDataFrames = None
    _model: Model = None
    additional_data: Dict = {}

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Model):
        self._model = model

    def save(self, experiments_folder: Path, model: Union[Model, None] = None):
        file_path = experiments_folder / f"{self.name}.pkl"
        if model != None:
            model_path = experiments_folder / f"{self.name}.keras"
            if model_path.exists():
                raise ValueError(
                    f"Model {self.name} already exists, please delete the file or rename your experiment."
                )
            else:
                model.save(model_path)

        if file_path.exists():
            raise ValueError(
                f"Experiment {self.name} already exists, please delete the file or rename your experiment."
            )
        else:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, experiments_folder, name):
        experiments_folder = Path(str(experiments_folder))
        pkl_file = experiments_folder / f"{name}.pkl"
        
        # Use os.path.normpath to normalize path separators
        pkl_file = os.path.normpath(str(pkl_file))
        
        with open(pkl_file, "rb") as f:
            loaded_instance = pickle.load(f)
        
        model_path = experiments_folder / f"{name}.keras"
        if model_path.exists():
            model = load_model(model_path)
            loaded_instance.model = model
        return loaded_instance
