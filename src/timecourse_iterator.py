"""Iterator over serialized Timecourses stored in a zip archive."""

import src.constants as cn  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.model import Model

import numpy as np  # type: ignore
import pickle
import zipfile
from typing import Iterator


class TimecourseIteratorItem:
    """A model name paired with its deserialized Timecourse."""

    def __init__(self, model_name: str, timecourse: Timecourse) -> None:
        self.model_name = model_name
        self.timecourse = timecourse


class TimecourseIterator:
    """Iterates over serialized Timecourses in the timecourse zip archive."""

    def __init__(self, zip_path: str = cn.TIMECOURSE_ZIP_PATH,
            num_model:int = -1, first_model_num:int = 0, last_model_num:int = -1,
            ) -> None:
        """
        Args:
            zip_path (str, optional): _description_. Defaults to cn.TIMECOURSE_ZIP_PATH.
            num_model (int, optional): number of models to process. Defaults to -1 (all)
            first_model_num (int, optional): number of the first model to process. Defaults to 0.
            last_model_num (int, optional): number of the last model to process. Defaults to -1 (all).
        """
        self.zip_path = zip_path
        self.num_model = num_model
        self.first_model_num = first_model_num
        self.last_model_num = last_model_num

    @staticmethod
    def getTimecourse(model_name: str | int, zip_path: str = cn.TIMECOURSE_ZIP_PATH,
            ) -> Timecourse:
        """Return the deserialized Timecourse for *model_name* from the zip.

        Args:
            model_name (str | int): BioModels identifier (e.g. 'BIOMDD0000000001') or model number (e.g. 1).
            zip_path (str, optional): Path to the zip file containing serialized Timecourses.
                Defaults to ``cn.TIMECOURSE_ZIP_PATH``.

        Raises
        ------
        KeyError
            If no entry named ``{model_name}_timecourse.pkl`` exists in the zip.
        """
        iterator = TimecourseIterator()
        if isinstance(model_name, int):
            model_name = Model.getBiomodelName(model_name)
        entry_name = f"{model_name}_timecourse.pkl"
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(entry_name) as entry_f:
                dct = pickle.load(entry_f)
        timecourse = iterator._timecourseFromDict(dct)
        return timecourse

    def __iter__(self) -> Iterator[TimecourseIteratorItem]:
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for idx, name in enumerate(sorted(zf.namelist())):
                if self.num_model > 0 and idx >= self.num_model:
                    break
                pos = len(name)-len("_timecourse.pkl")
                model_name = name[:pos]
                if model_name.startswith("BIOMD"):
                    model_num = Model.getBiomodelNumberFromName(model_name)
                    if model_num < self.first_model_num:
                        continue
                    if self.last_model_num >= 0 and model_num > self.last_model_num:
                        break
                if not name.endswith('_timecourse.pkl'):
                    continue
                model_name = name[: -len('_timecourse.pkl')]
                with zf.open(name) as entry_f:
                    dct = pickle.load(entry_f)
                yield TimecourseIteratorItem(
                    model_name=model_name,
                    timecourse=self._timecourseFromDict(dct)
                )

    @staticmethod
    def _timecourseFromDict(dct: dict) -> Timecourse:
        return Timecourse(
            model=dct['model'],
            start_time=dct['start_time'],
            end_time=dct['end_time'],
            num_point=dct['num_point'],
            timecourse_df=dct['timecourse_df'],
        )
