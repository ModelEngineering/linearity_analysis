"""Iterator over serialized Timecourses stored in a zip archive."""

import os

import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore

import numpy as np  # type: ignore
import pickle
import zipfile
from typing import Iterator, Optional, Union


class TimecourseIteratorItem:
    """A model name paired with its deserialized Timecourse."""

    def __init__(self, model_name: str, timecourse: Timecourse) -> None:
        self.model_name = model_name
        self.timecourse = timecourse


class TimecourseIterator:
    """Iterates over serialized Timecourses in the timecourse zip archive."""

    def __init__(self, zip_path: str = cn.TIMECOURSE_ZIP_PATH,
            num_model:int = -1, first_model_num:int = 0, last_model_num:int = -1,
            num_point:int = 1000) -> None:
        """
        Args:
            zip_path (str, optional): _description_. Defaults to cn.TIMECOURSE_ZIP_PATH.
            num_model (int, optional): number of models to process. Defaults to -1 (all)
            first_model_num (int, optional): number of the first model to process. Defaults to 0.
            last_model_num (int, optional): number of the last model to process. Defaults to -1 (all).
            num_point (int, optional): number of points in each timecourse. Defaults to 1000.
        """
        self.zip_path = zip_path
        self.num_model = num_model
        self.first_model_num = first_model_num
        self.last_model_num = last_model_num
        self.num_point = num_point

    @staticmethod
    def getTimecourse(model_name: Union[str, int], zip_path: str = cn.TIMECOURSE_ZIP_PATH,
            ) -> Timecourse:
        """Return the deserialized Timecourse for *model_name* from the zip.

        Args:
            model_name (str | int): BioModels identifier (e.g. 'BIOMD0000000001') or model number (e.g. 1).
            zip_path (str, optional): Path to the zip file containing serialized Timecourses.
                Defaults to ``cn.TIMECOURSE_ZIP_PATH``.

        Returns
        -------
        Timecourse

        Raises
        ------
        FileNotFoundError
            If *zip_path* does not exist.
        KeyError
            If no entry named ``{model_name}_timecourse.pkl`` exists in the zip.
        """
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(
                f"Zip file not found: {zip_path}. "
                f"Use Timecourse.makeBiomodelDF({model_name!r}) to generate on demand.")

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
        if not os.path.isfile(self.zip_path):
            print(f"Zip file not found: {self.zip_path}. "
                    f"Generating timecourses from SBML on the fly.")
            yield from self._generate_from_sbml()
            return

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            names = sorted(zf.namelist())
            for name in names:
                model_num = int(name[: -len('_timecourse.pkl')].replace('BIOMD', ''))
                if model_num < self.first_model_num:
                    continue
                if self.last_model_num >= 0 and model_num > self.last_model_num:
                    break
                model_name = name[: -len('_timecourse.pkl')]
                try:
                    with zf.open(name) as entry_f:
                        dct = pickle.load(entry_f)
                        timecourse = self._timecourseFromDict(dct)
                except Exception as e:
                    print(f"Could not load {name} from zip: {e}. "
                            f"Generating timecourse from SBML instead.")
                    try:
                        timecourse = Timecourse.makeBiomodelDF(
                            model_name, num_point=self.num_point)
                    except Exception as gen_e:
                        print(f"  Failed to generate timecourse for {model_name}: "
                                f"{gen_e}. Skipping.")
                        continue
                yield TimecourseIteratorItem(
                    model_name=model_name, timecourse=timecourse)

    def _generate_from_sbml(self) -> Iterator[TimecourseIteratorItem]:
        """Generate a Timecourse from SBML for every BioModel in ``cn.BIOMODELS_DIR``,
        respecting the ``first_model_num`` / ``last_model_num`` filters.
        """
        model_dir = cn.BIOMODELS_DIR
        if not os.path.isdir(model_dir):
            print(f"BioModels directory not found: {model_dir}. Nothing to generate.")
            return

        for entry_name in sorted(os.listdir(model_dir)):
            if not entry_name.startswith("BIOMD"):
                continue
            try:
                model_num = Model.getBiomodelNumberFromName(entry_name)
            except ValueError:
                continue
            if model_num < self.first_model_num:
                continue
            if self.last_model_num >= 0 and model_num > self.last_model_num:
                break
            yield TimecourseIteratorItem(
                model_name=entry_name,
                timecourse=Timecourse.makeBiomodelDF(entry_name, num_point=self.num_point),
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
