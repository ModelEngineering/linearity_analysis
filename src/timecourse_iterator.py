"""Iterator over serialized Timecourses stored in a zip archive."""

import os

import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.biomodels_iterator import getBiomodelsEndtimes  # type: ignore

import numpy as np  # type: ignore
import pandas as pd # type: ignore
import pickle
import zipfile
from typing import Iterator, Optional, Union, List


class TimecourseIteratorItem:
    """A model name paired with its deserialized Timecourse."""

    def __init__(self, model_name: str, timecourse: Timecourse) -> None:
        self.model_name = model_name
        self.timecourse = timecourse


class TimecourseIterator:
    """Iterates over serialized Timecourses in the timecourse zip archive."""

    def __init__(self,
            zip_path: Optional[str] = None,
            first_model_num:int = 0,
            num_model_collection: Optional[List[int]] = None,
            last_model_num:int = -1,
            num_point:int = cn.NUM_POINT, is_sedml_endtime: bool = True,
            is_report:bool = False) -> None:
        """
        Args:
            zip_path (str, optional): _description_. Defaults to cn.TIMECOURSE_ZIP_PATH.
            first_model_num (int, optional): number of the first model to process. Defaults to 0.
            num_model_collection (List[int], optional): list of model numbers to process. Defaults to None.
            last_model_num (int, optional): number of the last model to process. Defaults to -1 (all).
            num_point (int, optional): number of points in each timecourse. Defaults to cn.NUM_POINT.
            is_sedml_endtime (bool, optional): only includes models with endtime from SED-ML. Defaults to True.
        """
        if zip_path is None:
            zip_path = self.getZipPath(num_point=num_point)
        self.zip_path = zip_path
        self.first_model_num = first_model_num
        self.last_model_num = last_model_num
        self.num_point = num_point
        self.is_report = is_report
        self.num_model_collection = num_model_collection
        endtime_dct = getBiomodelsEndtimes(is_include_endtime_source=True)
        if is_sedml_endtime:
            self._valid_model_nums = [Model.getBiomodelNum(m)
                    for m, s in endtime_dct.items() if s[1] == "sedml"]
        else:
            self._valid_model_nums = [Model.getBiomodelNum(m) for m in endtime_dct.keys()]

    @staticmethod
    def getZipPath(num_point: int = cn.NUM_POINT) -> str:
        """Return the path to the zip file containing serialized Timecourses.

        Args:
            num_point (int, optional): Number of points in each timecourse. Defaults to cn.NUM_POINT.

        Returns:
            str: Path to the zip file.
        """
        if num_point == 1000:
            return cn.TIMECOURSE_ZIP_1000_PATH
        elif num_point == 10000:
            return cn.TIMECOURSE_ZIP_10000_PATH
        elif num_point == 100000:
            return cn.TIMECOURSE_ZIP_100000_PATH
        else:
            raise ValueError(f"Unsupported num_point {num_point}. Use 1000, 10000, or 100000.")

    @classmethod
    def getTimecourse(cls, model_name: Union[str, int],
            zip_path: str = cn.TIMECOURSE_ZIP_PATH, num_point: Optional[int]=None) -> Timecourse:
        """Return the deserialized Timecourse for *model_name* from the zip.

        Args:
            model_name (str | int): BioModels identifier (e.g. 'BIOMD0000000001') or model number (e.g. 1).
            zip_path (str, optional): Path to the zip file containing serialized Timecourses.
                Defaults to ``cn.TIMECOURSE_ZIP_PATH``.
            num_point: int: Number of points used to select the zippath

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
        if num_point is not None:
            zip_path = cls.getZipPath(num_point=num_point)
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
            raise FileNotFoundError(
                f"Zip file not found: {self.zip_path}. ")
        checker = CurationChecker()
        # Process the files in the zip path
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            names = sorted(zf.namelist())
            for name in names:
                if name.startswith("BIOMD"):
                    model_num = int(name[: -len('_timecourse.pkl')].replace('BIOMD', ''))
                    if model_num not in self._valid_model_nums:
                        continue
                    if self.is_report:
                        print(f"Processing {model_num}")
                    if model_num < self.first_model_num:
                        continue
                    if self.last_model_num >= 0 and model_num > self.last_model_num:
                        break
                    if self.num_model_collection is not None and model_num not in self.num_model_collection:
                        continue
                model_name = name[: -len('_timecourse.pkl')]
                # Process the model
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

    @staticmethod
    def _timecourseFromDict(dct: dict) -> Timecourse:
        return Timecourse(
            model=dct['model'],
            start_time=dct['start_time'],
            end_time=dct['end_time'],
            num_point=dct['num_point'],
            timecourse_df=dct['timecourse_df'],
        )

class CurationChecker(object):

    def __init__(self):
        self.curation_eliminated_models = pd.read_csv(cn.CURATION_PATH)[cn.COL_SYSTEM_ID].to_list() 

    def isCurated(self, model_num: Optional[int] = None, model_name: Optional[str] = None,
            ) -> bool:
        if (model_num is None) and (model_name is None):
            raise ValueError("Must specify one of model_num and model_name.")
        if model_num is not None:
            model_name = Model.getBiomodelName(model_num)
        if not model_name in self.curation_eliminated_models:
            return True
        else:
            return False