"""Iterator over piecewise predictions CSV and pickle files with coded metadata in name."""

import os
import pickle
import pandas as pd # type: ignore
from typing import Iterator, Optional, Callable, Union, cast, Dict, Any

from tomlkit import key

import src.constants as cn  # type: ignore
from src import util  # type: ignore
from src.model import Model
import src.util as util

import pandas as pd  # type: ignore
    
REQUIRED_META_DATA = ["manycp", "threshold", "removal", "numpoint"]


class PSDPredictionItem:
    """Represents a single piecewise predictions file with its metadata and data."""

    def __init__(
        self,
        filename: str,
        filepath: str,
        max_fractional_reduction: float,
        coefficient_threshold: float,
        is_changepoint_removal: bool,
        max_changepoint: int,
        num_point: int,
        repeat: Optional[int],
        manycp: bool,
        df: pd.DataFrame
    ) -> None:
        """
        Initialize a PSDPredictionsItem.

        Args:
            filename: Name of the prediction file.
            filepath: Path to the prediction file.
            max_fractional_reduction: Maximum fractional reduction parameter from filename.
            coefficient_threshold: Coefficient threshold parameter from filename.
            is_changepoint_removal: Whether changepoint removal was enabled.
            max_changepoint: Maximum number of change points.
            num_point: Number of time points.
            repeat: Repeat index (if present in filename).
            manycp: Whether many changepoints mode was used.
            df: DataFrame containing the prediction data.
        """
        self.max_fractional_reduction = max_fractional_reduction
        self.coefficient_threshold = coefficient_threshold
        self.is_changepoint_removal = is_changepoint_removal
        self.max_changepoint = max_changepoint
        self.num_point = num_point
        self.repeat = repeat
        self.manycp = manycp
        self.df = df
        self.filepath = filepath
        self.file_dir = os.path.dirname(filepath)
        self.filename = os.path.splitext(filename)[0]
        self.cvs_filepath = self.filename + ".csv"
        self.pkl_filepath = self.filename + ".pkl"

    @property
    def pkl_path(self) -> str:
        """Return the full path to the pickle file."""
        return os.path.join(self.file_dir, self.pkl_filepath)

    @property
    def csv_path(self) -> str:
        """Return the full path to the CSV file."""
        return os.path.join(self.file_dir, self.cvs_filepath)


class PSDPredictionFileIterator:
    """Iterates over piecewise predictions CSV and pickle files in cn.PSD_DATA_DIR."""

    def __init__(self, is_pkl: bool = False,
            data_dir: str = cn.PSD_DATA_DIR,
            is_guard_filename: bool = False) -> None:
        """
        Initialize the iterator.

        Args:
            is_pkl: If True, iterate over pickle files. If False, iterate over CSV files.
            data_dir: Directory to iterate over.:w
            is_guard_filename: If True, guard against filenames that don't match the expected
                pattern. If False, skip the guard and allow any filename to be processed.
        """
        self.is_pkl = is_pkl
        self.data_dir = data_dir
        self.is_guard_filenames = is_guard_filename
#
#    def _parse_filename(self, filename: str) -> Optional[dict]:
#        """
#        Parse the encoded metadata from a piecewise predictions filename.
#
#        Filenames follow patterns like:
#            piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__maxreduction_0.01.csv
#            piecewise_predictions__numpoint_100000__threshold_0.001__removal_1__maxreduction_1e-2.csv
#
#        Or for simple numbered files:
#            piecewise_predictions_0.csv
#            piecewise_predictions_1.pkl
#
#        Returns a dictionary with parsed metadata, or None if the filename doesn't match
#        the expected pattern.
#        """
#        # Process the filename to extract the base name without extension
#        name_without_ext = filename
#        if self.is_pkl:
#            ext = ".pkl"
#        else:
#            ext = ".csv"
#        if not filename.endswith(ext):
#            raise ValueError(f"Filename {filename} does not have expected extension {ext}.")
#        name_without_ext = filename[:-len(ext)]
#        # Check if it starts with "piecewise_predictions"
#        if not name_without_ext.startswith("piecewise_predictions"):
#            raise ValueError(f"Filename {filename} does not start with expected prefix 'piecewise_predictions'.")
#        # Initialize metadata with defaults
#        metadata: dict = {
#            "num_point": None,
#            "threshold": None,
#            "removal": None,
#            "maxreduction": None,
#            "manycp": None,
#            "repeat": None,
#        }
#        # Parse the filename for metadata
#        if not "__" in name_without_ext:
#            raise ValueError(f"Filename {filename} does not contain expected metadata separators '__'.")
#        # Extract parts after "piecewise_predictions"
#        rest = name_without_ext[len("piecewise_predictions"):]
#        # Remove leading __ if present
#        if rest.startswith("__"):
#            rest = rest[2:]
#        # Use codedstrToDict to parse the metadata
#        parsed_metadata = util.codedstrToDict(rest)
#        metadata.update(parsed_metadata)
#        #
#        return metadata

    def _load_dataframe(self, filepath: str) -> pd.DataFrame:
        """
        Load a DataFrame from either a CSV or pickle file.

        Args:
            filepath: Path to the file.

        Returns:
            Loaded DataFrame.
        """
        if self.is_pkl:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            return pd.read_csv(filepath)

    def __iter__(self) -> Iterator[PSDPredictionItem]:
        """
        Iterate over piecewise predictions files.

        Yields:
            PSDPredictionItem: Contains filename, metadata values, and DataFrame.
        """
        if not os.path.isdir(self.data_dir):
            return
        

        # Find all relevant files
        files = sorted([f for f in os.listdir(self.data_dir)])
        files = [f for f in files if "piecewise_predictions" in f]
        files = [f for f in files if "__numpoint_" in f]
        files = [f for f in files if not "dummy" in f]
        files = [f for f in files if (self.is_pkl and f.endswith(".pkl"))
                or ((not self.is_pkl) and f.endswith(".csv"))]
        
        if self.is_guard_filenames:
            files = [f for f in files if all([k in f for k in REQUIRED_META_DATA])]
        
        for filename in files:
            # Get the full path to the file
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.isfile(filepath):
                continue
            
            # Parse metadata from filename
            metadata = util.parseFilename(filename)
            if metadata is None:
                continue
            
            try:
                if self.is_pkl:
                    with open(filepath, "rb") as f:
                        df = pickle.load(f)
                else:
                    df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
            
            # Extract metadata values for the constructor
            yield PSDPredictionItem(
                filename=filename,
                filepath=filepath,
                max_fractional_reduction=metadata.get(cn.COL_MAX_FRACTIONAL_REDUCTION, None),  # type: ignore
                coefficient_threshold=metadata.get(cn.COL_COEFFICIENT_THRESHOLD, None),  # type: ignore
                is_changepoint_removal=bool(metadata.get(cn.COL_IS_CHANGEPOINT_REMOVAL, None)),  # type: ignore
                max_changepoint=metadata.get(cn.COL_MAX_CHANGEPOINT, None),  # type: ignore
                num_point=metadata.get(cn.COL_NUM_POINT, None),  # type: ignore
                repeat=metadata.get(cn.COL_REPEAT, None),  # type: ignore
                manycp=bool(metadata.get(cn.COL_MANYCP, None)),  # type: ignore
                df=df,
            )

    @classmethod
    def query(cls,
            file_filter: Optional[Callable[[dict], bool]] = None,
            data_frame_filter: Optional[Callable[[pd.DataFrame, dict], pd.DataFrame]] = None,
            is_pkl: bool = True,
            is_model: bool = True,
            data_dir: str = cn.PSD_DATA_DIR,
            is_guard_filename: bool = False,
            ) -> pd.DataFrame:
        """
        Query the piecewise predictions files based on filters for the files and their DataFrames.
        Adds a column to the result for the filename of the source file for each row.

        Args:
            file_filter (Callable[[dict], bool]): Filter files processed
                signature:  args: filename ; returns: True if it should be included in the query.
                The dictionary contains the parsed metadata from the filename.
            data_frame_filter (Callable[[pd.DataFrame, dict], pd.DataFrame]): filter of file
                signature: args: A DataFrame; returns: A filtered DataFrame.
            is_pkl (bool): If True, query pickle files; if False, query CSV files.
            is_model (bool): If True, only include files with model-level aggregation.
            data_dir (str): Directory to search for files.
            is_guard_filename (bool): If True, guard against filenames that don't match the expected
                pattern. If False, skip the guard and allow any filename to be processed.

        Returns:
            pd.DataFrame: A DataFrame containing the concatenated results of the filtered results
        """
        # Initializations
        if file_filter is None:
            # Default filter includes all files
            file_filter = lambda filename: True
        if data_frame_filter is None:
            # Default filter includes all DataFrames
            data_frame_filter = lambda df, _: df
        # Iterate across files
        dfs = []
        psd_iterator = PSDPredictionFileIterator(is_pkl=is_pkl, data_dir=data_dir,
                is_guard_filename=is_guard_filename)
        for item in psd_iterator:
            dct = util.codedstrToDict(item.filename)
            if not file_filter(dct):
                continue
            filtered_df = data_frame_filter(item.df, dct)
            if is_model:
                # Filter for model-level aggregation
                filtered_df = filtered_df[filtered_df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
            filtered_df[cn.COL_FILENAME] = item.filename  # Add the source filename
            dfs.append(filtered_df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs)

    @classmethod
    def getBiomodelSer(cls, model_name: Union[int, str],
            min_accuracy: float = 0.8,  # Minimum accuracy threshold for filtering
            accuracy_statistic: str = cn.COL_P50,  # Statistic to minimize (e.g., AUC)
            max_fractional_reduction:  Optional[float] = None,
            coefficient_threshold: Optional[float] = None,
            is_changepoint_removal: Optional[bool] = None,
            max_changepoint: Optional[int] = None,
            num_point: Optional[int] = None,
            repeat: Optional[int] = None,
            manycp: Optional[bool] = None,
            data_dir: str = cn.PSD_DATA_DIR) -> pd.Series:
        """
        Gets the PSD prediction series for a given biomodel name and optional metadata filters.

        Args:
            model_name (str): The name of the biomodel to filter for.
            accuracy_statistic (str): constraint for the accuracy statistic
            min_accuracy (float): The minimum accuracy threshold for filtering.
            max_fractional_reduction (float, optional): The maximum fractional reduction to filter for.
            coefficient_threshold (float, optional): The threshold for the coefficient to filter for.
            is_changepoint_removal (bool, optional): Whether to remove changepoints.
            max_changepoint (int, optional): The maximum number of changepoints to allow.
            num_point (int, optional): The number of points to include in the series.
            repeat (int, optional): The number of times to repeat the series.
            manycp (bool, optional): Whether to include multiple changepoints.
            data_dir (str): Directory to search for files.kkkk
        """
        if isinstance(model_name, int):
            model_name = Model.getBiomodelName(cast(int, model_name))
        ##
        # FIXME: Use names in filter filter, not dataframe columns
        def fileFilter(dct: dict) -> bool:
            ##
            def get(key: str) -> Any:
                """Helper function to get a value from the metadata dictionary."""
                return dct.get(cn.FILE_METADATA_KEY_2_COLUMN_METADATA_KEY_DCT.get(key))
            ##
            """Filter files based on the provided metadata."""
            if max_fractional_reduction is not None and get(cn.COL_MAX_FRACTIONAL_REDUCTION) != max_fractional_reduction:
                return False
            if coefficient_threshold is not None and get(cn.COL_COEFFICIENT_THRESHOLD) != coefficient_threshold:
                return False
            if is_changepoint_removal is not None and get(cn.COL_IS_CHANGEPOINT_REMOVAL) != int(is_changepoint_removal):
                return False
            if max_changepoint is not None and get(cn.COL_MAX_CHANGEPOINT) != max_changepoint:
                return False
            if num_point is not None and get(cn.COL_NUM_POINT) != num_point:
                return False
            if repeat is not None and get(cn.COL_REPEAT) != repeat:
                return False
            if manycp is not None and get(cn.COL_MANYCP) != int(manycp):
                return False
            return True
        ##
        def dataFrameFilter(df: pd.DataFrame, _) -> pd.DataFrame:
            """Filter DataFrame rows based on the model name."""
            return df[df[cn.COL_MODEL_NAME] == model_name]
        ##
        # Query the files and DataFrames
        df = PSDPredictionFileIterator.query(file_filter=fileFilter, data_frame_filter=dataFrameFilter,
                is_guard_filename=True)
        if len(df) == 0:
            raise ValueError(f"No data found for model_name={model_name} with the specified filters.")
        if len(df) == 1:
            return df.iloc[0]
        # Find the row with the minimum AUC subject to the accuracy constraint
        df = df[df[accuracy_statistic] >= min_accuracy]
        if len(df) == 0:
            raise ValueError(f"No rows found for model_name={model_name} with accuracy >= {min_accuracy}.")
        # Find the row with the minimum AUC
        best_ser = df.loc[df[cn.COL_AUC].idxmin()]
        return best_ser