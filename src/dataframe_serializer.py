"""Serializes and deserializes a DataFrame to and from a CSV file."""

import os  # type: ignore
from typing import Dict, Iterable  # type: ignore

import pandas as pd  # type: ignore


#########################################
class DataframeSerializer:
    """Persists a growing DataFrame to a CSV file.

    On construction the CSV is read if it already exists. serialize()
    appends new rows and rewrites the file; deserialize() reconstructs
    an instance from an existing file.
    """

    def __init__(self, path: str, is_initialize: bool = False) -> None:
        """
        Parameters
        ----------
        path : str
            Path to the CSV serialization file.
        is_initialize : bool
            Whether to initialize the CSV file by writing an empty DataFrame with the appropriate columns.
        """
        self._path = path
        if is_initialize:
            self.dataframe: pd.DataFrame = pd.DataFrame()
        elif os.path.exists(path):
            self.dataframe = pd.read_csv(path)
        else:
            self.dataframe = pd.DataFrame()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataframeSerializer):
            return NotImplemented
        return self.dataframe.equals(other.dataframe)

    def serialize(self, dicts: Iterable[Dict]) -> None:
        """Appends rows to self._df and writes the result to the CSV.

        Parameters
        ----------
        dicts : Iterable[Dict]
            Collection of row dictionaries. When self._df is non-empty,
            every dict must have exactly the same keys as the existing columns.

        Raises
        ------
        ValueError
            If self._df is non-empty and the dict keys do not match its columns.
        """
        new_df = pd.DataFrame(list(dicts))
        if not self.dataframe.empty:
            existing_cols = set(self.dataframe.columns)
            new_cols = set(new_df.columns)
            if new_cols != existing_cols:
                raise ValueError(
                        f"Dict keys {new_cols} do not match existing columns "
                        f"{existing_cols}.")
        self.dataframe = pd.concat([self.dataframe, new_df], ignore_index=True)
        self.dataframe.to_csv(self._path, index=False)

