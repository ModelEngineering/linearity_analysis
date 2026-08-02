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

    def __init__(self, serialization_path: str, is_initialize: bool = False,
            is_persist: bool = True) -> None:
        """
        Parameters
        ----------
        path : str
            Path to the CSV serialization file.
        is_initialize : bool
            Whether to initialize the CSV file by writing an empty DataFrame with the appropriate columns.
        is_persist : bool
            Whether to persist the DataFrame to the CSV file.
        """
        self.serialization_path = serialization_path
        self.is_persist = is_persist
        if self.is_persist:
            if is_initialize:
                self.dataframe: pd.DataFrame = pd.DataFrame()
            elif os.path.exists(serialization_path):
                self.dataframe = pd.read_csv(serialization_path)
            else:
                self.dataframe = pd.DataFrame()
        else:
            self.dataframe = pd.DataFrame()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataframeSerializer):
            return NotImplemented
        return self.dataframe.equals(other.dataframe)

    def serializeDct(self, dicts):
        """Appends rows to self.dataframe and writes the result to the CSV.

        Accepts either a dict-of-lists (e.g. {"mean": [0.25], "aggregation_type": ["model"]})
        or an iterable of row dictionaries (e.g. [{"mean": 0.25, "aggregation_type": "model"}]).

        Parameters
        ----------
        dicts : dict or Iterable[Dict]
            Collection of data to serialize. When self._df is non-empty,
            every dict must have exactly the same keys as the existing columns.

        Raises
        ------
        ValueError
            If self._df is non-empty and the dict keys do not match its columns.
        """
        new_df = pd.DataFrame(dicts)
        return self.serializeDf(new_df)

    def serializeDf(self, new_df: pd.DataFrame) -> None:
        """
        Appends the dataframe to the same information.

        Parameters
        ----------
        new_df : pd.DataFrame
            DataFrame to append. When self._df is non-empty,
            the columns must match the existing columns.

        Raises
        ------
        ValueError
            If self._df is non-empty and the dict keys do not match its columns.
        """
        if not self.dataframe.empty:
            existing_cols = set(self.dataframe.columns)
            new_cols = set(new_df.columns)
            if new_cols != existing_cols:
                raise ValueError(
                        f"Dict keys {new_cols} do not match existing columns "
                        f"{existing_cols}.")
        self.dataframe = pd.concat([self.dataframe, new_df], ignore_index=True)
        if self.is_persist:
            self.dataframe.to_csv(self.serialization_path, index=False)

