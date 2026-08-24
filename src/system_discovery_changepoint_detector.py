'''Detects changepoints based of changes in the accuracy of one-step predcition of derivatives of a timecourse.'''


import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import List, Tuple  # type: ignore

from src.change_point_detector import ChangePointDetector  # type: ignore
import src.constants as cn  # type: ignore
from src.score import Score  # type: ignore
from src.system_discovery import NULL_DF, SystemDiscovery  # type: ignore


class SystemDiscoveryChangepointDetector(object):
    # Detects changepoints in the timecourse of the training data for the SystemDiscover object.

    def __init__(self, system_discovery: SystemDiscovery):
        self.system_discovery = system_discovery
        self.training_df = system_discovery.training_df
        #
        self.changepoints = []  # List of indices representing detected changepoints

    def detect(self, num_changepoint: int, min_segment_length: int = 5) -> List[int]:
        """Detect changepoints in the timecourse based on changes in the accuracy of one-step prediction of derivatives.

        Args:
            num_changepoint (int): The number of changepoints to detect.
            min_segment_length (int): Minimum length of segments to consider for changepoint detection.
        """
        raise NotImplementedError("Changepoint detection is not yet implemented.")

    def _calculateOneStepPredictionNormalizedAccuracy(self) -> pd.DataFrame:
        """Calculate the accuracy of one-step prediction of derivatives for the given timecourse in
        the normalized space.

        Returns:
            pd.DataFrame: A DataFrame containing the accuracy of one-step prediction of derivatives for each time point.
        """
        # Placeholder implementation - replace with actual logic
        return pd.DataFrame()

    def plotTimecourseWithChangepoints(self, changepoints: List[int]):
        """Plot the timecourse with detected changepoints.

        Args:
            changepoints (List[int]): A list of indices representing the detected changepoints.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_df.index, self.training_df.values, label='Timecourse')
        for cp in changepoints:
            plt.axvline(x=cp, color='r', linestyle='--', label='Changepoint' if cp == changepoints[0] else "")
        plt.title('Timecourse with Detected Changepoints')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()