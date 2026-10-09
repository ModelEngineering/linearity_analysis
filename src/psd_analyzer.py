'''Analyzes a single Piecewise System Discovery Object'''

import src.contants as cn  # type: ignore
import src.util as util  # type: ignore
from src.nl_curve import NLCurve  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore

import numpy as np # type: ignore
import pandas as pd # type: ignore


class PSDAnalyzer(object):
    """Analyzes a single Piecewise System Discovery Object."""

    def __init__(self, psd: PiecewiseSystemDiscovery) -> None:
        """Construct a PSDAnalyzer object.

        Args:
            psd (PiecewiseSystemDiscovery): The piecewise system discovery object to analyze.
        """
        self._psd = psd
        self._psd.fit()
        self._boundaries = self._psd.boundaries
        self._nl_curve = NLCurve(self._boundaries)