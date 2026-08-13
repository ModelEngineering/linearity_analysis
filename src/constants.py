'''Constants used in the project'''

import numpy as np  # type: ignore
import os
import tellurium as te  # type: ignore

# Directories and paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REPO_DIR = os.path.dirname(PROJECT_DIR)
BIOMODELS_DIR = os.path.join(REPO_DIR, "temp-biomodels", "final")
CALCULATED_ENTIMES_PATH = os.path.join(DATA_DIR, "biomodels_endtime.csv")
SERIALIZATION_DIR = os.path.join(PROJECT_DIR, "data", "serialize")
TIMECOURSE_SERIALIZATION_DIR = os.path.join(SERIALIZATION_DIR,
        "timecourse")
TIMECOURSE_ZIP_PATH = os.path.join(TIMECOURSE_SERIALIZATION_DIR, "timecourse.zip")

# Types
TYPE_ROADRUNNER = "tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner"
NULL_ROADRUNNER = te.loada("")

# Default values
START_TIME = 0.0
END_TIME = 10.0
NUM_POINT = 1000
SYSTEM_DISCOVERY_THRESHOLD = 0.01

# Diameter metrics
DIAMETER_IVP = "weighted_eigenvectors"
DIAMETER_MAX_CV = "max_cv"

# Columns
COL_AGGREGATION_TYPE = "aggregation_type"  # model or species name
COL_AGGREGATION_TYPE_MODEL = "model"
COL_MAXCV = "max_cv"
COL_ENDTIME = "end_time"
COL_MODEL_NAME = "model_name"
COL_ENDTIME_SOURCE = "end_time_source"  # How end_time was determined (e.g. "reciprocal_min_eigenvalue", "default")
COL_NUM_REACTION = "num_reaction"
COL_NUM_SPECIES = "num_species"
COL_NUM_PERTURBATION = "num_perturbation"
COL_NUM_TIMEPOINT = "num_timepoint"
COL_NAMES = [COL_MODEL_NAME, COL_MAXCV, COL_ENDTIME, COL_ENDTIME_SOURCE]
COL_SYSTEM_ID = "system_id"  # Unique identifier for the system, e.g. model name or species name.
COL_MEAN = "mean" # Mean value of the valid values used in calculating the statistics.
COL_MIN = "min"  # Minimum value of the valid values used in calculating the statistics.
COL_MAX = "max"  # Maximum value of the valid values used in calculating the statistics.
COL_COUNT = "count"  # Number of valid values used in calculating the statistics.
COL_INVALID_COUNT = "invalid_count"  # Number of invalid (sentinel -1) values excluded from aggregation.
COL_LABEL = "label"  # Unique identifier for the row of statistics, e.g. model name or species name.
COL_P05 = "p05"  # 5th percentile of the valid values used in calculating the statistics.
COL_P10 = "p10"  # 10th percentile of the valid values used in calculating the statistics.
COL_P20 = "p20"  # 20th percentile of the valid values used in calculating the statistics.
COL_P25 = "p25"  # 25th percentile of the valid values used in calculating the statistics.
COL_P30 = "p30"  # 30th percentile of the valid values used in calculating the statistics.
COL_P50 = "p50"  # 50th percentile of the valid values used in calculating the statistics.
COL_P80 = "p80"  # 80th percentile of the valid values used in calculating the statistics.
COL_P90 = "p90"  # 90th percentile of the valid values used in calculating the statistics.
COL_P95 = "p95"  # 95th percentile of the valid values used in calculating the statistics.
COL_P99 = "p99"  # 99th percentile of the valid values used
COL_PERCENTILES = [COL_P05,COL_P10, COL_P20, COL_P25, COL_P30, COL_P50, COL_P80, COL_P90, COL_P95, COL_P99]
STATISTICS = [COL_MEAN, COL_MIN, COL_MAX, COL_COUNT, COL_INVALID_COUNT] + COL_PERCENTILES

# Symbolic values
ENDTIME_SOURCE_RECIROCAL_MIN_EIGENVALUE = "reciprocal_min_eigenvalue"
ENDTIME_SOURCE_SEDML = "sedml"
ENDTIME_SOURCE_STEADYSTATE = "steadystate"
ENDTIME_SOURCE_MAX_MEDIAN_CV = "max_median_cv"
ENDTIME_SOURCE_USER_SPECIFIED = "user_specified"

# Common default values
NULL_ARRAY = np.array([])

# Jacobian selection
JAC_FITTED = "fit_gershgorin" # Fit a Jacobian by using the timecourse for each row
JAC_MEDIAN = "median"  # Use the median Jacobian
JAC_FIRST = "first" # Use the first Jacobian

# Perturbation parameters for Jacobian fitting
PERTURBATION_VALUE_FRACTION = 0.0  # Perturb each point
PERTURBATION_SPECIES_FRACTION = 0.5  # Perturb this fraction of points

# Endtime fraction
ENDTIME_FRACTION_STEADYSTATE = 0.1  # Fraction of endtime to use for timecourse analysis
ENDTIME_FRACTION_MAXMEDIAN = 0.1  # Fraction of endtime to use for timecourse analysis
