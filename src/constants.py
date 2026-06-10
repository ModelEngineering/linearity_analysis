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

# Constants
START_TIME = 0.0
END_TIME = 10.0
NUM_POINTS = 10*int(END_TIME - START_TIME) + 1

# Diameter metrics
DIAMETER_IVP = "weighted_eigenvectors"
DIAMETER_MAX_CV = "max_cv"

# Columns
COL_MAXCV = "max_cv"
COL_ENDTIME = "end_time"
COL_MODEL_NAME = "model_name"
COL_ENDTIME_SOURCE = "end_time_source"  # How end_time was determined (e.g. "reciprocal_min_eigenvalue", "default")
COL_NUM_REACTION = "num_reaction"
COL_NUM_SPECIES = "num_species"
COL_NUM_PERTURBATION = "num_perturbation"
COL_NUM_TIMEPOINT = "num_timepoint"
ENDTIME_SOURCE_RECIROCAL_MIN_EIGENVALUE = "reciprocal_min_eigenvalue"
ENDTIME_SOURCE_SEDML = "sedml"
ENDTIME_SOURCE_STEADYSTATE = "steadystate"
ENDTIME_SOURCE_MAX_MEDIAN_CV = "max_median_cv"
ENDTIME_SOURCE_USER_SPECIFIED = "user_specified"
COL_NAMES = [COL_MODEL_NAME, COL_MAXCV, COL_ENDTIME, COL_ENDTIME_SOURCE]

# Common default values
NULL_ARRAY = np.array([])

# Jacobian selection
JAC_FITTED = "fit_gershgorin" # Fit a Jacobian by using the timecourse for each row
JAC_MEDIAN = "median"  # Use the median Jacobian
JAC_FIRST = "first" # Use the first Jacobian

# Perturbation parameters for Jacobian fitting
PERTURBATION_VALUE_FRACTION = 0.0  # Perturb each point
PERTURBATION_SPECIES_FRACTION = 0.5  # Perturb this fraction of points