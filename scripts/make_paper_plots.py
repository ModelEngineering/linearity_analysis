'''Creates the plots used in the paper.'''

from src.biomodels_iterator import BiomodelsIterator
import src.constants as cn
from src.model import Model
from src.score import Score
from src.simulator import Simulator
from src.system_discovery import SystemDiscovery, discoverNetwork
from src.timecourse import Timecourse
from src.timecourse_iterator import TimecourseIterator

IS_PLOT = False

import constants as cn
if not IS_PLOT:
    import matplotlib # type: ignore
    matplotlib.use("PDF")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import os
import pandas as pd # type: ignore
from typing import Optional

NUM_POINT = 1000

################ Data #####################
path = os.path.join(cn.DATA_DIR, "perturbation_study-0.001.csv")
df = pd.read_csv(path)
DF_P_DCT = {p: df[p] for p in ['min', 'p10', 'p50']}
#

############### Helper Functions####################

def doPlot(model_num: int, poly_degree=1, threshold=0.001, species_names: Optional[list[str]] = None,
        num_point:int=NUM_POINT):
    start_time = 0
    model = Model.makeBiomodel(model_num=model_num)
    timecourse = Timecourse(model, start_time=start_time, num_point=num_point)
    sdr = discoverNetwork(timecourse.timecourse_df, poly_degree=poly_degree, threshold=threshold,
            plot_species_names=species_names, is_plot=IS_PLOT, subtitle=f"BioModel {model_num}",
            is_plot_heatmap=False, is_print_equations=False, is_plot_comparisons=True, is_print_accuracy=False)
    return sdr

def plotPerturbation(col: str):
    dff = DF_P_DCT[col]
    dff["aggregation_type"] = "model"
    score = Score.deserialize()
    score.score_df = dff
    score.plotCDF([-0.5, -0.1, 0, 0.10, 0.50], title="CDF: " + col)

################################################
# Linear Fits
################################################
############### Time course ####################
sdr = doPlot(968, species_names=["SOCS1", "IL7IL7RJAK1"])
sdr.fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_968.pdf"), bbox_inches="tight", dpi=300) # type: ignore
##
sdr = doPlot(1004, species_names= ["IL6ext", "STAT3mRNA"])
sdr.fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_1004.pdf"), bbox_inches="tight", dpi=300) # type: ignore
#
############### CDFs ####################
path = os.path.join(cn.DATA_DIR, "linear_predictor_scores-0.001.csv")
score = Score.deserialize(path)
fig = score.plotCDF(["min", "p10", "p50", "max"], xlabel= "model accuracy", is_plot_species=False,
        is_plot_model=True, title=f"BioModel Models").fig
if IS_PLOT:
    plt.show()
fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_model_cdf.pdf"), bbox_inches="tight", dpi=300) # type: ignore
#
fig = score.plotCDF(["min", "p10", "p50", "max"], xlabel= "model accuracy", is_plot_species=True,
        is_plot_model=False, title=f"BioModel Species").fig
fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_species_cdf.pdf"), bbox_inches="tight", dpi=300) # type: ignore
if IS_PLOT:
    plt.show()

################################################
# Perturbations
################################################
############### Time course ####################
apr = SystemDiscovery.analyzePerturbations(1004, perturbations=[-50, -10, 0, 10, 50], frac_scatter_skip=0.05,
        subtitle="BioModel 1004", plot_species_names=["IL6ext","IL6int"])
apr.fig.savefig(os.path.join(cn.PAPER_DIR, "perturbation_fit_1004.pdf"), bbox_inches="tight", dpi=300) # type: ignore
#
apr = SystemDiscovery.analyzePerturbations(968, perturbations=[-50, -10, 0, 10, 50], frac_scatter_skip=0.05,
        subtitle=f"BioModel 968", plot_species_names= ["SOCS1", "IL7IL7RJAK1"])
apr.fig.savefig(os.path.join(cn.PAPER_DIR, "perturbation_fit_968.pdf"), bbox_inches="tight", dpi=300) # type: ignore

