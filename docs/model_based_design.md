# Model-Based Design

This writeup describes a new design for modules in the project based on a better understanding of the following requirements:

* separating static and dynamic aspects of models;
* the need to use the same dynamics when splitting a model since we are not guaranteed to get the same numerical outputs on subsequent runs; and
* better balancing the size of modules.

Refer to archive/l_roadrunner.py and archive/trajectory.py for algorithmic approaches. Use @plot_options for plotting. Do not change the files in archive; they are for reference only.

## Class Architecture

### ``Model``

This class contains properties of the static model, those properties derived from the model specification (not its execution).
Included are the properties:

* ``model_name``
* ``sbml_str`` is an SBML string for the model. Antimony models are converted to SBML using roadrunner.
* Various static properties, such as ``species_names``, ``num_species``
* ``__eq__`` which checks the ``model_name`` and ``sbml_str``

StaticModel uses roadrunner transiently to query the underlying model (e.g., names of floating species).

### ``Trajectory``

This class is a container for properties obtained running a simulation.

The constructor has arguments for: static model, and the following:

* ``jacobian_collection_arr`` the Jacobians at each of the timepoints
* ``timepoint_arr`` is the times at which simulation results are reported. The first timepoint is used to obtain initial values for the Trajectory.
* ``forcing_input_collection_arr`` an array of arrays of forced inputs calculated at each timepoint
* ``timecourse_df`` is the timecourse for the dynamics

The following are properties calculated from the properties above:

* ``jacobian_median_arr`` is the median of the values in ``jacobian_collection_arr``, a computed property.
* ``jacobian_std_arr`` is the standard deviation of Jacobian values, a computed property.
* ``start_time``, ``end_time``, ``num_point``

The following are external methods:

* ``makeSubmodel(start_time, end_time)`` uses a subset of the dynamical data (as specified by start and end) to construct a new dynamical model that copies of subset of the data in the current model into the new model. This method uses slicing to obtain values for constructing a new DynamicModel.
* ``makeFromSimulation(start_time, end_time, num_point, StaticModel)`` is a class method that runs simulations to obtain the arguments for the DynamicModel constructor. This is the only method in DynamicModel that uses roadrunner. If the model_name begins with "BIOMD", then: (a) the model is in BIOMODELS_DIR and (b) ``end_time`` is obtained from the module ``biomodels_iterator``. ``end_time=None`` triggers autodetect of ``end_time``.
* ``_makeEndtime`` is a method formerly in LRoadrunner.
* ``__eq__`` which checks that the constructor arguments are the same

### ``LinearPredictor``

This class does linear prediction and evaluations of these predictions. It is constructed with a ``DynamicModel``, ``jacobian_selection`` and ``num_step``, the number of steps ahead to do the prediction. It has the following methods

* ``predict`` provides predictions for the timecourse of the ``DynamicModel``.
* ``score`` scores the prediction using the Score class.
* ``plotPrediction`` plots the timecourse and the prediction from the ``start_time`` to the ``end_time``
* ``cost`` is a property of a LinearPredictor and has a value of type float. For a single species, cost is mean over timepoints of the squared difference between the predicted and actual (simulated) values, where each is divided by the actual value. For multiple species, the median of the species costs is calculated.

### ``Score``

* We will use ``makeScoreInfo`` to construct the various score metrics to evaluate the quality of a prediction.

### ``TrajectoryCollection``

This class represents a collection of ``Trajectory`` with the same
``Model``. Methods include:

* Constructor takes a list of ``Tracjectory`` all of which have the same ``Model``. Error checking is done to ensure that timepoints do not overlap. The list is sorted using ``Trajectory.__lt__``. The constructor does not verify that adjacent ``Trajectory`` overlap their end_time and start_time.
* ``plotTimecourse`` pieces together the timecourse for each ``DynamicModel``, and plots it with a vertical dashed line separating each ``Trajectory``. The arguments to this method are the kwards used by PlotOptions. Internally, the method uses PlotOptions. It should return PlotOptions.
* ``__eq__`` which checks if it's the same as another ``TrajectoryCollection`` by comparing each ``Trajectory``.
* ``split`` is a class method that takes as input timepoints to create multiple Trajectory objects. Its signature is ``split(cls, trajectory: Trajectory, timepoints: List[float])``.
  * A timepoint specifies the last time for the preceeding Trajectory and the first time for the next Trajectory.
  * Returns ``TrajectoryCollection``.
  * If a split time falls between existing timepoints (i.e., not exactly in timepoint_arr), it snaps to the nearest timepoint.
* ``autoSplit`` uses dynamic programming to find the best split in terms of minimizing the sum of the LinearPredictor costs for each Trajectory in the collection. The inputs to this method are: (a) Trajectory; and (b) the number of splits to construct; (c) ``num_step``, (d) ``jacobian_selection``. The output is a TrajectoryCollection.

### ``MultipleLinearPredictor``

This class performs piece-wise linear prediction. The times at which
there is a partition of the linear model is a "split point".

* Constructor has the arguments ``TrajectoryCollection``, ``jacobian_selection``, and ``num_step``, the number of steps ahead for which prediction is done.
* ``predict`` uses LinearPredi:w
* ctor.predict for each Trajectory, dropping the starting timepoint of interior segements.
* ``score`` constructs the concatenation of the predictions for each split and similarly the concatenation of timecourse_df for each segment.
* ``plotPrediction`` The plot shows predicted and actual (simulated) values with vertical dashed lines to indicate regions for submodels. It has arguments PlotOptions and should use PlotOptions internally.
* ``cost`` property is calculated using the predictions and timecourse used in score.

### Prompts

1. Review the design and provide comments on where confusions exist as well as where improvements can be made. Do not implement yet.
1. Implement StaticModel as described in @model_based_design.md and associated tests.
1. Implement DynamicModel as described in @model_based_design.md and associated tests.
1. Update the modules in scripts to use.
