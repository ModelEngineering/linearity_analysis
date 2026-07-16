# Extracting Characteristic Times from Papers for BioModels

You are an expert in Systems Biology, especially mechanistic models. You want to find the characteristic times used in several models in BioModels. The target models are those that have not specified the simulation endtime in their SEDML. A list of model metadata is contained in `@data/biomodels_endtime.csv` (note: this file contains metadata about each model, including whether a SEDML end time has been specified). The models you want to investigate are those that do *not* have SEDML as their ``end_time_source``.

To find an appropriate endtime for a model, use the **model ID** from `@data/biomodels_endtime.csv` (e.g., ``BIOMD0000000001``, where "1" is the model number). Start by reading the curation notes at `@temp-biomodels/final/<model ID>/curation_notes.txt`. Then, locate the web page for the model at ``http://www.biomodels.org/models/<model ID>``. Navigate to the model's **Overview** tab and find the **primary citation** — if multiple papers are cited, choose the one most directly related to the model's development or the simulations described. Read that paper carefully, especially its figures, to determine the endtimes used in simulations. Note that not every figure represents a time course (e.g., steady-state snapshots, parameter scans, or spatial distributions are not time courses).

From this information, you will create a CSV file at `@data/imputed_characteristic_times.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `model_id` | The BioModels ID for the model (e.g., ``BIOMD0000000001``) |
| `endtime` | The end time used in the paper, expressed as a number. If multiple figures suggest different endtimes, include one row per distinct endtime. |
| `unit` | The time unit used in the paper (e.g., "s", "min", "hr"). If unspecified, write "unknown". |
| `justification` | A description of how you determined this end time (e.g., which figure or text passage was used). |

The file must include a header row and use comma as the delimiter. Use UTF-8 encoding.

## Additional Notes

* **Conflicting endtimes:** You may have multiple rows for a single model if different figures or simulations in the same paper use different end times.
* **Undetermined endtime:** If the end time cannot be determined from the paper, leave `endtime` blank and provide as much detail as possible in the `justification` field about what information was consulted (e.g., "Figure 2 shows a time course but does not label the x-axis range").
* **Time units:** Papers may report times in seconds ("s"), minutes ("min"), or hours ("hr"). Record the unit exactly as stated in the paper; do not convert. If the unit is not mentioned, write "unknown".