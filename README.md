# The E-H PhaseSync Model and the analysis script for behavioral data

The model was built to simulate the 3-fold periodicity of the Hippocampal BOLD signals, which were found to fluctuate as a function of movement directions in conceptual object space. Our results, described in our manuscript entitled "The spatial periodic computation of hippocampus-entorhinal circuit in navigation", with a DOI: https://doi.org/10.1101/2022.01.29.478346, suggested a causal relationship between the 3-fold periodicity found in the hippocampus and the three primary axes of the hexagonal firing pattern of grid cells. Therefore, we have uploaded the python codes to demonstrate our hypothesis on how the 3-fold periodicity in the hippocampus inherits the hexagonal signals of grid cells, thereby supporting navigation.<br /><br />


Script `Simul_gridcells.py` generates grid cell population.<br />

Script `Simul_vector_representation.py` simulates the process of the formation of the 3-fold periodicity in the hippocampus through imagination, and navigation using winner-take-all dynamics.<br />

<br />

![alt tag](https://github.com/ZHANGneuro/The-E-H-PhaseSync-Model/blob/main/model_output.png)
<br /><br />


Two analysis scripts were included in the folder `analysis script for behavioral data`, one R script and one python script.<br />
<br />
The R script `calculate_movement_directions.R` was used to calculate the subjective movement directions using a bin of 10°, while the python script `calculate_behavioral_performance_python.py` was used to calculate the behavioral performance of human subjects.<br />
<br />
To run the two analysis scripts, please download the raw behavior data at the Science Data Bank (https://doi.org/10.57760/sciencedb.18351).<br />
<br /><br />

bo <br />
2024-04-28
