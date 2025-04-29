# The E-H PhaseSync Model

The model was built to simulate the 3-fold periodicity of Hippocampal activity, which was found in the manuscript entitled "The spatial periodic computation of hippocampus-entorhinal circuit in navigation", with a DOI: https://doi.org/10.1101/2022.01.29.478346. 

<br />

The paper supports a causal relationship between the 3-fold periodicity found in the hippocampus and the three primary axes of grid cells in the entorhinal cortex, while the python codes demonstrate how the 3-fold periodicity in the hippocampus inherits the populational activity of grid cells, thereby supporting navigation.<br /><br />


Script `Simul_gridcells.py` generates grid cell population.<br />

Script `Simul_vector_representation.py` simulates the process of the formation of the 3-fold periodicity in the hippocampus through imagination, and navigation using winner-take-all dynamics.<br />

<br />

![alt tag](https://github.com/ZHANGneuro/The-E-H-PhaseSync-Model/blob/main/model_output.png)
<br /><br />

--------------------------------
Two analysis scripts were included in the folder `analysis script for behavioral data`, one R script and one python script.<br />
<br />
The R script `calculate_movement_directions.R` was used to calculate the subjective movement directions using a bin of 10°, while the python script `calculate_behavioral_performance_python.py` was used to calculate the behavioral performance of human subjects.<br />
<br />
To run the two analysis scripts, please download the raw behavior data at the Science Data Bank (https://doi.org/10.57760/sciencedb.18351).<br />
<br /><br />

bo <br />
2024-04-28
