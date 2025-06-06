# The E-H PhaseSync Model

The model was built to simulate the 3-fold periodicity of Hippocampal activity, which was found in the manuscript entitled "The spatial periodic computation of hippocampus-entorhinal circuit in navigation", DOI: https://doi.org/10.1101/2022.01.29.478346. 

<br />

The main finding of this paper demonstrates that the 3-fold periodicity of the hippocampus is the result of the activity projections of grid cell population in the entorhinal cortex.<br />

## Python environment for grid cell simulation: <br />
python `3.9.16` <br />
matplotlib `3.9.4` <br />
scipy `0.16.0` <br />
moviepy `2.2.1` <br />
imageio `2.37.0` <br />

<br />
Script `Simul_gridcells.py` is for the generation of grid cell population .<br />

Script `Simul_vector_representation.py` simulates the process of the formation of the 3-fold periodicity in the hippocampus during mental planning<br />

<br />

![alt tag](https://github.com/ZHANGneuro/The-E-H-PhaseSync-Model/blob/main/model_output.png)
<br /><br />

--------------------------------
Two analysis scripts were included in the folder `analysis script for behavioral data`, one R script and one python script.<br />
<br />
The R script `calculate_movement_directions.R` was used to calculate the movement directions using a bin of 10°, while the python script `calculate_behavioral_performance_python.py` was used to calculate the behavioral performance of human participants.<br />
<br />
To run the two analysis scripts, please download the raw behavior data at the Science Data Bank (https://doi.org/10.57760/sciencedb.18351).<br />
<br /><br />

bo <br />
2024-04-28
