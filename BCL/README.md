This folder is for generating BCL features.

For an introduction to BCL, please refer to:
https://www.frontiersin.org/articles/10.3389/fphar.2022.833099/full
# Configuration
The feature configuration files are in `BCL-configuration` folder. 

# Scripts
The scripts for generating BCL features are in the folder `BCL-scripts`. Run the commands in the following order **in the root folder**:

`1_combine_sdf.py` # This combines active and inactive SDF files into one SDF file.

`2_add_id_to_sdf.py` # This will add IDs to the header of each molecule in the SDF. So it is easy to identify which molecules are filtered out in the later steps.

`3_filter.py` # Add hydrogen, neutralize molecules and filter out molecules that do not have simple atoms (C, O, N, S, P, F, Cl, Br, I)

`4_data_preparation.py` # This creates the BCL features and stores that in the `BCL-feats` folder.

`5_count_unmatched.py` # This counts how many molecules are filtered out.

`6_clean_intermediate_files.py` # This cleans up the generated intermediate files to save hardware space.
`
