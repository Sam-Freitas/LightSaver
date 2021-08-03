------------------------------------------------------------------------------------------------

# LightSaver

The LightSaver system is data analysis package for fluorecent C. elegans imaging 

Developed by Samuel Freitas with help from Raul Castro-Portugez

The University of Arizona, Sutphin Lab MCB, BME

------------------------------------------------------------------------------------------------

# File parameters setup

this is how the directories (folders) should be setup to use the Data_analysis_and_export.m script properly

  Overarching experiment (example: Oxidative stress)
    - Sub experiment 1 (example: day 1)
      - some directory with the *.tiff* files (example: exported images from LEICA)
    - ...
    - Sub Experiment N (example: day n)
      - some directory with the *.tiff* files (example: exported images from LEICA)

------------------------------------------------------------------------------------------------

# Usage:

1.  open "GFP_AUC_script_ez_UI.m"

2.  set parameters

3.  Run "GFP_AUC_script_ez_UI.m"

4.  Check output data if necessary

  - If there are problems

  - Large blobs?
    - use the large_blob_fix option in GFP_AUC_script_ez_UI.m

  - Just completely messed up?
    - GFP_AUC_script_ez_UI_bad_images_fix.m

5.  There should now be a data.csv file in the directory that contains the *.tifs 

