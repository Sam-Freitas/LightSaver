# LightSaver

------------------------------------------------------------------------------------------------

![LightSaver](img1.jpg)

------------------------------------------------------------------------------------------------

The LightSaver system is data analysis package for fluorecent C. elegans imaging 

Developed by Samuel Freitas with help from Raul Castro-Portugez

The University of Arizona, Sutphin Lab Microbiology (MCB), Biomedical Engineering (BME)

------------------------------------------------------------------------------------------------

# File parameters setup

this is how the directories (folders) should be setup to use the multiple_samples -> Lightsaver_batch.m (recommended) script properly

  - Overarching experiment (example: Oxidative stress )
  
    - Sub experiment 1 (example: day 1 or 1/1/1995)
    
      - some directory with the *.tiff* files (example: exported images from LEICA 
      - Note: this script recursively scans all possible files for every .*tiff file)
      
    - ...
   
    - Sub Experiment N (example: day n or 1/N/1995)
    
      - some directory with the *.tiff* files (example: exported images from LEICA)
      

------------------------------------------------------------------------------------------------

# Usage: Data processing/exporting an entire experiment (recommended)
> Batching an entire experiment at once, and exporting for easy plotting in Prism(TM) 

1.  Set up data as shown in the above example

2.  Open "Lightsaver_batch.m" under the "multiple_samples" directory

3.  Run "Lightsaver_batch.m" by either pressing F5 or the run button at the top of MATLAB

4.  The parameters prompt will open and ask for experiment specific parameters (press ok when completed)

5.  The experiment selection prompt will open, please select the Overarching experiment folder (as specified above)

6.  The script will then display progress bars and where all the data gets exported to

------------------------------------------------------------------------------------------------

# Usage: Data processing single sub-experiments individually (not recommended unless data is extremely noisy and "Bad_images_fix.m" must be used)

1.  open "GFP_AUC_script_ez_UI.m"

2.  set parameters

3.  Run "GFP_AUC_script_ez_UI.m"

4.  Choose the directory containing the *.tiff* images

5.  Check output data if necessary

  - If there are problems

  - Large blobs?
    - use the large_blob_fix option in GFP_AUC_script_ez_UI.m

  - Just completely messed up?
    - GFP_AUC_script_ez_UI_bad_images_fix.m

6.  There should now be a data.csv file in the directory that contains the *.tifs 

# Usage: data analysis

1.  open and run "Data_analysis_and_export.m"

2.  Choose the overarching experiment folder from the dropdown menu

3.  Check to make sure the "Analyzed_data.csv" is correct and the output_figures directory is present


