# LightSaver

![LightSaver](img1.jpg)

**LightSaver** is a powerful data analysis package designed for fluorescent C. elegans imaging. Developed by Samuel Freitas with contributions from Raul Castro-Portugez at the University of Arizona, Sutphin Lab in the Microbiology (MCB) and Biomedical Engineering (BME) departments.

*Please note: We're actively working on both a Python version and a standalone application for enhanced accessibility.*

## Required MATLAB Packages
- 'Image Processing Toolbox'
- 'Computer Vision Toolbox'

## File Parameters Setup

![File Setup](img2.jpg)

> This directory structure is essential for the proper functioning of the `multiple_samples -> Lightsaver_batch.m` script. In this example, the overarching experiment is the "Example Experiment" folder under the data directory.

**Important Notes:**
- The script scans files recursively, sorting them by timepoint (following the nomenclature DN, Day N).
- Even if there's only a single timepoint, this directory format must still be followed, but with a single sub-experiment folder.

**Image Naming Guidelines:**
- Each image should have a descriptive name (e.g., `skn-1-HT115-EV_D1_1.tiff`, `skn-1-HT115-EV_D1_2.tiff`). The naming convention typically follows `exp-name-and-sumbnames_dayN_replicateN.tiff`.

## Usage: Automatic Data Processing/Exporting/Analyzing of an Entire Experiment (Recommended)

1. Set up data as shown above.
2. Open `Lightsaver_batch.m` under the `multiple_samples` directory.
3. Run the script (press F5 or the run button in MATLAB).
4. The parameters prompt will ask for experiment-specific details (press OK when completed).
5. Choose the overarching experiment folder in the selection prompt.
6. The script will display progress bars and export the data.
7. Check the "Exported images" folder (usually in documents/github/lightsaver) for the output. Rerun with the "Use large blob fix" flag if needed.

## Usage: Data Processing Single Sub-Experiments Individually (Not Recommended Unless Data Is Extremely Noisy and "Bad_images_fix.m" Must Be Used)

1. Open `Ligthsaver_script.m`.
2. Set parameters.
3. Run `lightsaver_script.m`.
4. Choose the directory containing the *.tiff* images.
5. Check output data if necessary.

**If there are problems:**
- Large blobs? Use the `large_blob_fix` option in `lightsaver_script.m`.
- Major issues? Employ `bad_images_fix.m`.

Now, you should find a `data.csv` file in the directory containing the *.tifs.

## Usage: Data Analysis (Automatically Analyzed When Using Recommended Settings)

1. Open and run `Data_analysis_and_export.m`.
2. Choose the overarching experiment folder from the dropdown menu.
3. Verify that "Analyzed_data.csv" is correct and the `output_figures` directory is present.
