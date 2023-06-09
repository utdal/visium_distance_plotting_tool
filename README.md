# Visium Distance Plotting Tool (Euclidian Distances)

The Visium Distance Plotter is a tool that offers a range of plots for analyzing visium single-cell data(e.g. csv files).

## Installation/Setup of Visium Distance Plotting Tool:
You can install Visium Distance Plotting Tool via git:
```
git clone https://github.com/utdal/visium_distance_plots.git
```

## Functionality
It is recommended to run all the functionalities in the following order, as input to a functionality is dependent on the output of another functionality.
- Generate distance matrix
- Generate plots:
  - Male and Female Histogram Distribution Plot
  - Strip Plots of Neuronal subpopulations for MIN, MAX, MEAN and MEDIAN values
  - Co-ordinate plots/heatmaps for a tissue and for a given protein
  - Scatter Plots based on neuronal identity and gene expression
  - Heatmap on the distance matrix

Throughout the tool, all the methods are provided with necessary logging, hence one would easily be able to track all the runs and if the tool runs into any errors, one can easily debug and get to the bottom of the issue swiftly. The tool logs are contained in the directory: `/Logs`.


### Details about the functionality of each package are below.
##### Generating distance matrix
Here the distance matrixes between the barcode-of-interest and a sample dataset is calculated. Further analysis utilizes these distance matrices, including the generation of heatmaps within the pipeline.

Note: When we say distance, we refer to Euclidean distance as the visium datasets we have (X, Y) co-ordinates, that enables us to calculate the absolute distance between the two points.

##### Generate plots
The aforementioned functionality produces a variety of charts, which effectively convey the information contained in the visium data. Additionally, there are a few interactive charts available, adding another dimension to the data.

## How to run the Visium Distance Plotting Tool
We have used `click` which enables the users to interactively provide the run-time arguments if and when needed. It is understood that python is installed and a new environment is created for running this tool.


### Running the tool in an IDE
1. Open this project in PyCharm/VSCode
2. Check and create a new environment for this tool.
3. Install all the requirements i.e. as listed in the `requirements.txt`.
   > `pip install -r requirements.txt`
4. Run the file `immune_neurons_distances.py`
   Inputs needed during the run time:
   > - Final_matrix: Directory path to final matrix
   > 
   >   Ex. `/path/to/final_matrix_space_ranger`

   > - Neuronal_barcodes: Directory path to neuronal barcodes
   > 
   >   Ex. `/path/to/neurons_ident`

   > - Processed_files: Directory path where the output files should be stored
   > 
   >   Ex. `/path/to`
   >
   >   Here the output files are now stored at the same directory path where `Final_matrix` and `Neuronal_barcodes` are present 
   
   > - Barcodes_of_interest: Barcodes on which the analysis needs to be run
   >   Note: Here the barcodes should be separated with a comma, no extra spaces should be present in the input
   > 
   >   Ex. `CD4,CROT,PHTF2,USP13,ADCK1,FSCN1,SPTBN1,POLG,TSR2,UCK2,ZFAND2B`
   
   > - Scaling_factor: Scaling factor is a factor used while calculating the euclidian distances in the cdist() function.
   > 
   >   Ex. `12`
5. Provide the arguments needed interactively. Now the distance matrix's and plots are generated in the directory: `/Processed_files/runs/<unique_ten_digit_dir_code>`. Usually once the run is completed, this directory is printed out in the return message.

### Running the tool from command-line
1. Activate python environment using conda(`conda activate visdistplot`)/by running the **activate** source file.
    > **Here is how to run the source file**:
    >
    > Windows: `/path/to/python/venv/activate`
    >
    > Linux and Mac: `source /path/to/python/venv/activate`

2. Run the `immune_neurons_distances.py` file and provide the necessary inputs as mentioned in step-4 of **Running the tool in an IDE**.

If the run is successful, we get the following output;
```
# Output
{'Status': 'Success', 'Response': 'Plots generated are saved here /Users/user/Downloads/Visium_euclidean_dist/Processed_files/runs/zS2yDdHswx/Plots.'}
{'Status': 'Success', 'Response': 'Data generated is stored here; /Users/user/Downloads/Visium_euclidean_dist/Processed_files/runs/zS2yDdHswx'}
```

### Output
The output is predominantly is distance matrices and plots, should be in the `Processed_files` directory provided during 
the run-time or the plots and matrices directory path is displayed in the `response`, as shown above.

### License
License information can be found in the LICENSE file.

