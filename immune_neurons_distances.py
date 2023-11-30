# Imports
from logger import my_logger
import constants as constants
from datetime import datetime
from scipy.spatial import distance
import pandas as pd, matplotlib as mpl, os, click, random, string, json
from generate_plots import plot_immune_neur_distances, generate_scatter_plot_for_final_neuronal_barcodes


# Run this file and provide the necessary metadata.csv along with Final_matrix & Neuronal_ident files to
# generate the plots.
# Note: pip install -r requirements.txt will take care of all the dependency packages.

def calculate_distances(neurons_df, final_df, dist_path, stat_path):
    try:
        my_logger.info("<---- Calculate distances function is initiated ---->")
        temp_df = pd.DataFrame((distance.cdist(neurons_df, final_df, 'euclidean')),
                               index=neurons_df.index, columns=final_df.index)
        temp_df.to_csv(dist_path, index=False)
        temp_df['mean'], temp_df['median'] = temp_df.mean(axis=1), temp_df.median(axis=1)
        temp_df['min'], temp_df['max'] = temp_df.min(axis=1), temp_df.max(axis=1)
        temp_df.to_csv(stat_path)
        temp_df.to_csv(stat_path, index=False)
        my_logger.info("<---- Calculate distances function run is completed ---->")
        return temp_df

    except Exception as error_msg:
        my_logger.exception(error_msg)
        return error_msg


def random_generator(run_path):
    """
    This function generated a random 10 character-length string, which is then used to create a directory.
    :param run_path: Directory path where the directory needs to be created to check and generate a unique random string
    :return: returns a unique 10 char-length string
    """
    try:
        my_logger.info("<---- Random generator function is initiated ---->")
        # Checking and generating random variable of length 10
        check_list = list()
        for directory in os.listdir(run_path):
            check_list.append(directory)
        # Random char-num generator of length 10
        while True:
            random_variable = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            if random_variable not in check_list:
                break
        my_logger.info("<---- Here is a randomly generated directory: {}".format(random_variable))
        my_logger.info("<---- Random generator function run is completed ---->")
        return random_variable

    except Exception as error_msg:
        my_logger.exception(error_msg)
        return error_msg


@click.command()
@click.option("--final_matrix_fp", prompt="Enter file-path for Final_matrix directory", type=str,
              help='Directory path to final matrix - Final_matrix')
@click.option("--neuronal_barcodes_identity_fp", prompt="Enter file-path for Neuronal_barcode directory", type=str,
              help='Directory path to neuronal barcodes - Neuronal Barcodes')
@click.option("--file_path", prompt="Enter file-path where Processed_files directory needs to be generated",
              type=str, help='Directory path where processed files need to be saved')
@click.option("--barcodes_of_interest", prompt="Enter the barcodes of interest (Ex. CROT,PHTF2,USP13,FSCN1,etc.)",
              type=str, help='Directory path where processed files need to be saved', default=None)
# @click.option("--scaling_factor", prompt="Scaling factor for the distance matrix",
#               type=int, help='Scaling factor for the distance matrix', default=100)
def generate_immune_neuronal_distance_matrix(final_matrix_fp: str,
                                             neuronal_barcodes_identity_fp: str,
                                             file_path: str,
                                             barcodes_of_interest: str):
    """
        This method generated visium plots
        :param final_matrix_fp:
        :param neuronal_barcodes_identity_fp:
        :param file_path:
        :param barcodes_of_interest:Genes separated by a comma; Ex.: SNAP25,OSM

               Note: There should be no spaces in the barcodes_of_interest

        :return: A response template with Status: success or failed
    """
    try:
        start_time = datetime.now()
        label_size = 30
        response_template = constants.response_template
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams["font.sans-serif"] = ["Arial"]
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        processed_files_dir = file_path + os.sep + constants.PROCESSED_FILES
        runs_fp = processed_files_dir + os.sep + constants.RUNS
        col_names = constants.COLNAMES_VISIUM_SINGLE_NEURONS

        my_logger.info("<---- Generating immune neural distance matrix's and plots started ---->")

        # Exception handling
        if not os.path.exists(final_matrix_fp):
            raise Exception("Final_matrix path: {} does not exist".format(final_matrix_fp))
        if not os.path.exists(neuronal_barcodes_identity_fp):
            raise Exception("Neuronal_barcodes path: {} does not exist".format(neuronal_barcodes_identity_fp))

        # Checking the corresponding file(s)
        if not len(os.listdir(final_matrix_fp)) == len(os.listdir(neuronal_barcodes_identity_fp)):
            raise Exception(constants.FILES_NOT_MATCHING)

        # Checking and creating the directories
        if not os.path.exists(runs_fp):
            os.makedirs(runs_fp)

        rand_str = random_generator(runs_fp)
        run_id_path = runs_fp + os.sep + rand_str
        matrix_fp = runs_fp + os.sep + rand_str + os.sep + constants.MATRIX_DIR
        plots_data_dir = runs_fp + os.sep + rand_str + os.sep + constants.PLOTS_DIR
        plots_filepath = runs_fp + os.sep + rand_str + os.sep + constants.PLOTS

        if not os.path.exists(matrix_fp):
            os.makedirs(matrix_fp)
        if not os.path.exists(plots_data_dir):
            os.makedirs(plots_data_dir)
        if not os.path.exists(plots_filepath):
            os.makedirs(plots_filepath)

        print("Triggering the function to generate distance matrix's, running for samples: {}".format(
            sorted(list(set(barcodes_of_interest.split(','))))))
        my_logger.info("Triggering the function to generate distance matrix's, running for samples: {}".format(
            sorted(list(set(barcodes_of_interest.split(','))))))

        my_logger.info("Here are the filepaths and variables provided for this run:")
        my_logger.info("Final matrix filepath: {}".format(final_matrix_fp))
        my_logger.info("Neuronal barcodes identity filepath: {}".format(neuronal_barcodes_identity_fp))
        my_logger.info("Processed files filepath: {}, this is where the output files and plots are saved".format(file_path))
        my_logger.info("Barcodes of interest: {}".format(barcodes_of_interest))

        metadata_df = pd.read_csv('metadata.csv', names=[constants.final_mat, constants.neuronal_barcodes,
                                                         constants.SEX, constants.scaling_fac], skiprows=1)

        if barcodes_of_interest is not None:
            for final_matrix_file, neuronal_identity_file, scaling_factor_dir in zip(metadata_df[constants.final_mat],
                                                                                 metadata_df[constants.neuronal_barcodes],
                                                                                 metadata_df[constants.scaling_fac]):
                # Scaling Factor
                with open(os.path.join(scaling_factor_dir, [i for i in os.listdir(scaling_factor_dir) if i.endswith(".json")][0]), "r") as f:
                    json_data = json.load(f)
                scaling_factor = 65 / json_data['spot_diameter_fullres']
                my_logger.info("Scaling factor: {}".format(final_matrix_fp))

                # Generate pixel co-ordinates
                final_matrix_df = pd.read_csv(os.path.join(final_matrix_fp, final_matrix_file), names=col_names, header=None)
                pixel_df = pd.read_csv(os.path.join(scaling_factor_dir, [i for i in os.listdir(scaling_factor_dir) if i.endswith(".csv")][0]),
                                       names=["Barcodes", "Z", "X", "Y", "x_pixel", "y_pixel"])
                temp_pixel_mer_df = final_matrix_df.merge(pixel_df, on="Barcodes", how="left")   # final_matrix_df.merge(pixel_df, on="Barcodes", how="left").head()
                final_matrix_df = temp_pixel_mer_df[["Barcodes", "x_pixel", "y_pixel", "Gene ID", "Gene Name", "Gene Expression"]]
                final_matrix_df = final_matrix_df.rename(columns={"x_pixel": "X", "y_pixel": "Y"})

                mat, neur = str(final_matrix_file.split('.')[0]), str(neuronal_identity_file.split('.')[0])
                neuronal_barcode_df = pd.read_csv(os.path.join(neuronal_barcodes_identity_fp, neuronal_identity_file))
                neurons_list = neuronal_barcode_df[constants.BARCODES].to_list()
                # Filtering final matrix based on the neuron indent file barcodes
                neurons_df = final_matrix_df[final_matrix_df[constants.BARCODES].isin(neurons_list)]

                for barcode in sorted(list(set(barcodes_of_interest.split(',')))):
                    print("Running: {} - {}, for barcode {}".format(final_matrix_file, neuronal_identity_file, barcode))
                    my_logger.info("Running: {} - {}, for barcode {}".format(final_matrix_file, neuronal_identity_file,
                                                                             barcode))
                    try:
                        updated_final_matrix_df = final_matrix_df.loc[final_matrix_df[constants.GENENAME] == barcode]
                        if updated_final_matrix_df.empty:
                            print("<---- This gene is not found in the current data: {} ---->".format(barcode))
                            my_logger.warning("<---- This gene is not found in the current data: {} ---->".format(barcode))
                            continue
                    except Exception as error_msg:
                        my_logger.info(error_msg)
                        raise Exception(
                            "Barcode {} does not exist in the data, please try again with a different barcode.".format(
                                barcode))

                    # Scatter Plots - FinalMatrix, NeuronalBarcodes and Barcode
                    generate_scatter_plot_for_final_neuronal_barcodes(final_matrix_fp, neuronal_barcodes_identity_fp,
                                                                      final_matrix_file, neuronal_identity_file, barcode,
                                                                      plots_filepath)

                    '''
                    
                    
                    Additional code can be added here, where final_matrix and neuronal_indent files are needed
                    
                    
                    '''

                    updated_final_matrix_df = updated_final_matrix_df.loc[:, [constants.BARCODES,
                                                                              constants.X, constants.Y]]
                    neurons_coord_df = neurons_df.loc[:, [constants.BARCODES, constants.X,
                                                          constants.Y]].drop_duplicates()
                    neurons_coord_df.set_index(constants.BARCODES, inplace=True)
                    updated_final_matrix_df.set_index(constants.BARCODES, inplace=True)

                    temp_df = pd.DataFrame(
                        (distance.cdist(neurons_coord_df, updated_final_matrix_df, 'euclidean') * scaling_factor),
                        index=neurons_coord_df.index, columns=updated_final_matrix_df.index)

                    temp_dist_file_path = os.path.join(matrix_fp, mat + '_' + neur + '_' + str(barcode) + '_distance_matrix' + constants.CSV_FILE)
                    temp_stats_file_path = os.path.join(matrix_fp, mat + '_' + neur + '_' + str(barcode) + '_stats' + constants.CSV_FILE)

                    if not os.path.exists(os.path.join(plots_data_dir, barcode)):
                        os.mkdir(os.path.join(plots_data_dir, barcode))
                    temp_dist_plots_path = os.path.join(plots_data_dir, barcode, mat + '_' + neur + '_distance_matrix' + constants.CSV_FILE)
                    temp_stats_plots_path = os.path.join(plots_data_dir, barcode, mat + '_' + neur + '_stats' + constants.CSV_FILE)
                    temp_df.to_csv(temp_dist_file_path)
                    temp_df.to_csv(temp_dist_plots_path)
                    temp_df[constants.MEAN], temp_df[constants.MEDIAN] = temp_df.mean(axis=1), temp_df.median(axis=1)
                    temp_df[constants.MIN], temp_df[constants.MAX] = temp_df.min(axis=1), temp_df.max(axis=1)
                    temp_df = temp_df.loc[:, [constants.MEAN, constants.MEDIAN, constants.MIN, constants.MAX]]
                    temp_df[constants.IDENT] = [
                        neuronal_barcode_df[neuronal_barcode_df[constants.BARCODES] == i][constants.IDENT].values[0]
                        for i in temp_df.index]
                    temp_df[constants.CELL_TYPE] = constants.NEURONS
                    temp_df.to_csv(temp_stats_file_path)
                    temp_df.to_csv(temp_stats_plots_path)

        else:
            raise Exception("No barcodes are provided to compare with the given data.")

        plot_immune_neur_distances(run_id_path.replace("\\", "/"), final_matrix_fp,
                                   neuronal_barcodes_identity_fp, barcodes_of_interest)  # plots func call

        my_logger.info("<---- Generating immune neural distance matrix's and plots completed ---->")
        my_logger.info("Generated results are stored to {}".format(run_id_path.replace("\\", "/")))
        my_logger.info("Total time take to generate immune neural distance files and plots is {}".format(
            datetime.now()-start_time))
        response_template[constants.STATUS] = constants.SUCCESS
        response_template[constants.RESPONSE] = "Data generated is stored here; {}".format(
            run_id_path.replace("\\", "/"))

    except Exception as error_msg:
        my_logger.exception(error_msg)
        response_template[constants.STATUS] = constants.FAILED
        response_template[constants.RESPONSE] = error_msg
        response_template[constants.BARCODES] = "Provide the barcodes to plots function.".format(barcodes_of_interest)

    finally:
        print(response_template)
        return response_template


if __name__ == '__main__':
    generate_immune_neuronal_distance_matrix()

# Testing
# generate_immune_neuronal_distance_matrix(final_matrix_fp=r"C:\Users\{username}\Downloads\euc_dist\f",
#                                          neuronal_barcodes_identity_fp=r"C:\Users\{username}\Downloads\euc_dist\n",
#                                          file_path=r"C:\Users\{username}\Downloads\euc_dist",
#                                          barcodes_of_interest="SNAP25,CD4")  # barcodes should be separated by a ","
