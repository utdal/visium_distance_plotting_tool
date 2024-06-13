# Imports
from logger import my_logger
import plotly.graph_objects as go
import plotly.io as pio, warnings, numpy as np
import plotly.express as px, seaborn as sns, click
import constants as constants, os, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_immune_neur_distances(filepath: str):  # final_matrix_fp: str, neuronal_barcodes_identity_fp: str, barcodes_of_interest: list)
    """
    :param filepath:
    Example: C:/Users/NXI220005/Desktop/Visium_analysis/Processed_files/runs/5BFG3wbBgd
    :return:
    """
    try:
        print('\n')
        print('Triggering the function to generate plots...\n')
        my_logger.info("<---- Plotting immune neuronal distance charts function initiated ---->")
        response_template = constants.response_template.copy()
        male_file_list, female_file_list = list(), list()
        plots_data_dir = filepath + os.sep + constants.PLOTS_DIR
        plots_filepath = filepath + os.sep + constants.PLOTS
        metadata_df = pd.read_csv(constants.METADATA_FILE)
        resource_dict = {i.split('.')[0]: j for i, j in zip(metadata_df[constants.final_mat], metadata_df["gender"])}
        male_list = [i for i, j in resource_dict.items() if j == constants.MALE]
        female_list = [i for i, j in resource_dict.items() if j == constants.FEMALE]

        if not os.path.exists(plots_filepath):
            os.makedirs(plots_filepath)

        for barcode_dir in os.listdir(os.path.join(filepath, constants.PLOTS_DIR)):
            print("Barcode {}".format(barcode_dir))
            my_logger.info("Running for barcode {}".format(barcode_dir))
            barcode_dir_i = os.path.join(plots_data_dir, barcode_dir)
            for file in sorted(os.listdir(barcode_dir_i)):
                if any(string in file for string in male_list) and 'distance' in file:
                    male_file_list.append(file)
                if any(string in file for string in female_list) and 'distance' in file:
                    female_file_list.append(file)
            male_file_list, female_file_list = list(set(male_file_list)), list(set(female_file_list))

            if not os.path.exists(os.path.join(filepath, constants.PLOTS, str(barcode_dir))):
                os.makedirs(os.path.join(filepath, constants.PLOTS, str(barcode_dir)))

            # Plotting heatmaps
            #####################################################
            for male in male_file_list:
                print("Running for -- {} -- MALE".format(male))
                my_logger.info("Running for -- {} -- MALE".format(male))
                temp_df_male = pd.read_csv(plots_data_dir + os.sep + barcode_dir + os.sep + male)
                temp_df_male.set_index(constants.BARCODES, inplace=True)
                m = px.imshow(temp_df_male, aspect=True)
                m.update_layout(title="Distance Matrix Plot for {}".format(male.split('.')[0]))
                temp_plot_name = plots_filepath + os.sep + str(barcode_dir) + os.sep + male.split('.')[0] + '_{}_male_heatmap'.format(
                    barcode_dir) + constants.HTML_FILE
                m.write_html(temp_plot_name)

            for female in female_file_list:
                print("Running for -- {} -- FEMALE".format(female))
                my_logger.info("Running for -- {} -- FEMALE".format(female))
                temp_df_female = pd.read_csv(plots_data_dir + os.sep + barcode_dir + os.sep + female)
                temp_df_female.set_index(constants.BARCODES, inplace=True)
                f = px.imshow(temp_df_female, aspect=True)
                f.update_layout(title="Distance Matrix Plot for {}".format(female.split('.')[0]))
                temp_plot_name = plots_filepath + os.sep + str(barcode_dir) + os.sep + female.split('.')[0] + '_{}_female_heatmap'.format(
                    barcode_dir) + constants.HTML_FILE
                f.write_html(temp_plot_name)

            # Plotting bar graph
            #####################################################
            male_concat_list, female_concat_list = list(), list()
            male_stat_file_list, female_stat_file_list = list(), list()
            for file in sorted(os.listdir(barcode_dir_i)):
                if any(string in file for string in male_list) and 'stat' in file:
                    male_stat_file_list.append(file)
                if any(string in file for string in female_list) and 'stat' in file:
                    female_stat_file_list.append(file)
            male_stat_file_list, female_stat_file_list = list(set(male_stat_file_list)), list(set(female_stat_file_list))

            for male_file in sorted(male_stat_file_list):
                temp_df_male = pd.read_csv(plots_data_dir + os.sep + barcode_dir + os.sep + male_file)
                temp_df_male[constants.SEX] = constants.MALE
                temp_df_male.set_index(constants.BARCODES, inplace=True)
                male_concat_list.append(temp_df_male)
            for female_file in sorted(female_stat_file_list):
                temp_df_female = pd.read_csv(plots_data_dir + os.sep + barcode_dir + os.sep + female_file)
                temp_df_female[constants.SEX] = constants.FEMALE
                temp_df_female.set_index(constants.BARCODES, inplace=True)
                female_concat_list.append(temp_df_female)
            df_concat = pd.concat([pd.concat(male_concat_list), pd.concat(female_concat_list)])
            hist_fig = px.histogram(df_concat.loc[:, [constants.MEAN, constants.MEDIAN, constants.MIN, constants.MAX,
                                                      constants.SEX]], color=constants.SEX, marginal="rug", nbins=100,
                                    color_discrete_sequence=['bisque', 'burlywood'])
            hist_fig.update_layout(plot_bgcolor="white")
            hist_fig.update_xaxes(gridcolor='black', griddash='dash', minor_griddash="dot")
            hist_fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title_text='Male and Female Distribution')
            hist_fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, title_text='Frequency')
            hist_fig.update_traces(marker_line_color='black', marker_line_width=0.25)
            temp_plot_name = plots_filepath + os.sep + str(barcode_dir) + os.sep + '{}_male_and_female_histogram'.format(
                barcode_dir) + constants.HTML_FILE
            pio.write_html(hist_fig, file=temp_plot_name, auto_open=False)

            # Plotting Bar-Strip Plots
            #####################################################
            label_size = 30
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size
            mpl.rcParams["font.sans-serif"] = ["Arial"]
            mpl.rcParams["font.family"] = "Arial"
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42

            sns.set_context("paper")
            sns.set(rc={'figure.figsize': (20, 15)})
            sns.set_style("whitegrid")
            sns.set(style="ticks")
            # ####################################################
            for tendency in [constants.MEAN, constants.MEDIAN,
                             constants.MIN, constants.MAX]:
                sns_fig = sns.barplot(y=tendency, x=constants.CELL_TYPE, hue=constants.IDENT,
                                      data=df_concat, alpha=0.95, capsize=0.04)
                sns_fig = sns.stripplot(y=tendency, x=constants.CELL_TYPE, hue=constants.IDENT,
                                        data=df_concat, dodge=True, marker='o', edgecolor='black',
                                        linewidth=0.75, size=5)
                plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
                sns_fig.set_xlabel('Cell-type')
                handles, labels = sns_fig.get_legend_handles_labels()
                # the following needs to be changed if need be
                sns_fig.legend(handles[len(handles)//2:], labels[len(labels)//2:], loc='upper right')
                # sns_fig.set(yticks=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
                sns_fig.set_title("{} -- {}: Round 1 and 2".format(tendency.upper(), barcode_dir))
                fig_bar_strip = sns_fig.get_figure()
                fig_bar_strip.savefig(plots_filepath + os.sep + str(barcode_dir) + os.sep + '{}_{}_strip_plot'.format(
                    tendency, barcode_dir) + constants.PDF_FILE)
                plt.clf()  # empty the plot
            # ####################################################
            for tendency in [constants.MEAN, constants.MEDIAN,
                             constants.MIN, constants.MAX]:
                sns_fig = sns.barplot(y=tendency, x=constants.SEX, hue=constants.IDENT,
                                      data=df_concat, alpha=0.95, capsize=0.04)
                sns_fig = sns.stripplot(y=tendency, x=constants.SEX, hue=constants.IDENT,
                                        data=df_concat, dodge=True, marker='o', edgecolor='black',
                                        linewidth=0.75, size=5)
                plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
                sns_fig.set_xlabel('Cell-type based on gender/sex')
                handles, labels = sns_fig.get_legend_handles_labels()
                sns_fig.legend(handles[len(handles)//2:], labels[len(labels)//2:], loc='upper right')
                sns_fig.set_title("{} -- {}: Round 1 and 2".format(tendency.upper(), barcode_dir))
                fig_bar_strip = sns_fig.get_figure()
                fig_bar_strip.savefig(plots_filepath + os.sep + str(barcode_dir) + os.sep + '{}_{}_sex_filter_strip_plot'.format(
                    tendency, barcode_dir) + constants.PDF_FILE)
                plt.clf()  # empty the plot
            # ####################################################
            for tendency in [constants.MEAN, constants.MEDIAN,
                             constants.MIN, constants.MAX]:
                fig_bar_strip = sns.barplot(y=tendency, x=constants.IDENT, data=df_concat, palette=constants.COLOR_PAL,
                                            hue=constants.SEX, alpha=0.95, capsize=0.04)
                fig_bar_strip.set_xlabel('Cell-type based on gender/sex')
                plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
                plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=15)
                fig_bar_strip.set_title("{} -- {}: Round 1 and 2".format(tendency.upper(), barcode_dir))
                fig_bar_strip = sns_fig.get_figure()
                plt.legend(loc='upper right', title=constants.SEX.capitalize())
                fig_bar_strip.savefig(
                    plots_filepath + os.sep + str(barcode_dir) + os.sep + '{}_{}_bar_plot'.format(
                        tendency, barcode_dir) + constants.PDF_FILE)
                plt.clf()  # empty the plot

            '''


            Additional code can be added here, where gender based plots or other plots can be generated,
            iterating on the barcode ==> look up line (33) / generate_plots.py


            '''

        response_template[constants.STATUS] = constants.SUCCESS
        response_template[constants.RESPONSE] = "Plots generated are saved here {}.".format(plots_filepath)
        my_logger.info("<---- Plotting immune neuronal distance charts function completed ---->")

    except Exception as error_msg:
        my_logger.exception(error_msg)
        response_template[constants.STATUS] = constants.FAILED
        response_template[constants.RESPONSE] = error_msg

    finally:
        print(response_template)
        return response_template


def generate_cord_plot(dataframe, fig_path):
    try:
        plt.clf()
        my_logger.info("<---- Generating cord plot initiated ---->")
        my_logger.info("<---- Cord plot saved to {} ---->".format(fig_path))
        heatmap, xedges, yedges = np.histogram2d(dataframe[constants.X],
                                                 dataframe[constants.Y], bins=50)
        p = plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1],
                                      yedges[0], yedges[-1]], origin='lower', aspect='auto')
        cb = plt.colorbar()
        cb.set_label('Binned counts')
        plt.xlabel("Bins of X Co-ordinates")
        plt.ylabel("Bins of Y Co-ordinates")
        plt.title("Binned Neuronal Co-ordinate Plot")
        plt.savefig(fig_path)
        my_logger.info("<---- Generating cord plot completed ---->")

    except Exception as error_msg:
        my_logger.exception(error_msg)
        return error_msg


def generate_scatter_plot_for_final_neuronal_barcodes(final_matrix_df, updated_final_matrix_df, neurons_df,
                                                       final_matrix_file, neuronal_identity_file, barcode,
                                                       plots_filepath, neuronal_barcode_df):
    barcode_dir = plots_filepath + os.sep + str(barcode)
    if not os.path.exists(barcode_dir):
        os.makedirs(barcode_dir)
    print(
        "Generating neuron co-ordinate plot for a barcode: {}, neuronal data: {} and final matrix: {}".format(
            barcode, neuronal_identity_file, final_matrix_file))
    my_logger.info(
        "Generating neuron co-ordinate plot for a barcode: {}, neuronal data: {} and final matrix: {}".format(
            barcode, neuronal_identity_file, final_matrix_file))

    COLOR_DICT = {i: j for i, j in zip(constants.IDENT_LIST, constants.COLOR_LIST)}
    final_matrix_df_filtered_barcode, final_matrix_df_filtered_neuronal_barcode = updated_final_matrix_df, neurons_df

    final_matrix_df = final_matrix_df.loc[:, [constants.BARCODES, constants.X, constants.Y, constants.GENE_EXPRESSION]].drop_duplicates()
    final_matrix_df_filtered_barcode = final_matrix_df_filtered_barcode.loc[:, [constants.BARCODES, constants.X, constants.Y, constants.GENE_EXPRESSION]].drop_duplicates()
    final_matrix_df_filtered_neuronal_barcode = final_matrix_df_filtered_neuronal_barcode.loc[:, [constants.BARCODES, constants.X, constants.Y, constants.GENE_EXPRESSION]].drop_duplicates()

    barcode_to_Ident = dict(zip(neuronal_barcode_df[constants.BARCODES], neuronal_barcode_df[constants.IDENT]))

    final_matrix_df_filtered_neuronal_barcode[constants.IDENT] = final_matrix_df_filtered_neuronal_barcode[constants.BARCODES].map(barcode_to_Ident)
    final_matrix_df_filtered_neuronal_barcode_limited = final_matrix_df_filtered_neuronal_barcode.loc[final_matrix_df_filtered_neuronal_barcode[constants.IDENT].isin(constants.FILTERED_IDENT_LIST)]
    final_matrix_nocicep_df = final_matrix_df_filtered_neuronal_barcode[final_matrix_df_filtered_neuronal_barcode[constants.IDENT].isin(constants.nociceps)]
    final_matrix_non_nocicep_df = final_matrix_df_filtered_neuronal_barcode[final_matrix_df_filtered_neuronal_barcode[constants.IDENT].isin(constants.non_nociceps)]

    fig, fig1, fig2 = go.Figure(), go.Figure(), go.Figure()
    fig.add_trace(go.Scatter(x=final_matrix_df[constants.X], y=final_matrix_df[constants.Y],
                             mode='markers', name='Tissue', marker=dict(size=13.5, color='#DCDCDC', opacity=0.3),
                             text=final_matrix_df[['Barcodes']], hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig.add_trace(go.Scatter(x=final_matrix_df_filtered_neuronal_barcode[constants.X], y=final_matrix_df_filtered_neuronal_barcode[constants.Y],
                             mode='markers', name='Neuronal Barcode', marker=dict(color='#7f8de1', size=14, opacity=0.5),
                             text=final_matrix_df_filtered_neuronal_barcode_limited[['Barcodes']], hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig.add_trace(go.Scatter(x=final_matrix_df_filtered_barcode[constants.X], y=final_matrix_df_filtered_barcode[constants.Y],
                             mode='markers', name='Barcode of Interest: {}'.format(barcode), marker=dict(size=14, color='#ff7f50', opacity=0.8),
                             text=final_matrix_df_filtered_barcode[['Gene Expression']], hovertemplate='X: %{x}<br>Y: %{y}<br>Gene Expression: %{text}<extra></extra>'))
    fig.update_layout(title='Scatter Plot of Final Matrix: {}, Neuronal Barcodes and {} markers'.format(final_matrix_file.split('.')[0], barcode),
                      xaxis_title='X co-ordinate in final matrix', yaxis_title='Y co-ordinate in final matrix',
                      legend_title='Neuron type', plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridcolor='lightgray', mirror=True, tickformat='.0f'),
                      yaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridcolor='lightgray', mirror=True, tickformat='.0f'))
    fig.write_html(os.path.join(plots_filepath, str(barcode), "{}_".format(constants.CORD_SCATTER)+str(final_matrix_file.split('.')[0])+constants.HTML_FILE))
    fig.write_image(os.path.join(plots_filepath, str(barcode), "{}_".format(constants.CORD_SCATTER)+str(final_matrix_file.split('.')[0])+constants.PDF_FILE), width=1500, height=950)

    # COLOR: final_matrix_df_filtered_neuronal_barcode[constants.IDENT] color = px.colors.qualitative.G10[5]
    final_matrix_df_filtered_neuronal_barcode[constants.IDENT] = final_matrix_df_filtered_neuronal_barcode[constants.IDENT].map(COLOR_DICT)

    fig1.add_trace(go.Scatter(x=final_matrix_df[constants.X], y=final_matrix_df[constants.Y],
                              mode='markers', name='Tissue', marker=dict(size=13.5, color='#DCDCDC', opacity=0.3),
                              text=final_matrix_df[['Barcodes']], hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=final_matrix_nocicep_df[constants.X], y=final_matrix_nocicep_df[constants.Y],
                              mode='markers', name='Nociceptor Neuronal Barcodes', marker=dict(color="#b9f0c5", size=14, opacity=0.5),
                              text=final_matrix_df_filtered_neuronal_barcode_limited[['Barcodes']], hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=final_matrix_non_nocicep_df[constants.X], y=final_matrix_non_nocicep_df[constants.Y],
                              mode='markers', name='Non-nociceptor Neuronal Barcodes', marker=dict(color="#a8b6ed", size=14, opacity=0.5),
                              text=final_matrix_df_filtered_neuronal_barcode_limited[['Barcodes']], hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=final_matrix_df_filtered_barcode[constants.X], y=final_matrix_df_filtered_barcode[constants.Y],
                              mode='markers', name='Barcode of Interest: {}'.format(barcode), marker=dict(size=14, color='#ff7f50', opacity=0.8),
                              text=final_matrix_df_filtered_barcode[['Gene Expression']], hovertemplate='X: %{x}<br>Y: %{y}<br>Gene Expression: %{text}<extra></extra>'))
    fig1.update_layout(title='Scatter Plot of Final Matrix: {}, Neuronal Barcodes and {} markers'.format(final_matrix_file.split('.')[0], barcode),
                       xaxis_title='X co-ordinate in final matrix', yaxis_title='Y co-ordinate in final matrix',
                       legend_title='Neuron type', plot_bgcolor='white', paper_bgcolor='white',
                       xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridcolor='lightgray', mirror=True, tickformat='.0f'),
                       yaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridcolor='lightgray', mirror=True, tickformat='.0f'))
    fig1.write_html(os.path.join(plots_filepath, str(barcode), "{}_".format(constants.CORD_SCATTER)+"and_neur_indentity_"+str(final_matrix_file.split('.')[0])+constants.HTML_FILE))
    fig1.write_image(os.path.join(plots_filepath, str(barcode), "{}_".format(constants.CORD_SCATTER)+"and_neur_indentity_"+str(final_matrix_file.split('.')[0])+constants.PDF_FILE), width=1500, height=950)

    fig2.add_trace(go.Scatter(x=final_matrix_df[constants.X], y=final_matrix_df[constants.Y],
                              mode='markers', name='Tissue', marker=dict(size=13.5, color='#DCDCDC', opacity=0.3),
                              text=final_matrix_df[['Barcodes']], hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=final_matrix_df_filtered_neuronal_barcode_limited[constants.X],
                              y=final_matrix_df_filtered_neuronal_barcode_limited[constants.Y],
                              mode='markers', name='Neuronal Barcode', marker=dict(color='#7f8de1', size=14, opacity=0.5),
                              text=final_matrix_df_filtered_neuronal_barcode_limited[['Barcodes']],
                              hovertemplate='X: %{x:.f}<br>Y: %{y:.f}<br>Barcode: %{text}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=final_matrix_df_filtered_barcode[constants.X], y=final_matrix_df_filtered_barcode[constants.Y],
                              mode='markers', name='Barcode of Interest: {}'.format(barcode),
                              marker=dict(size=final_matrix_df_filtered_barcode[constants.GENE_EXPRESSION],
                                          sizemode='diameter', color='#f59145', sizeref=0.1),
                              text=final_matrix_df_filtered_barcode[['Gene Expression']], hovertemplate='X: %{x}<br>Y: %{y}<br>Gene Expression: %{text}<extra></extra>'))
    fig2.update_layout(title='Scatter Plot of Final Matrix: {}, Neuronal Barcodes and {} markers for Putative and Pruritogen receptors'.format(final_matrix_file.split('.')[0], barcode),
                       xaxis_title='X co-ordinate in final matrix', yaxis_title='Y co-ordinate in final matrix',
                       legend_title='Neuron type', plot_bgcolor='white', paper_bgcolor='white',
                       xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridcolor='lightgray', mirror=True, tickformat='.0f'),
                       yaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridcolor='lightgray', mirror=True, tickformat='.0f'))
    fig2.write_html(os.path.join(plots_filepath, str(barcode), "{}_".format(constants.CORD_SCATTER)+"filtered_"+str(final_matrix_file.split('.')[0])+constants.HTML_FILE))
    fig2.write_image(os.path.join(plots_filepath, str(barcode), "{}_".format(constants.CORD_SCATTER)+"filtered_"+str(final_matrix_file.split('.')[0])+constants.PDF_FILE), width=1500, height=950)

# plot_immune_neur_distances(r"C:\Users\{username}\Desktop\Visium_spatial_analysis_eucl_distances\Processed_files\runs\UoBUBH1arl",
#                            r"C:\Users\{username}\Desktop\Visium_spatial_analysis_eucl_distances\final_matrix_space_ranger",
#                            r"C:\Users\{username}\Desktop\Visium_spatial_analysis_eucl_distances\neurons_ident", "SNAP25,OSM")
