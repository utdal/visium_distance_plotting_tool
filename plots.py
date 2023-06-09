#import libraries
from operator import index
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

1+1

label_size = 30
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#change path where appropriate
import pandas as pd
import numpy as np
s1r2=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\neuron_coordinates\coordinates\r2\s8r2_neurons_coord_ident.csv", index_col = 'Barcodes') #neuronal subtype identity
#s1r2=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\neuron_coordinates\s8r1_neuron_barcode_new.csv", index_col = 'Barcodes')
s1r2=s1r2.loc[:, ['ident'] ]
cd14_s1=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_distances\CD3G_distance_stats_s8r2.csv")#distance file
cd14_s1_distance = cd14_s1.loc[:,['Barcodes', 'mean', 'median', 'min', 'max']]

s1r1_cd14_final=pd.merge(s1r2,cd14_s1_distance,on=['Barcodes'],how='outer')
s1r1_cd14_final.to_csv(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_final\s8r2_CD3G_final.csv", index=False)

#repeat for the remaining samples s1r2 to s8r2



#concatenate dataframes
#change path
#female
s1r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s1r1_ITGAM_final.csv")
s3r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s3r1_ITGAM_final.csv")
#s5r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\CD19\cd19_final\s5r1_cd19_final.csv")
#s7r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\CD19\cd19_final\s7r1_cd19_final.csv")
s1r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s1r2_ITGAM_final.csv")
#s3r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s3r2_ITGAM_final.csv")
s5r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s5r2_ITGAM_final.csv")
#s7r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s7r2_ITGAM_final.csv")
data_frames_f = [s1r1_final,s3r1_final,s1r2_final,s5r2_final]
female_cd14= pd.concat(data_frames_f)
female_cd14['Neurons'] = 'Neurons'
female_cd14.to_csv(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_final\female_CD3G_final.csv", index=False)
#male
#s2r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s2r1_ITGAM_final.csv")
s4r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s4r1_ITGAM_final.csv")
#s6r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s6r1_ITGAM_final.csv")
s8r1_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s8r1_ITGAM_final.csv")
#s2r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s2r2_ITGAM_final.csv")
s4r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s4r2_ITGAM_final.csv")
#s6r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s6r2_ITGAM_final.csv")
s8r2_final=pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\ITGAM\ITGAM_final_distances\s8r2_ITGAM_final.csv")

data_frames_m = [s4r1_final,s8r1_final,s4r2_final,s8r2_final]
male_cd14= pd.concat(data_frames_m)
male_cd14['Neurons'] = 'Neurons'
male_cd14.to_csv(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_final\male_CD3G_final.csv", index=False)



#last, plot the data
#change path
d1 = pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\male_cd14_final.csv")
d2 = pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\female_cd14_final.csv")

# MIN
sns.set_context("paper")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_style("whitegrid")
sns.set_style("whitegrid")
custom_palette = ("medium grey", "mid blue")
esm1 = sns.barplot(y='min', x='Neurons', hue='ident',
            data=d1, ci=68, alpha=0.95, capsize=0.04)
esm1 = sns.stripplot(y='min', x='Neurons', hue='ident',
            data=d1, dodge=True, marker='o', edgecolor='black',
            linewidth=0.75, size=5)
esm1.set(yticks=[0,500,1000,1500,2000,2500,3000,3500,4000])
#esm1.set(yticks=[]) #CHANGE YTICKS TO BE COMPARABLE BETWEEN MALE AND FEMALE if necessary
sns.set_style("whitegrid")
sns.set(style="ticks")
esm1.set_title("Minimum MALE CD14 Round 1 and 2")
fig11 = esm1.get_figure()
fig11.savefig(r"C:\Users\mxk190051\Desktop\Interactome\cd14\plots\combined_cd14_plots\distance_plot_minimum_MALE_CD14.pdf")


# FEMALE
sns.set_context("paper")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_style("whitegrid")
sns.set_style("whitegrid")
custom_palette = ("medium grey", "mid blue")
esm11 = sns.barplot(y='min', x='Neurons', hue='ident',
            data=d2, ci=68, alpha=0.95, capsize=0.04)
esm11 = sns.stripplot(y='min', x='Neurons', hue='ident',
            data=d2, dodge=True, marker='o', edgecolor='black',
            linewidth=0.75, size=5)
esm11.set(yticks=[0,500,1000,1500,2000,2500,3000,3500,4000]) #CHANGE YTICKS TO BE COMPARABLE BETWEEN MALE AND FEMALE if necessary
sns.set_style("whitegrid")
sns.set(style="ticks")
esm11.set_title("Minimum Female CD14 Round 1 and 2")
fig11 = esm11.get_figure()
fig11.savefig(r"C:\Users\mxk190051\Desktop\Interactome\cd14\plots\combined_cd14_plots\distance_plot_minimum_FEMALE_CD14.pdf")
# MAX
#MALE
sns.set_context("paper")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_style("whitegrid")
sns.set_style("whitegrid")
custom_palette = ("medium grey", "mid blue")
esm2 = sns.barplot(y='max', x='Neurons', hue='ident',
            data=d1, ci=68, alpha=0.95, capsize=0.04)
esm2 = sns.swarmplot(y='max', x='Neurons', hue='ident',
            data=d1, dodge=True, marker='o', edgecolor='black',
            linewidth=0.75, size=5)
esm2.set(yticks=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000])
sns.set_style("whitegrid")
sns.set(style="ticks")
esm2.set_title("Maximum Male CD3G Round 1 and 2")
fig11 = esm2.get_figure()
fig11.savefig(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_combined_plots\distance_plot_MAXIMUM_MALE_CD3G.pdf")
#CHANGE YTICKS TO BE COMPARABLE BETWEEN MALE AND FEMALE if necessary

#FEMALE
sns.set_context("paper")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_style("whitegrid")
sns.set_style("whitegrid")
custom_palette = ("medium grey", "mid blue")
esm22 = sns.barplot(y='max', x='Neurons', hue='ident',
            data=d2, ci=68, alpha=0.95, capsize=0.04)
esm22 = sns.swarmplot(y='max', x='Neurons', hue='ident',
            data=d2, dodge=True, marker='o', edgecolor='black',
            linewidth=0.75, size=5)
esm22.set(yticks=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000])
sns.set_style("whitegrid")
sns.set(style="ticks")
esm22.set_title("Maximum Female CD3G Round 1 and 2")
fig11 = esm22.get_figure()

fig11.savefig(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_combined_plots\distance_plot_MAXIMUM_FEMALE_CD3G.pdf")

#MEAN
sns.set_context("paper")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_style("whitegrid")
sns.set_style("whitegrid")
custom_palette = ("medium grey", "mid blue")
esm3 = sns.barplot(y='mean', x='Neurons', hue='ident',
            data=d1, ci=68, alpha=0.95, capsize=0.04)
esm3 = sns.swarmplot(y='mean', x='Neurons', hue='ident',
            data=d1, dodge=True, marker='o', edgecolor='black',
            linewidth=0.75, size=5)
esm3.set(yticks=[0,750,1500,2250,3000,3750,4500,5250,6000,6750]) #CHANGE YTICKS TO BE COMPARABLE BETWEEN MALE AND FEMALE if necessary
sns.set_style("whitegrid")
sns.set(style="ticks")
esm3.set_title("Mean Male CD3G Round 1 and 2")
fig11 = esm3.get_figure()
fig11.savefig(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_combined_plots\distance_plot_MEAN_MALE_CD3G.pdf")




sns.set_context("paper")
sns.set(rc={'figure.figsize':(20,15)})
sns.set_style("whitegrid")
sns.set_style("whitegrid")
custom_palette = ("medium grey", "mid blue")
esm3 = sns.barplot(y='mean', x='Neurons', hue='ident',
            data=d2, ci=68, alpha=0.95, capsize=0.04)
esm3 = sns.swarmplot(y='mean', x='Neurons', hue='ident',
            data=d2, dodge=True, marker='o', edgecolor='black',
            linewidth=0.75, size=5)
 #CHANGE YTICKS TO BE COMPARABLE BETWEEN MALE AND FEMALE if necessary
sns.set_style("whitegrid")
sns.set(style="ticks")
esm3.set(yticks=[0,750,1500,2250,3000,3750,4500,5250,6000,6750])
esm3.set_title("Mean Female CD3G Round 1 and 2")
fig11 = esm3.get_figure()
fig11.savefig(r"C:\Users\mxk190051\Desktop\Interactome\CD3G\CD3G_combined_plots\distance_plot_MEAN_FEMALE_ITGAM.pdf")




# combine r1 and r2

r1_female = pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\female_r1_cd14_final.csv")
r2_female = pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\female_cd14_r2_final.csv")
#female
data_frames_f = [r1_female,r2_female]
female_cd14= pd.concat(data_frames_f)
female_cd14['Neurons'] = 'Neurons'
female_cd14.to_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\female_cd14_final.csv", index=False)
# male

r1_male = pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\male_r1_cd14_final.csv")
r2_male = pd.read_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\male_cd14_r2_final.csv")
data_frames_f = [r1_male,r2_male]
male_cd14= pd.concat(data_frames_f)
male_cd14['Neurons'] = 'Neurons'
male_cd14.to_csv(r"C:\Users\mxk190051\Desktop\Interactome\cd14\final\male_cd14_final.csv", index=False)