# constants file
COLNAMES_VISIUM_SINGLE_NEURONS = ['Barcodes', 'X', 'Y',
                                  'Gene ID', 'Gene Name',
                                  'Text', 'Gene Expression']

# constants
GENEID = 'Gene ID'
GENENAME = 'Gene Name'
GENE_EXPRESSION = 'Gene Expression'
BARCODES = 'Barcodes'
CSV_FILE = ".csv"
PNG_FILE = ".png"
HTML_FILE = ".html"
PDF = "PDF"
PNG = "PNG"
PDF_FILE = ".pdf"
MEAN = 'mean'
MEDIAN = 'median'
MIN = 'min'
MAX = 'max'
X = 'X'
Y = 'Y'
RUNS = 'runs'
COUNTER = 0
SAMPLE = 'sample'
SEX = 'sex'
MALE = 'male'
FEMALE = 'female'
NEURONS = 'Neurons'
CELL_TYPE = 'Cell-type'
IDENT = 'ident'
CORD = 'Neur_coordinate_plot'
CORD_BAR = 'Neur_coordinate_plot_barcode'
CORD_SCATTER = 'Neur_coordinate_plot_based_on_barcode_identity'
METADATA_FILE = 'metadata.csv'

# response_template
response_template = {'Status': None,
                     'Response': None}
SUCCESS = 'Success'
FAILED = 'Failed'
STATUS = 'Status'
RESPONSE = 'Response'

MATRIX_DIR = "Dist_matrix_data"
PLOTS_DIR = "Plots_data"
PLOTS = 'Plots'
PROCESSED_FILES = "Processed_files"
FILES_NOT_MATCHING = "The files in Final_matrix and Neuronal_ident are not a match, try again later!"

IDENT_LIST = ['Aß SA LTMRs', 'Putative C-LTMRs',
              'Proprioceptors', 'TRPA1+ nociceptors',
              'Aß RA LTMRs',  'Ad HTMRs',
              'Pruritogen receptor enriched',
              'Cold nociceptors', 'Aß nociceptors',
              'Ad LTMR', 'PENK+ nociceptors',
              'Putative silent nociceptors']
FILTERED_IDENT_LIST = ['Putative silent nociceptors',  'Putative C-LTMRs', 'Pruritogen receptor enriched']

COLOR_PAL = {'male': 'cornflowerblue', 'female': 'violet'}
COLOR_LIST = ['#86baf3',  '#0b6b50', '#86baf3',  '#0b6b50',  '#86baf3',  '#86baf3',
              '#0b6b50',  '#0b6b50',  '#0b6b50',  '#86baf3', '#0b6b50',   '#0b6b50']
COLOR_LIST1 = ['#86baf3',  '#0b6b50', '#86baf3',  '#0b6b50',  '#86baf3',  '#86baf3',
               '#0b6b50',  '#0b6b50',  '#0b6b50',  '#86baf3', '#0b6b50',   '#0b6b50']

nociceps = ['Putative C-LTMRs', 'TRPA1+ nociceptors', 'Pruritogen receptor enriched',
            'Cold nociceptors', 'Aß nociceptors', 'PENK+ nociceptors', 'Putative silent nociceptors']

non_nociceps = ['Aß SA LTMRs', 'Proprioceptors', 'Aß RA LTMRs', 'Ad HTMRs', 'Ad LTMR']

final_mat = "final_matrix"
neuronal_barcodes = "neuronal_barcodes"
scaling_fac = "scaling_factor"
