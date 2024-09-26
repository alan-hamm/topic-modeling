
#%%

from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ParameterGrid


#%%
import pyLDAvis.gensim  # Library for interactive topic model visualization
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpld3 import save_html


from tqdm import tqdm  # Creates progress bars to visualize the progress of loops or tasks
from gensim.models import LdaModel  # Implements LDA for topic modeling using the Gensim library
from gensim.corpora import Dictionary  # Represents a collection of text documents as a bag-of-words corpus
from gensim.models import CoherenceModel  # Computes coherence scores for topic models
import pyLDAvis
import IProgress 
import os  # Provides functions for interacting with the operating system, such as creating directories
import itertools  # Provides various functions for efficient iteration and combination of elements
import numpy as np  # Library for numerical computing in Python, used for array operations and calculations
from time import time, sleep # Measures the execution time of code snippets or functions
import pprint as pp  # Pretty-printing library, used here to format output in a readable way
import pandas as pd
import logging # Logging module for generating log messages
import sys # Provides access to some variables used or maintained by the interpreter and to functions that interact with the interpreter 
import shutil # High-level file operations such as copying and removal 
import zipfile # Provides tools to create, read, write, append, and list a ZIP file
from tqdm.notebook import tqdm  # Creates progress bars in Jupyter Notebook environment
from json import load
import random
import logging
import csv
import pprint as pp
from pandas.api.types import CategoricalDtype
from typing import Union, List
import math
from scipy import stats

from dask.distributed import as_completed
import dask   # Parallel computing library that scales Python workflows across multiple cores or machines 
from dask.distributed import Client, LocalCluster, wait   # Distributed computing framework that extends Dask functionality 
from dask.diagnostics import ProgressBar   # Visualizes progress of Dask computations
from dask.distributed import progress
from distributed import Future
from dask.delayed import Delayed # Decorator for creating delayed objects in Dask computations
#from dask.distributed import as_completed
from dask.bag import Bag
from dask import delayed
import dask.config
#from dask.distributed import wait
from dask.distributed import performance_report, wait, as_completed #,print
from distributed import get_worker
import gc
import hashlib
import pickle



#%%


import logging
from datetime import datetime

DECADE_TO_PROCESS ='2010s'
LOG_DIRECTORY = f"C:/_harvester/data/lda-models/{DECADE_TO_PROCESS}/log/"
# Ensure the LOG_DIRECTORY exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Get the current date and time
now = datetime.now()

# Format the date and time as per your requirement
# Note: %w is the day of the week as a decimal (0=Sunday, 6=Saturday)
#       %Y is the four-digit year
#       %m is the two-digit month (01-12)
#       %H%M is the hour (00-23) followed by minute (00-59) in 24hr format
log_filename = now.strftime('log-%w-%m-%Y-%H%M.log')
LOGFILE = os.path.join(LOG_DIRECTORY,log_filename)

# Configure logging to write to a file with this name
logging.basicConfig(
    filename=LOGFILE,
    filemode='a',  # Append mode if you want to keep adding to the same file during the day
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


#%%

# Dask dashboard throws deprecation warnings w.r.t. Bokeh
import warnings
from bokeh.util.deprecation import BokehDeprecationWarning
from numpy import ComplexWarning

# Suppress ComplexWarnings
# generated in create_vis() function with js_PCoA use of matplotlib
warnings.simplefilter('ignore', ComplexWarning)

# Disable Bokeh deprecation warnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
# Filter out the specific warning message
# Set the logging level for distributed.utils_perf to suppress warnings
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="distributed.utils_perf")


#%%



# Define the range of number of topics for LDA and step size
START_TOPICS = 20
END_TOPICS = 120
STEP_SIZE = 5

# define the decade that is being modelled 
DECADE = DECADE_TO_PROCESS

# In the case of this machine, since it has an Intel Core i9 processor with 8 physical cores (16 threads with Hyper-Threading), 
# it would be appropriate to set the number of workers in Dask Distributed LocalCluster to 8 or slightly lower to allow some CPU 
# resources for other tasks running on your system.
CORES = 8
MAXIMUM_CORES = 12

THREADS_PER_CORE = 8

RAM_MEMORY_LIMIT = "16GB" 

CPU_UTILIZATION_THRESHOLD = 125 # eg 85%
MEMORY_UTILIZATION_THRESHOLD = 12 * (1024 ** 3)  # Convert GB to bytes

# Specify the local directory path, spilling will be written here
DASK_DIR = '/_harvester/tmp-dask-out'

# specify the number of passes for Gensim LdaModel
PASSES = 15

# specify the number of iterations
ITERATIONS = 50

# Number of documents to be iterated through for each update. 
# Set to 0 for batch learning, > 1 for online iterative learning.
UPDATE_EVERY = 5

# Log perplexity is estimated every that many updates. 
# Setting this to one slows down training by ~2x.
EVAL_EVERY = 10

RANDOM_STATE = 75

PER_WORD_TOPICS = True

# number of documents to extract from the JSON source file when testing and developing
NUM_DOCUMENTS = 25

# the number of documents to read from the JSON source file per batch
FUTURES_BATCH_SIZE = 100

# Constants for adaptive batching and retries
# Number of futures to process per iteration
BATCH_SIZE = 75 # number of documents
MAX_BATCH_SIZE = 150 
INCREASE_FACTOR = 1.05  # Increase batch size by p% upon success
DECREASE_FACTOR = .10 # Decrease batch size by p% upon failure or timeout
MAX_RETRIES = 5        # Maximum number of retries per task
BASE_WAIT_TIME = 30     # Base wait time in seconds for exponential backoff


# Load data from the JSON file
DATA_SOURCE = "C:/_harvester/data/tokenized-sentences/10s/2010-2014_min_six_word-w-bigrams.json"
TRAIN_RATIO = .80

TIMEOUT = None #"90 minutes"

EXTENDED_TIMEOUT = None #"120 minutes"

# Enable serialization optimizations
dask.config.set(scheduler='distributed', serialize=True)
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.scheduler.worker-ttl": None})
#dask.config.set({"distributed.scheduler.worker-ttl": None})


#%%


import gc
def garbage_collection(development: bool, location: str):
    if development:
        # Enable debugging flags for leak statistics
        gc.set_debug(gc.DEBUG_LEAK)

    # Before calling collect, get a count of existing objects
    before = len(gc.get_objects())

    # Perform garbage collection
    collected = gc.collect()

    # After calling collect, get a new count of existing objects
    after = len(gc.get_objects())

    # Print or log before and after counts along with number collected
    logging.info(f"Garbage Collection at {location}:")
    logging.info(f"  Before GC: {before} objects")
    logging.info(f"  After GC: {after} objects")
    logging.info(f"  Collected: {collected} objects\n")



#%%


import os
import pandas as pd
import zipfile

# Define the top-level directory and subdirectories
DECADE = "2010s"  # Replace with your actual decade value
ROOT_DIR = f"C:/_harvester/data/lda-models/{DECADE}_html"
LOG_DIR = os.path.join(ROOT_DIR, "log")
IMAGE_DIR = os.path.join(ROOT_DIR, "visuals")
PYLDA_DIR = os.path.join(IMAGE_DIR, 'pyLDAvis')
PCOA_DIR = os.path.join(IMAGE_DIR, 'PCoA')
METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
TEXTS_ZIP_DIR = os.path.join(ROOT_DIR, "texts_zip")

# Ensure that all necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PYLDA_DIR, exist_ok=True)
os.makedirs(PCOA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TEXTS_ZIP_DIR, exist_ok=True)

# Function to save text data to a zip file and return the path
def save_text_to_zip(text_data):
    # Generate a unique filename based on current timestamp
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
    text_zip_filename = f"{timestamp_str}.zip"
    
    # Write the text content to a zip file within TEXTS_ZIP_DIR
    zip_path = os.path.join(TEXTS_ZIP_DIR, text_zip_filename)
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("text.txt", text_data)
    
    return zip_path

# Function to save text data and model to single ZIP file
def save_to_zip(time, text_data, ldamodel):
    # Generate a unique filename based on current timestamp
    timestamp_str = hashlib.md5(time.strftime('%Y%m%d%H%M%S%f').encode()).hexdigest()
    text_zip_filename = f"{timestamp_str}.zip"
    
    # Write the text content and model to a zip file within TEXTS_ZIP_DIR
    zip_path = os.path.join(TEXTS_ZIP_DIR, text_zip_filename)
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"doc_{text_zip_filename}.txt", text_data)
        ldamodel_bytes = pickle.dumps(ldamodel)
        zf.writestr(f"model_{text_zip_filename}.pkl", ldamodel_bytes)
    
    return zip_path

# method to deserialize and return the LDA model object
def load_pkl_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        pkl_files = [file for file in zf.namelist() if file.endswith('.pkl')]
        if len(pkl_files) == 0:
            raise ValueError("No pkl files found in the ZIP archive.")
        
        pkl_file = pkl_files[0]
        pkl_bytes = zf.read(pkl_file)
        loaded_pkl = pickle.loads(pkl_bytes)
    
    return loaded_pkl

# Function to add new model data to metadata Parquet file
def add_model_data_to_metadata(model_data, batchsize):
    #print("we are in the add_model_data_to_metadata method()")
    # Save large body of text to zip and update model_data reference
    texts_zipped = []
    
    #for text_list in model_data['text']:
    for text_list in model_data['text']:
        combined_text = ' '.join([''.join(sent) for sent in text_list])  # Combine all sentences into one string
        zip_path = save_to_zip(model_data['time'], combined_text, model_data['lda_model'])
        texts_zipped.append(zip_path)
    # Update model data with zipped paths
    model_data['text'] = texts_zipped
     # Ensure other fields are not lists, or if they are, they should have only one element per model
    for key, value in model_data.items():
        #if isinstance(value, list) and key != 'text':
        if isinstance(value, list) and key not in ['text', 'alpha', 'beta', 'topics', 'model_scores']:
            assert len(value) == 1, f"Field {key} has multiple elements"
            model_data[key] = value[0]  # Unwrap single-element list

    # Define the expected data types for each column
    expected_dtypes = {
        'type': str,
        'batch_size': int,
        'text': object,  # Use object dtype for lists of strings (file paths)
        'text_sha256': str,
        'text_md5': str,
        'corpus': object,
        'dictionary': object,
        #'perplexity': 'float32',
        #'coherence': 'float32',
        'topics': int,
        'alpha': list,
        'beta': list,
        'max_iter': int,
        'eval_every': int,
        'chunksize': int,
        'random_state': int,
        'top_words': object,
        'lda_model': object,
        'best_lda_model': object,
        'model_scores': list,
        'cv_results': dict,
        'best_params': dict,
        'log_likelihood': 'float32',
        'n_splits': int,
        'refit_time_': 'float32',
        'time': 'datetime64[ns]',
    }   

    
    try:
        #df_new_metadata = pd.DataFrame({key: [value] if not isinstance(value, list) else value 
        #                                for key, value in model_data.items()}).astype(expected_dtypes)
        # Create a new DataFrame without enforcing dtypes initially
        df_new_metadata = pd.DataFrame({key: [value] if not isinstance(value, list) else value 
                                        for key, value in model_data.items()})
        
        # Apply type conversion selectively
        #for col_name in ['perplexity', 'coherence']:
        #    df_new_metadata[col_name] = df_new_metadata[col_name].astype('float32')
            
        df_new_metadata['topics'] = df_new_metadata['topics'].astype(int)
        #df_new_metadata['time'] = pd.to_datetime(df_new_metadata['time'])
        df_new_metadata['batch_size'] = batchsize
        #df_new_metadata['create_pylda'] = pylda_success
        #df_new_metadata['create_pcoa'] = pcoa_success
        # drop lda model from dataframe
        df_new_metadata = df_new_metadata.drop('dictionary', axis=1)
        df_new_metadata = df_new_metadata.drop('corpus', axis=1)
        df_new_metadata = df_new_metadata.drop('lda_model', axis=1)
        df_new_metadata = df_new_metadata.drop('best_lda_model', axis=1)
    except ValueError as e:
        # Initialize an error message list
        error_messages = [f"Error converting model_data to DataFrame with enforced dtypes: {e}"]
        
        
        # Iterate over each item in model_data to collect its key, expected dtype, and actual value
        for key, value in model_data.items():
            expected_dtype = expected_dtypes.get(key, 'No expected dtype specified')
            actual_dtype = type(value).__name__
            error_messages.append(f"Column: {key}, Expected dtype: {expected_dtype}, Actual dtype: {actual_dtype}, Value: {value}")
        
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)

        logging.error(full_error_message)

        raise ValueError("Data type mismatch encountered during DataFrame conversion. Detailed log available.")

    # Path to the metadata Parquet file
    parquet_file_path = os.path.join(METADATA_DIR, "metadata.parquet")

    # Check if the Parquet file already exists
    if os.path.exists(parquet_file_path): 
        # If it exists, read the existing metadata and append the new data 
        df_metadata = pd.read_parquet(parquet_file_path) 
        df_metadata = pd.concat([df_metadata, df_new_metadata], ignore_index=True) 
    else: 
        # If it doesn't exist, use the new data as the starting point 
        df_metadata = df_new_metadata


    # Save updated metadata DataFrame back to Parquet file
    df_metadata.to_parquet(parquet_file_path)
    del df_metadata, df_new_metadata, model_data
    #garbage_collection(True, 'add_model_data_to_metadata(...)')
    #print("\nthis is the value of the parquet file")
    #print(df_metadata)


# Function to read a specific text from its zip file based on metadata query
def get_text_from_zip(zip_path): 
    with zipfile.ZipFile(zip_path, 'r') as zf: 
        return zf.read('text.txt').decode('utf-8')

# Example usage: Load metadata and retrieve texts based on some criteria
def load_texts_for_analysis(metadata_path, coherence_threshold=0.7): 
    # Load the metadata into a DataFrame 
    df_metadata = pd.read_parquet(metadata_path)

    # Filter metadata based on some criteria (e.g., coherence > threshold)
    filtered_metadata = df_metadata[df_metadata['coherence'] > coherence_threshold]

    # Retrieve and decompress associated texts from their zip files
    texts = [get_text_from_zip(zip_path) for zip_path in filtered_metadata['text']]

    return texts


#%%



num_topics = len(range(START_TOPICS, END_TOPICS + 1, STEP_SIZE))

# Calculate numeric_alpha for symmetric prior
numeric_symmetric = 1.0 / num_topics
# Calculate numeric_alpha for asymmetric prior (using best judgment)
numeric_asymmetric = 1.0 / (num_topics + np.sqrt(num_topics))
# Create the list with numeric values
numeric_alpha = [numeric_symmetric, numeric_asymmetric] + np.arange(0.01, 1, 0.3).tolist()
numeric_beta = [numeric_symmetric] + np.arange(0.01, 1, 0.3).tolist()


# The parameter `alpha` in Latent Dirichlet Allocation (LDA) represents the concentration parameter of the Dirichlet 
# prior distribution for the topic-document distribution.
# It controls the sparsity of the resulting document-topic distributions.

# A lower value of `alpha` leads to sparser distributions, meaning that each document is likely to be associated with fewer topics.
# Conversely, a higher value of `alpha` encourages documents to be associated with more topics, resulting in denser distributions.

# The choice of `alpha` affects the balance between topic diversity and document specificity in LDA modeling.
alpha_values = ['symmetric', 'asymmetric']
alpha_values += np.arange(0.01, 1, 0.3).tolist()

# In Latent Dirichlet Allocation (LDA) topic analysis, the beta parameter represents the concentration 
# parameter of the Dirichlet distribution used to model the topic-word distribution. It controls the 
# sparsity of topics by influencing how likely a given word is to be assigned to a particular topic.

# A higher value of beta encourages topics to have a more uniform distribution over words, resulting in more 
# general and diverse topics. Conversely, a lower value of beta promotes sparser topics with fewer dominant words.

# The choice of beta can impact the interpretability and granularity of the discovered topics in LDA.
beta_values = ['symmetric']
beta_values += np.arange(0.01, 1, 0.3).tolist()



#%%


from decimal import Decimal
def calculate_numeric_alpha(alpha_str, num_topics=num_topics):
    if alpha_str == 'symmetric':
        return Decimal('1.0') / num_topics
    elif alpha_str == 'asymmetric':
        return Decimal('1.0') / (num_topics + Decimal(num_topics).sqrt())
    else:
        # Use Decimal for arbitrary precision
        return Decimal(alpha_str)

def calculate_numeric_beta(beta_str, num_topics=num_topics):
    if beta_str == 'symmetric':
        return Decimal('1.0') / num_topics
    else:
        # Use Decimal for arbitrary precision
        return Decimal(beta_str)

def validate_alpha_beta(alpha_str, beta_str):
    valid_strings = ['symmetric', 'asymmetric']
    if isinstance(alpha_str, str) and alpha_str not in valid_strings:
        logging.error(f"Invalid alpha_str value: {alpha_str}. Must be 'symmetric', 'asymmetric', or a numeric value.")
        raise ValueError(f"Invalid alpha_str value: {alpha_str}. Must be 'symmetric', 'asymmetric', or a numeric value.")
    if isinstance(beta_str, str) and beta_str not in valid_strings:
        logging.error(f"Invalid beta_str value: {beta_str}. Must be 'symmetric', or a numeric value.")
        raise ValueError(f"Invalid beta_str value: {beta_str}. Must be 'symmetric', or a numeric value.")
    

#%%


"""
!!! DO NOT EXECUTE THIS CELL OR ANY CELL USING IT WITHOUT FIRSST
!!! UPDATING THE OUTPUT FILEPATH FOR THE TRAINING AND EVAL DATA
"""
import os
from json import load
import random

def get_num_records(filename):
    with open(filename, 'r') as jsonfile:
        data = load(jsonfile)
        data = data
        num_samples = len(data)  # Count the total number of samples
    return num_samples

import os
import json
from random import shuffle

def load(jsonfile):
    return json.load(jsonfile)

def futures_create_lda_datasets(filename, train_ratio, batch_size=FUTURES_BATCH_SIZE):
    with open(filename, 'r') as jsonfile:
        data = load(jsonfile)
        print(f"the number of records read from the JSON file: {len(data)}")
        num_samples = len(data)  # Count the total number of samples
        #print(f"the number of documents sampled from the JSON file: {len(data)}\n")
        
        # Shuffle data indices since we can't shuffle actual lines in a file efficiently
        indices = list(range(num_samples))
        shuffle(indices)
        
        num_train_samples = int(num_samples * train_ratio)  # Calculate number of samples for training
        
        cumulative_count = 0  # Initialize cumulative count
        # Initialize counters for train and eval datasets
        train_count = 0
        eval_count = num_train_samples
        
        # Yield batches as dictionaries for both train and eval datasets along with their sample count
        while train_count < num_train_samples or eval_count < num_samples:
            if train_count < num_train_samples:
                # Yield a training batch
                train_indices_batch = indices[train_count:train_count + batch_size]
                train_data_batch = [data[idx] for idx in train_indices_batch]
                if len(train_data_batch) > 0:
                    yield {
                        'type': 'train',
                        'data': train_data_batch,
                        'indices_batch': train_indices_batch,
                        'cumulative_count': train_count,
                        'num_samples': num_train_samples,
                        'whole_dataset': data[:num_train_samples]
                    }
                    train_count += len(train_data_batch)
                    cumulative_count += train_count
            
            if (eval_count < num_samples or train_count >= num_train_samples):
                # Yield an evaluation batch
                #print("we are in the method to create the futures trying to create the eval data.")
                #print(f"the eval count is {eval_count} and the train count is {train_count} and the num train samples is {num_train_samples}\n")
                eval_indices_batch = indices[eval_count:eval_count + batch_size]
                eval_data_batch = [data[idx] for idx in eval_indices_batch]
                #print(f"This is the size of the eval_data_batch from the create futures method {len(eval_data_batch)}\n")
                if len(eval_data_batch) > 0:
                    yield {
                        'type': 'eval',
                        'data': eval_data_batch,
                        'indices_batch': eval_indices_batch,
                        'cumulative_count': num_train_samples - eval_count,
                        'num_samples': num_train_samples - num_samples,
                        'whole_dataset': data[num_train_samples:]
                    }
                    eval_count += len(eval_data_batch)
                    cumulative_count += eval_count
                
    #garbage_collection(False,'futures_create_lda_datasets(...)')


#%%


#@dask.delayed
def create_vis(ldaModel, filename, corpus, dictionary):
    create_pylda = False
    create_pcoa = False
    PCoAfilename = filename
    filename = filename + '.html'
    

    IMAGEFILE = os.path.join(PYLDA_DIR,filename)
    PCoAIMAGEFILE = os.path.join(PCOA_DIR, PCoAfilename)

    # Disable notebook mode since we're saving to HTML.
    pyLDAvis.disable_notebook()
    
    # Prepare the visualization data.
    # Note: sort_topics=False will prevent reordering topics after training.
    try:
        vis = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary,  n_jobs=int(CORES*(2/3)), sort_topics=False)

        pyLDAvis.save_html(vis, IMAGEFILE)
        create_pylda = True

    except Exception as e:
        logging.error(f"The pyLDAvis HTML could not be saved: {e}")
        create_pylda = False


    # try Jensen-Shannon Divergence & Principal Coordinate Analysis (aka Classical Multidimensional Scaling)
    topic_distributions = [ldaModel.get_document_topics(doc, minimum_probability=0) for doc in corpus]

    # Ensure all topics are represented even if their probability is 0
    num_topics = ldaModel.num_topics
    distributions_matrix = np.zeros((len(corpus), num_topics))

    for i, doc_topics in enumerate(topic_distributions):
        for topic_num, prob in doc_topics:
            distributions_matrix[i, topic_num] = prob
    
    try: 
        pcoa_results = pyLDAvis.js_PCoA(distributions_matrix) 

        # Assuming pcoa_results is a NumPy array with shape (n_dists, 2)
        x = pcoa_results[:, 0]  # X-coordinates
        y = pcoa_results[:, 1]  # Y-coordinates

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y)  # Create a scatter plot of the PCoA results
        plt.title('PCoA Results')
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        # Save the figure as an image
        plt.savefig(f'{PCoAIMAGEFILE}.jpg')

        # If you want to save it as an HTML file instead:
        #save_html(plt.gcf(), f'{PCoAIMAGEFILE}.html')

        plt.close('all')
        plt.cla()
        plt.clf()
        garbage_collection(True,'Create Vis')
        
        create_pcoa = True
    except Exception as e: 
        logging.error(f"An error occurred during PCoA transformation: {e}")
        create_pcoa = False


    return create_pylda, create_pcoa


#%%


import hashlib
import re
# specify the chunk size for LdaModel object
# Number of documents to be used in each training chunk
CHUNKSIZE = (get_num_records(DATA_SOURCE)//5)
def train_model(n_topics: list, alpha_value: list, beta_value: list, data: list, train_eval: str, chunksize=CHUNKSIZE):
        models_data = []
        corpus_batch = []
        time_of_method_call = pd.to_datetime('now')

        model = LatentDirichletAllocation()

        # Convert the Delayed object to a Dask Bag and compute it to get the actual data
        try:
            streaming_documents = dask.compute(*data)
            chunksize = int(len(streaming_documents) // 5)
        except Exception as e:
            logging.error(f"Error computing streaming_documents data: {e}")
            raise

        # Select documents for current batch
        tokenized_documents = streaming_documents # tokenized as this is the data structure for the JSON input

        # Convert each list of tokens into a sentence string
        documents = [' '.join(tokens) for tokens in tokenized_documents]

        #print("This is the conversion of tokenized documents to original paragraphs:\n",documents[:5])

        # Iterate over each document in batch_documents
        number_of_documents = len(documents)
        logging.info(f"There is a total of {number_of_documents} documents.")
            
        # Define your parameter grid
        param_grid = {
            'n_components': n_topics,  # Number of topics
            'learning_decay': [0.5, 0.7, 0.9],
            'doc_topic_prior': alpha_value, # alpha parameter
            'topic_word_prior': beta_value, # eta parameter
            'learning_method': ['online'],
            'learning_offset': [10.0], # set to 1.0 so as to align with Gensim LDAModel defaults. SKlearn default is 10.0
            'max_iter': [PASSES], # same as passes LDAModel parameter
            'batch_size': [chunksize],
            #'n_batch_iter_': [UPDATE_EVERY], # same as Gensim LDAModel update_every parameter
            'total_samples': [number_of_documents],
            'evaluate_every': [EVAL_EVERY], # same as with Gensim LDAModel though sklearn default is -1 which causes exclusion of perplexity calc
            'n_jobs': [CORES],
            'random_state': [RANDOM_STATE]
        }
                
        # Assuming 'texts' is a list of preprocessed documents (strings)
        texts = [item for sublist in tokenized_documents for item in sublist]
        try:
            #vectorizer = CountVectorizer()
            vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{2,}\b')
            X = vectorizer.fit_transform(documents) 
        except Exception as e:
            print(f"there was an issue using the fit_transform() method on vectorized documents: {e}")
            #sys.exit()
            raise

        #try:
            # Run grid search with only the X matrix
        #    grid_search.fit(X)     
        #except Exception as e:
        #    print(f"there was an issue using the fit() method for grid search of X: {e}")
        #    raise
        #    sys.exit()

        try:
            gensim_dictionary = Dictionary(list(tokenized_documents))
        except Exception as e:
            print("the gensim dictionary couldnt be created")
        model_scores = []

        # Iterate over each document in batch_documents
        for doc_tokens in tokenized_documents:
            # Create the bag-of-words representation for the current document using the dictionary
            bow_out = gensim_dictionary.doc2bow(doc_tokens)
            # Append this representation to the corpus
            corpus_batch.append(bow_out)
            number_of_documents += 1
        print(f"There was a total of {number_of_documents} documents added to the corpus_batch.")

        # Generate all combinations of parameters using ParameterGrid
        for params in ParameterGrid(param_grid):
            print("we are inside the ParameterGrid search")
            # Initialize an LDA model with the current set of parameters
            try:
                print("we are creating the LDA model")
                lda_model = LatentDirichletAllocation(**params)
            except Exception as e:
                print(f"there was an issue in creating the lda model: {e}")
            
            # Fit the LDA model on your data (X)
            try:
                
                lda_model.fit(X)
                print("we are fitting the model")
            except Exception as e:
                print(f"there was an issue with the fit() method: {e}")
            
            # Calculate the perplexity score for the fitted LDA model on X
            try:
                
                perplexity_score = lda_model.perplexity(X)
                print("we are calculating perplexity")
            except Exception as e:
                print(f"there was an issue calculating the perplexity score: {e}")

            # Get topics from LDA model (each topic is represented as a list of word indices)
            try:
                
                lda_topics = [topic.argsort()[::-1] for topic in lda_model.components_]
                #lda_topics = [[(word_id, prob) for word_id, prob in enumerate(topic)] for topic in lda_model.components_]
                print("we are extracting the topics")
            except Exception as e:
                print(f"there was an issue in extracting the topics: {e}")
            
            # Convert word indices to actual words for coherence calculation
            try:
                print("we are converting word indices to actual words")
                topics_words = [[gensim_dictionary[id] for id in topic] for topic in lda_topics]
            except Exception as e:
                print(f"there was an issue converting word indicies: {e}")
            
            # Calculate coherence score using Gensim's CoherenceModel (you may need to adjust this part based on your setup)
            try:
                
                coherence_model_lda = CoherenceModel(topics=topics_words, texts=tokenized_documents, dictionary=gensim_dictionary, corpus=corpus_batch, coherence='c_v')
                coherence_score = coherence_model_lda.get_coherence()
                print("we are calculating the coherence score")
            except Exception as e:
                print(f"there was an issue calculating the coherence score: {e}")
            
            # Store the model, its parameters, and its scores
            model_scores.append({
                'model': lda_model,
                'params': params,
                'perplexity': perplexity_score,
                'coherence': coherence_score
            })

            try:
                # Use scikit-learn's GridSearchCV
                
                grid_search = GridSearchCV(lda_model, param_grid, n_jobs=CORES, return_train_score=True, cv=10)                      
                print("we are performing the GridSearchCV")
            except Exception as e:
                print(f"An error occurred during GridSearchCV: {e}")
                #sys.exit()

            try:
                # After fitting GridSearchCV, you can find the best model and its parameters:
                best_lda_model = grid_search.best_estimator_
                #print("Best Model's Params: ", grid_search.best_params_)
                print("Best Log Likelihood Score: ", grid_search.best_score_)
            except Exception as e:
                logging.error(f"there was a problem getting the best parameters and Log Likelihood score {e}")
                #sys.exit()

            # Get top words for each topic (similar to Gensim's top_topics) 
            try:
                print("we are getting the top words for each topic")
                # Get feature names to use as words for each topic
                feature_names = vectorizer.get_feature_names_out()
                top_words_per_topic = []
                for topic_idx, topic in enumerate(best_lda_model.components_):
                    top_features_ind = topic.argsort()[:-11:-1]  # Indices of top features/words for this topic
                    top_features = [feature_names[i] for i in top_features_ind]
                    weights = topic[top_features_ind]
                    top_words_per_topic.append((topic_idx, zip(top_features, weights)))
            except Exception as e:
                logging.error("the top words for each topic could not be extracted")

            # Serialize the LDA model using pickle
            lda_model_bytes = pickle.dumps(model) 
            best_lda_model_bytes = pickle.dumps(best_lda_model)

            current_increment_data = {
                    'type': train_eval, 
                    'batch_size': BATCH_SIZE,
                    'text': [documents],
                    'text_sha256': hashlib.sha256(documents.encode()).hexdigest(),
                    'text_md5': hashlib.md5(documents.encode()).hexdigest(),
                    'corpus': pickle.dumps(corpus_batch),
                    'dictionary': pickle.dumps(gensim_dictionary),
                    'perplexity': perplexity_score,
                    'coherence': coherence_score,
                    'topics': n_topics,
                    'alpha': alpha_value,
                    'beta': beta_value,
                    'max_iter': PASSES,
                    'eval_every': EVAL_EVERY,
                    'chunksize': chunksize,
                    'random_state': RANDOM_STATE,
                    'top_words': top_words_per_topic,
                    'lda_model': lda_model_bytes,
                    'best_lda_model': best_lda_model_bytes,
                    'model_scores': model_scores,
                    'cv_results': grid_search.cv_results_,
                    'best_params': grid_search.best_params_,
                    'log_likelihood': grid_search.best_score_,
                    'n_splits': grid_search.n_splits_,
                    'refit_time_': grid_search.refit_time_,
                    'time': time_of_method_call
            }

            models_data.append(current_increment_data)
        print("we are at the end of the train_model function")
        #garbage_collection(False, 'train_model(): convergence and perplexity score calculations')
        del documents, streaming_documents, model, gensim_dictionary, current_increment_data #, vis, success

        return models_data
         


#%%
from tqdm import tqdm

if __name__ == "__main__":
    # https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os

    cluster = LocalCluster(
            n_workers=CORES,
            threads_per_worker=THREADS_PER_CORE,
            processes=False,
            memory_limit=RAM_MEMORY_LIMIT,
            local_directory=DASK_DIR,
            #dashboard_address=None,
            dashboard_address=":8787",
            protocol="tcp",
    )


    # Create the distributed client
    client = Client(cluster)

    client.cluster.adapt(minimum=CORES, maximum=MAXIMUM_CORES)
    
    # Get information about workers from scheduler
    workers_info = client.scheduler_info()["workers"]

    # Iterate over workers and set their memory limits
    for worker_id, worker_info in workers_info.items():
        worker_info["memory_limit"] = RAM_MEMORY_LIMIT

    # Verify that memory limits have been set correctly
    #for worker_id, worker_info in workers_info.items():
    #    print(f"Worker {worker_id}: Memory Limit - {worker_info['memory_limit']}")

    # Check if the Dask client is connected to a scheduler:
    if client.status == "running":
        print("Dask client is connected to a scheduler.")
        # Scatter the embedding vectors across Dask workers
    else:
        print("Dask client is not connected to a scheduler.")
        print("The system is shutting down.")
        client.close()
        cluster.close()
        #sys.exit()

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
        print(f"{CORES} Dask workers are running.")
    else:
        print("No Dask workers are running.")
        print("The system is shutting down.")
        client.close()
        cluster.close()
        #sys.exit()

    print("Creating training and evaluation samples...")
    
    started = time()
    
    scattered_train_data_futures = []
    scattered_eval_data_futures = []

    total_num_samples = get_num_records(DATA_SOURCE)

    whole_train_dataset = None
    whole_eval_dataset = None

    with tqdm(total=total_num_samples) as pbar:
        # Process each batch as it is generated
        for batch_info in futures_create_lda_datasets(DATA_SOURCE, TRAIN_RATIO):
            if batch_info['type'] == 'train':
                # Handle training data
                #print("We are inside the IF/ELSE block for producing TRAIN scatter.")
                try:
                    scattered_future = client.scatter(batch_info['data'])
                    scattered_train_data_futures.append(scattered_future)
                except Exception as e:
                    print("there was an issue with creating the TRAIN scattered_future list")
                
                if whole_train_dataset is None:
                    whole_train_dataset = batch_info['whole_dataset']
            elif batch_info['type'] == 'eval':
                # Handle evaluation data
                #print("We are inside the IF/ELSE block for producing EVAL scatter.")
                try:
                    scattered_future = client.scatter(batch_info['data'])
                    scattered_eval_data_futures.append(scattered_future)
                except Exception as e:
                    print("there was an issue with creating the EVAL scattererd_future list.")
                    print(e)
                    
                
                if whole_eval_dataset is None:
                    whole_eval_dataset = batch_info['whole_dataset']

            # Update the progress bar with the cumulative count of samples processed
            #pbar.update(batch_info['cumulative_count'] - pbar.n)
            pbar.update(len(batch_info['data']))

        pbar.close()  # Ensure closure of the progress bar

    print(f"Completed creation of training and evaluation documents in {round((time() - started)/60,2)} minutes.\n")
   
    print("Data scatter complete...\n")

train_futures = []  # List to store futures for training
    eval_futures = []  # List to store futures for evaluation
   
    num_topics = [x for x in range(START_TOPICS, END_TOPICS + 1, STEP_SIZE)]
    num_alpha_values = len(alpha_values)
    num_beta_values = len(beta_values)


    train_eval = ['eval', 'train']

    # number of futures that complete in WAIT method
    #completed_tasks = 0
    # Create empty lists to store all future objects for training and evaluation
    train_futures = []
    eval_futures = []
    
    # Iterate over the combinations and submit tasks
    #for train_eval_type in train_eval:

        # determine if throttling is needed
    logging.info("\nEvaluating if adaptive throttling is necessary (method exponential backoff)...")
    started, throttle_attempt = time(), 0

        # https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
    while throttle_attempt < MAX_RETRIES:
        scheduler_info = client.scheduler_info()
        all_workers_below_cpu_threshold = all(worker['metrics']['cpu'] < CPU_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values())
        all_workers_below_memory_threshold = all(worker['metrics']['memory'] < MEMORY_UTILIZATION_THRESHOLD for worker in scheduler_info['workers'].values())

        if not (all_workers_below_cpu_threshold and all_workers_below_memory_threshold):
            logging.info(f"Adaptive throttling (attempt {throttle_attempt} of {MAX_RETRIES-1})")
            # Uncomment the next line if you want to log hyperparameters information as well.
            #logging.info(f"for LdaModel hyperparameters combination -- type: {train_eval_type}, topic: {n_topics}, ALPHA: {alpha_value} and ETA {beta_value}")
            sleep(exponential_backoff(throttle_attempt))
            throttle_attempt += 1
        else:
            break

    if throttle_attempt == MAX_RETRIES:
        logging.error("Maximum retries reached. The workers are still above the CPU or Memory threshold.")
        garbage_collection(True, 'Max Retries - throttling attempt')
    else:
        logging.info("Proceeding with workload as workers are below the CPU and Memory thresholds.")

    try:
        alpha_value = [float(calculate_numeric_alpha(alpha_value)) for alpha_value in alpha_values]
        beta_value = [float(calculate_numeric_alpha(beta_value)) for beta_value in beta_values]
    except Exception as e:
        print(f"there was an issue converting alpha and beta to numeric: {e}")
    
    # Submit a future for each scattered data object in the training list
    try:
        for scattered_data in scattered_train_data_futures:
            future = client.submit(train_model, num_topics, alpha_value, beta_value, scattered_data, 'train')
            train_futures.append(future)
            print(f"The training value is being appended to the train_futures list. Size: {len(train_futures)}")
    except Exception as e:
        print("Failed to submit train scatter")

        # Submit a future for each scattered data object in the evaluation list
        #if train_eval_type == 'eval':
    try:
        for scattered_data in scattered_eval_data_futures:
            future = client.submit(train_model, num_topics, alpha_value, beta_value, scattered_data, 'eval')
            eval_futures.append(future)
            print(f"The evaluation value is being appended to the eval_futures list. Size: {len(eval_futures)}")
    except Exception as e:
        print("Failed to submit eval scatter")
        #garbage_collection(False, 'client.submit(train_model(...) train and eval)')

            
    # Check if it's time to process futures based on BATCH_SIZE
    #if int(len(train_futures)/3) >= (BATCH_SIZE % 10):
    train_eval_count = len(train_futures) + len(eval_futures)
    if train_eval_count >= BATCH_SIZE:
    #if True:
        print("In holding pattern until WAIT completes.\n")
        started = time()
        #del train_eval_count
 
        # Wait for completion of eval_futures
        done_eval, not_done_eval = wait(eval_futures, timeout=None)  # return_when='FIRST_COMPLETED'
        logging.info(f"This is the size of the done_eval list: {len(done_eval)} and this is the size of the not_done_eval list: {len(not_done_eval)}")

        # Wait for completion of train_futures
        done_train, not_done_train = wait(train_futures, timeout=None)  # return_when='FIRST_COMPLETED'
        logging.info(f"This is the size of the done_train list: {len(done_train)} and this is the size of the not_done_train list: {len(not_done_train)}")

        done = done_train.union(done_eval)
        not_done = not_done_eval.union(not_done_train)
                
        elapsed_time = round(((time() - started) / 60), 2)
        logging.info(f"WAIT completed in {elapsed_time} minutes")
        print(f"This is the size of DONE {len(done)}. And this is the size of NOT_DONE {len(not_done)}\n")
        #print(f"this is the value of done_train {done_train}")

        completed_train_futures = [f for f in done_train]
        #print(f"We have completed the TRAIN list comprehension. The size is {len(completed_train_futures)}")
        #print(f"This is the length of the TRAIN completed_train_futures var {len(completed_train_futures)}")
            
        completed_eval_futures = [f for f in done_eval]
        #print(f"We have completed the EVAL list comprehension. The size is {len(completed_eval_futures)}")
        #print(f"This is the length of the EVAL completed_eval_futures var {len(completed_eval_futures)}")