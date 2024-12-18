import pyLDAvis.gensim  # Library for interactive topic model visualization
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

import logging
from datetime import datetime

DECADE_TO_PROCESS ='2010s'
LOG_DIRECTORY = f"C:/_harvester/data/lda-models/{DECADE_TO_PROCESS}_html/log/"
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

# Now when you use logging.info(), logging.debug(), etc., it will write to that log file.

# Dask dashboard throws deprecation warnings w.r.t. Bokeh
import warnings
from bokeh.util.deprecation import BokehDeprecationWarning

# Disable Bokeh deprecation warnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
# Filter out the specific warning message
# Set the logging level for distributed.utils_perf to suppress warnings
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="distributed.utils_perf")


# Define the range of number of topics for LDA and step size
START_TOPICS = 50
END_TOPICS = 61
STEP_SIZE = 1

# define the decade that is being modelled 
DECADE = DECADE_TO_PROCESS

# In the case of this machine, since it has an Intel Core i9 processor with 8 physical cores (16 threads with Hyper-Threading), 
# it would be appropriate to set the number of workers in Dask Distributed LocalCluster to 8 or slightly lower to allow some CPU 
# resources for other tasks running on your system.
CORES = 8
MAXIMUM_CORES = 12

THREADS_PER_CORE = 2

RAM_MEMORY_LIMIT = "10GB" 

# Specify the local directory path
DASK_DIR = '/_harvester/tmp-dask-out'

# specify the number of passes for Gensim LdaModel
PASSES = 15

# specify the number of iterations
ITERATIONS = 50

# specify the chunk size for LdaModel object
# Number of documents to be used in each training chunk
CHUNKSIZE = 2000

# Number of documents to be iterated through for each update. 
# Set to 0 for batch learning, > 1 for online iterative learning.
UPDATE_EVERY = 3

# Log perplexity is estimated every that many updates. 
# Setting this to one slows down training by ~2x.
EVAL_EVERY = 5

RANDOM_STATE = 75

PER_WORD_TOPICS = True

# number of documents to extract from the JSON source file when testing and developing
NUM_DOCUMENTS = 25

# the number of documents to read from the JSON source file per batch
FUTURES_BATCH_SIZE = 20

# Constants for adaptive batching and retries
# Number of futures to process per iteration
BATCH_SIZE = int(FUTURES_BATCH_SIZE * 1.2) # number of documents
MAX_BATCH_SIZE = int(BATCH_SIZE*1.5) # number of combinations
INCREASE_FACTOR = 1.05  # Increase batch size by n% upon success
DECREASE_FACTOR = .10 # Decrease batch size by p% upon failure or timeout
MAX_RETRIES = 5        # Maximum number of retries per task
BASE_WAIT_TIME = 1     # Base wait time in seconds for exponential backoff


# Load data from the JSON file
DATA_SOURCE = "C:/_harvester/data/tokenized-sentences/10s/tokenized_min_three_word-w-bigrams-08312024.json"
TRAIN_RATIO = 0.8

TIMEOUT = None #"90 minutes"

EXTENDED_TIMEOUT = None #"120 minutes"

CPU_UTILIZATION_THRESHOLD = 85 # ie 95%

# Enable serialization optimizations
dask.config.set(scheduler='distributed', serialize=True)
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.scheduler.worker-ttl": None})
#dask.config.set({"distributed.scheduler.worker-ttl": None})

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


import os
import pandas as pd
import zipfile

# Define the top-level directory and subdirectories
DECADE = "2010s"  # Replace with your actual decade value
ROOT_DIR = f"C:/_harvester/data/lda-models/{DECADE}_html"
LOG_DIR = os.path.join(ROOT_DIR, "log")
IMAGE_DIR = os.path.join(ROOT_DIR, "visuals")
METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
TEXTS_ZIP_DIR = os.path.join(ROOT_DIR, "texts_zip")

# Ensure that all necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
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
def add_model_data_to_metadata(model_data):
    #print("we are in the add_model_data_to_metadata method()")
    # Save large body of text to zip and update model_data reference
    texts_zipped = []
    
    for text_list in model_data['text']:
        combined_text = ''.join([''.join(sent) for sent in text_list])  # Combine all sentences into one string
        zip_path = save_to_zip(model_data['time'], combined_text, model_data['lda_model'])
        texts_zipped.append(zip_path)
    # Update model data with zipped paths
    model_data['text'] = texts_zipped

     # Ensure other fields are not lists, or if they are, they should have only one element per model
    for key, value in model_data.items():
        if isinstance(value, list) and key != 'text':
            assert len(value) == 1, f"Field {key} has multiple elements"
            model_data[key] = value[0]  # Unwrap single-element list
               
    # Define the expected data types for each column
    expected_dtypes = {
        'type': str,
        'batch_size': int,
        'text': object,  # Use object dtype for lists of strings (file paths)
        'text_sha256': str,
        'text_md5': str,
        'convergence': 'float64',
        'perplexity': 'float64',
        'coherence': 'float64',
        'topics': int,
        # Use pd.Categorical.dtype for categorical columns
        # Ensure alpha and beta are already categorical when passed into this function
        # They should not be wrapped again with CategoricalDtype here.
        'alpha_str': str,
        'n_alpha': 'float64',
        'beta_str': str,
        'n_beta': 'float64',
        'passes': int,
        'iterations': int,
        'update_every': int,
        'eval_every': int,
        'chunksize': int,
        'random_state': int,
        'per_word_topics': bool,
        'lda_model': object,
        # Enforce datetime type for time
        'time': 'datetime64[ns]',
    }   

    
    try:
        #df_new_metadata = pd.DataFrame({key: [value] if not isinstance(value, list) else value 
        #                                for key, value in model_data.items()}).astype(expected_dtypes)
        # Create a new DataFrame without enforcing dtypes initially
        df_new_metadata = pd.DataFrame({key: [value] if not isinstance(value, list) else value 
                                        for key, value in model_data.items()})
        
        # Apply type conversion selectively
        #for col_name in ['convergence', 'perplexity', 'coherence', 'n_beta', 'n_alpha']:
        for col_name in ['convergence', 'perplexity', 'coherence', 'n_beta', 'n_alpha']:
            df_new_metadata[col_name] = df_new_metadata[col_name].astype('float64')
            
        df_new_metadata['topics'] = df_new_metadata['topics'].astype(int)
        #df_new_metadata['time'] = pd.to_datetime(df_new_metadata['time'])
        df_new_metadata['batch_size'] = BATCH_SIZE
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

PERFORMANCE_TRAIN_LOG = os.path.join(LOG_DIR, "train_model_performance.html")
# INCLUDE EVAL AND TRAINING DATA OUTPUT FILEPATHS HERE


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
        print(f"the number of documents sampled from the JSON file: {len(data)}\n")
        
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
            
            if eval_count < num_samples or train_count >= num_train_samples:
                # Yield an evaluation batch
                #print("we are in the method to create the futures trying to create the eval data.")
                #print(f"the eval count is {eval_count} and the train count is {train_count} and the num train samples is {num_train_samples}")
                eval_indices_batch = indices[eval_count:eval_count + batch_size]
                eval_data_batch = [data[idx] for idx in eval_indices_batch]
                yield {
                    'type': 'eval',
                    'data': eval_data_batch,
                    'indices_batch': eval_indices_batch,
                    'cumulative_count': eval_count - num_train_samples,
                    'num_samples': num_samples - num_train_samples,
                    'whole_dataset': data[num_train_samples:]
                }
                eval_count += len(eval_data_batch)
                cumulative_count += eval_count
                
        #garbage_collection(False,'futures_create_lda_datasets(...)')

# create training and eval dictionaries used in train_model(...) method
def create_dictionary(filename):
    with open(filename, 'r') as jsonfile:
        data = load(jsonfile)
        num_samples = len(data)  # Count the total number of samples
        logging.info(f"The min five with bigrams has {num_samples} sentences")
        return data
#minfivedict = create_dictionary(DATA_SOURCE)

def create_vis(ldaModel, filename, corpus, dictionary):
    LOGFILE = os.path.join(IMAGE_DIR,filename)

    pyLDAvis.disable_notebook()
    vis = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary)

    pyLDAvis.save_html(vis, LOGFILE)

import hashlib
def train_model(n_topics: int, alpha_str: list, beta_str: list, data: list, train_eval: str, chunksize=CHUNKSIZE):
        models_data = []
        coherehce_score_list = []
        corpus_batch = []
        zipped_texts = []
        time_of_method_call = pd.to_datetime('now')

        #print("this is an investigation into the full datafile")
        #pp.pprint(full_datafile)
        # Convert the Delayed object to a Dask Bag and compute it to get the actual data
        try:
            streaming_documents = dask.compute(*data)
            #print("these are the streaming documents")
            #print(streaming_documents)
            #garbage_collection(False, 'train_model(): streaming_documents = dask.compute(*data)')
        except Exception as e:
            logging.error(f"Error computing streaming_documents data: {e}")
            raise
        #print(f"This is the dtype for 'streaming_documents' {type(streaming_documents)}.\n")  # Should output <class 'tuple'>
        #print(streaming_documents[0][0])     # Check the first element to see if it's as expected

        # Select documents for current batch
        batch_documents = streaming_documents
        
        # Create a new Gensim Dictionary for the current batch
        try:
            dictionary_batch = Dictionary(list(batch_documents))
            #print("The dictionary was cretaed.")
        except TypeError:
            print("Error: The data structure is not correct.")
        #else:
        #    print("Dictionary created successfully!")

        #if isinstance(batch_documents[0], list) and all(isinstance(doc, list) for doc in batch_documents[0]):
        #bow_out = dictionary_batch.doc2bow(batch_documents[0])
        flattened_batch = [item for sublist in batch_documents for item in sublist]
        #bow_out = dictionary_batch.doc2bow(flattened_batch)
        #else:
        #    raise ValueError(f"Expected batch_documents[0] to be a list of token lists. Instead received {type(batch_documents[0])} with value {batch_documents[0]}\n")

        # Iterate over each document in batch_documents
        number_of_documents = 0
        for doc_tokens in batch_documents:
            # Create the bag-of-words representation for the current document using the dictionary
            bow_out = dictionary_batch.doc2bow(doc_tokens)
            # Append this representation to the corpus
            corpus_batch.append(bow_out)
            number_of_documents += 1
        logging.info(f"There was a total of {number_of_documents} documents added to the corpus_batch.")
            
        #logger.info(f"HERE IS THE TEXT for corpus_batch using LOGGER: {corpus_batch}\n")
        #except Exception as e:
        #    logger.error(f"An unexpected error occurred with BOW_OUT: {e}")
                
        #if isinstance(texts_out[0], list):
        #    texts_batch.append(texts_out[0])
        #else:
        #    logging.error("Expected texts_out to be a list of strings (words), got:", texts_out[0])
        #    raise ValueError("Expected texts_out to be a list of strings (words), got:", texts_out[0])
                
        n_alpha = calculate_numeric_alpha(alpha_str)
        n_beta = calculate_numeric_beta(beta_str)
        try:
            #logger.info("we are inside the try block at the beginning")
            lda_model_gensim = LdaModel(corpus=corpus_batch,
                                                id2word=dictionary_batch,
                                                num_topics=n_topics,
                                                alpha= float(n_alpha),
                                                eta= float(n_beta),
                                                random_state=RANDOM_STATE,
                                                passes=PASSES,
                                                iterations=ITERATIONS,
                                                update_every=UPDATE_EVERY,
                                                eval_every=EVAL_EVERY,
                                                chunksize=CHUNKSIZE,
                                                per_word_topics=True)
            #logger.info("we are inside the try block after the constructor")

                                          
        except Exception as e:
            logging.error(f"An error occurred during LDA model training: {e}")
            raise  # Optionally re-raise the exception if you want it to propagate further      

        ldamodel_bytes = pickle.dumps(lda_model_gensim)

        #coherence_score = None  # Assign a default value
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                #coherence_model_lda = CoherenceModel(model=lda_model_gensim, processes=math.floor(CORES*(2/3)), dictionary=dictionary_batch, texts=batch_documents[0], coherence='c_v') 
                coherence_model_lda = CoherenceModel(model=lda_model_gensim, processes=math.floor(CORES*(1/3)), dictionary=dictionary_batch, texts=batch_documents, coherence='c_v') 
                coherence_score = coherence_model_lda.get_coherence()
                coherehce_score_list.append(coherence_score)
            except Exception as e:
                logging.error("there was an issue calculating coherence score. value 'Inf' has been assigned.\n")
                coherence_score = float('inf')
                coherehce_score_list.append(coherence_score)
                #sys.exit()

        try:
            convergence_score = lda_model_gensim.bound(corpus_batch)
        except Exception as e:
            logging.error("there was an issue calculating convergence score. value 'Inf' has been assigned.\n")
            convergence_score = float('inf')
                    
        try:
            perplexity_score = lda_model_gensim.log_perplexity(corpus_batch)
        except RuntimeWarning as e:
            logging.info("there was an issue calculating perplexity score. value 'Inf' has been assigned.\n")
            perplexity_score = float('inf')
            #sys.exit()
        
        top_words = []
        for topic_id, word_value in lda_model_gensim.print_topics(num_topics=n_topics):
            for str_float in word_value:
                top_words.append(str_float)
            
       #print(f"type: {train_eval}, coherence: {coherence_score}, n_topics: {n_topics}, n_alpha: {n_alpha}, alpha_str: {alpha_str}, n_beta: {n_beta}, beta_str: {beta_str}")
        logging.info(f"type: {train_eval}, coherence: {coherence_score}, n_topics: {n_topics}, n_alpha: {n_alpha}, alpha_str: {alpha_str}, n_beta: {n_beta}, beta_str: {beta_str}\n      batch documents: {batch_documents}\n")     

        # transform list of tokens comprising the doc into a single string
        string_result = ' '.join(map(str, flattened_batch))

        # Convert numeric beta value to string if necessary
        if isinstance(beta_str, float):
            beta_str = str(beta_str)
                
        # Convert numeric alpha value to string if necessary
        if isinstance(alpha_str, float):
            alpha_str = str(alpha_str)

        current_increment_data = {
                'type': train_eval, 
                'batch_size': BATCH_SIZE,
                'text': [string_result],
                'text_sha256': hashlib.sha256(string_result.encode()).hexdigest(),
                'text_md5': hashlib.md5(string_result.encode()).hexdigest(),
                'convergence': convergence_score,
                'perplexity': perplexity_score,
                'coherence': coherence_score,
                'topics': n_topics,
                'alpha_str': [alpha_str],
                'n_alpha': calculate_numeric_alpha(alpha_str),
                'beta_str': [beta_str],
                'n_beta': calculate_numeric_beta(beta_str),
                'passes': PASSES,
                'iterations': ITERATIONS,
                'update_every': UPDATE_EVERY,
                'eval_every': EVAL_EVERY,
                'chunksize': CHUNKSIZE,
                'random_state': RANDOM_STATE,
                'per_word_topics': PER_WORD_TOPICS,
                'lda_model': ldamodel_bytes,
                'time': time_of_method_call
        }

        models_data.append(current_increment_data)
        #garbage_collection(False, 'train_model(): convergence and perplexity score calculations')

        return models_data
"""
                    - The `process_completed_future` function is called when all futures in a batch complete within the specified timeout. It 
                        can be used to continue with your program using both completed training and evaluation futures.
                    - The `retry_processing` function is called when there are incomplete futures after iterating through a batch of 
                        data. It can be used to retry processing with those incomplete futures.
                    - The code checks if there are any remaining futures in the lists after completing all iterations. If so, it 
                        waits for them to complete and handles them accordingly.
"""

# List to store parameters of models that failed to complete even after a retry
failed_model_params = []

# Mapping from futures to their corresponding parameters (n_topics, alpha_value, beta_value)
future_to_params = {}
def process_completed_futures(completed_train_futures, completed_eval_futures, log_dir):
    #print("we are in the process_completed_futures method()")
    # Process training futures
    for future in completed_train_futures:
        try:
            # Retrieve the result of the training future
            #if isinstance(future.result(), list):
            models_data = future.result()  # This should be a list of dictionaries
            if not isinstance(models_data, list):
                models_data = list(future.result())  # This should be a list of dictionaries
            logging.info(f"this is the value of the TRAIN MODELS_DATA within the process_completed method: {models_data}")
            #else:
            #    models_data = list(future.result())
            #print("this is the value of models data:", models_data)
            
        except TypeError as e:
            logging.error(f"Error occurred during training: {e}")
            sys.exit()
        else:
            # Iterate over each model's data and save it
            for model_data in models_data:
                # Check if models_data is a non-empty list before iterating
                if isinstance(models_data, list) and models_data:
                    for model_data in models_data:
                        logging.info(f"this is the value of model TRAIN data: {model_data}")
                        #save_model_and_log(model_data=model_data, log_dir=log_dir, train_or_eval=True)
                        add_model_data_to_metadata(model_data)
                else:
                    # Handle the case where models_data is not as expected
                    logging.info(f"Received unexpected result from TRAIN future: {models_data}")

    # Process evaluation futures
    for future in completed_eval_futures:
        try:
            # Retrieve the result of the training future
            #if isinstance(future.result(), list):
            models_data = future.result()  # This should be a list of dictionaries
            if not isinstance(models_data, list):
                models_data = list(future.result())  # This should be a list of dictionaries
            logging.info(f"this is the value of the EVAL MODELS_DATA within the process_completed method: {models_data}")
            #else:
            #    models_data = list(future.result())
            #print("this is the value of models data:", models_data)
        except TypeError as e:
            logging.error(f"Error occurred during evaluation: {e}")
            sys.exit()
        else:
            # Iterate over each model's data and save it
            for model_data in models_data:
                # Check if models_data is a non-empty list before iterating
                if isinstance(models_data, list) and models_data:
                    for model_data in models_data:
                        logging.info(f"this is the value of model EVAL data: {model_data}")
                        #save_model_and_log(model_data=model_data, log_dir=log_dir, train_or_eval=False)
                        add_model_data_to_metadata(model_data)
                else:
                    # Handle the case where models_data is not as expected
                    logging.info(f"Received unexpected result from EVAL future: {models_data}")
                
    #garbage_collection(False, 'process_completed_futures(...)')

            
# Function to retry processing with incomplete futures
def retry_processing(incomplete_train_futures, incomplete_eval_futures, timeout=None):
    #print("we are in the retry_processing method()")
    # Retry processing with incomplete futures using an extended timeout
    # Process completed ones after reattempting
    #done_train = [f for f in done if f in train_futures]
    #done_eval = [f for f in done if f in eval_futures]
    # Wait for completion of eval_futures
    done_eval, not_done_eval = wait(incomplete_eval_futures, timeout=timeout)  # return_when='FIRST_COMPLETED'
    #print(f"This is the size of the done_eval list: {len(done_eval)} and this is the size of the not_done_eval list: {len(not_done_eval)}")

    # Wait for completion of train_futures
    done_train, not_done_train = wait(incomplete_train_futures, timeout=timeout)  # return_when='FIRST_COMPLETED'
    #print(f"This is the size of the done_train list: {len(done_train)} and this is the size of the not_done_train list: {len(not_done_train)}")

    done = done_train.union(done_eval)
    not_done = not_done_eval.union(not_done_train)
                
    #print(f"WAIT completed in {elapsed_time} minutes")
    #print(f"This is the size of DONE {len(done)}. And this is the size of NOT_DONE {len(not_done)}\n")
    #print(f"this is the value of done_train {done_train}")

    completed_train_futures = [f for f in done_train]
    #print(f"We have completed the TRAIN list comprehension. The size is {len(completed_train_futures)}")
    #print(f"This is the length of the TRAIN completed_train_futures var {len(completed_train_futures)}")
            
    completed_eval_futures = [f for f in done_eval]
    #print(f"We have completed the EVAL list comprehension. The size is {len(completed_eval_futures)}")
    #print(f"This is the length of the EVAL completed_eval_futures var {len(completed_eval_futures)}")

    #logging.info(f"This is the size of completed_train_futures {len(completed_train_futures)} and this is the size of completed_eval_futures {len(completed_eval_futures)}")
    if len(completed_eval_futures) > 0 or len(completed_train_futures) > 0:
        process_completed_futures(completed_train_futures, completed_eval_futures, LOG_DIR) 
    
    # Record parameters of still incomplete futures for later review
    failed_model_params.extend(future_to_params[future] for future in not_done)
    print("We have exited the retry_preprocessing() method.")
    logging.info(f"There were {len(not_done_eval)} EVAL documents that couldn't be processed.")
    logging.info(f"There were {len(not_done_train)} TRAIN documents that couldn't be processed.")

    #garbage_collection(False, 'retry_processing(...)')

    # Dictionary to keep track of retries for each task
task_retries = {}

# Function to perform exponential backoff
def exponential_backoff(attempt):
    return BASE_WAIT_TIME * (2 ** attempt)

# Function to handle failed futures and potentially retry them
def handle_failed_future(future, future_to_params, train_futures, eval_futures, client):
    print("We are in the handle_failed_future() method.\n")
    params = future_to_params[future]
    attempt = task_retries.get(params, 0)
    
    if attempt < MAX_RETRIES:
        print(f"Retrying task {params} (attempt {attempt + 1}/{MAX_RETRIES})")
        wait_time = exponential_backoff(attempt)
        sleep(wait_time)  
        
        task_retries[params] = attempt + 1
        
        new_future_train = client.submit(train_model, *params)
        new_future_eval = client.submit(train_model, *params)
        
        future_to_params[new_future_train] = params
        future_to_params[new_future_eval] = params
        
        train_futures.append(new_future_train)
        eval_futures.append(new_future_eval)
    else:
        print(f"Task {params} failed after {MAX_RETRIES} attempts. No more retries.")

    #garbage_collection(False,'handle_failed_future')

from tqdm import tqdm
if __name__=="__main__":

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
        sys.exit()

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
        print("Dask workers are running.")
    else:
        print("No Dask workers are running.")
        print("The system is shutting down.")
        client.close()
        cluster.close()
        sys.exit()

    print("Creating training and evaluation samples...")
    
    started = time()
    
    scattered_train_data_futures = []
    scattered_eval_data_futures = []

    total_num_samples = get_num_records(DATA_SOURCE)

    whole_train_dataset = None
    whole_eval_dataset = None

    with tqdm(total=total_num_samples) as pbar:
        # Process each batch as it is generated
        for batch_info in futures_create_lda_datasets(DATA_SOURCE, TRAIN_RATIO, 5):
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
                
                if whole_eval_dataset is None:
                    whole_eval_dataset = batch_info['whole_dataset']

            # Update the progress bar with the cumulative count of samples processed
            #pbar.update(batch_info['cumulative_count'] - pbar.n)
            pbar.update(len(batch_info['data']))

        pbar.close()  # Ensure closure of the progress bar

    print(f"Completed creation of training and evaluation documents in {round((time() - started)/60,2)} minutes.\n")
   
    print("Data scatter complete...\n")
    #garbage_collection(False, 'scattering training and eval data')


    train_futures = []  # List to store futures for training
    eval_futures = []  # List to store futures for evaluation
   
    num_topics = len(range(START_TOPICS, END_TOPICS + 1, STEP_SIZE))
    num_alpha_values = len(alpha_values)
    num_beta_values = len(beta_values)

    TOTAL_MODELS = (num_topics * num_alpha_values * num_beta_values) * 2

    progress_bar = tqdm(total=TOTAL_MODELS, desc="Creating and saving models")

    train_eval = ['eval', 'train']

    # Create a list of all combinations of n_topics, alpha_value, beta_value, and train_eval
    combinations = list(itertools.product(range(START_TOPICS, END_TOPICS + 1, STEP_SIZE), alpha_values, beta_values, train_eval))

    # Separate the combinations into two lists based on 'train' and 'eval'
    train_combinations = [combo for combo in combinations if combo[-1] == 'train']
    eval_combinations = [combo for combo in combinations if combo[-1] == 'eval']

    # Calculate the sample size for each category
    sample_size = min(len(train_combinations), len(eval_combinations))

    # Select random combinations from each category
    random_train_combinations = random.sample(train_combinations, sample_size)
    random_eval_combinations = random.sample(eval_combinations, sample_size)

    # Combine the randomly selected train and eval combinations
    random_combinations = random_eval_combinations+ random_train_combinations
    sample_size = max(1, int(len(combinations) * 0.375))

    # Select random_combinations conditionally
    random_combinations = random.sample(combinations, sample_size) if sample_size < len(combinations) else combinations
    print(f"The random sample contains {len(random_combinations)}")

    # Determine which combinations were not drawn by using set difference
    undrawn_combinations = list(set(combinations) - set(random_combinations))

    print(f"this leaves {len(undrawn_combinations)} remaining\n")

    # Create empty lists to store all future objects for training and evaluation
    train_futures = []
    eval_futures = []
    
    # Iterate over the combinations and submit tasks
    for n_topics, alpha_value, beta_value, train_eval_type in random_combinations:

        # determine if throttling is needed
        logging.info("\nEvaluating if adaptive throttling is necessary (method exponential backoff)...")
        started, throttle_attempt = time(), 0

        while throttle_attempt < MAX_RETRIES and not all(worker['metrics']['cpu'] < CPU_UTILIZATION_THRESHOLD for worker in client.scheduler_info()['workers'].values()):
            logging.info(f"Adaptive throttling (attempt {throttle_attempt} of {MAX_RETRIES-1}")
            #logging.info(f"for LdaModel hyperparameters combination -- type: {train_eval_type}, topic: {n_topics}, ALPHA: {alpha_value} and ETA {beta_value}")
            sleep(exponential_backoff(throttle_attempt))
            throttle_attempt += 1

        logging.info(f"Adaptive throttling (method: exponential backoff) {'completed in {:.2f} seconds'.format(time() - started) if throttle_attempt else 'was not necessary...'}\n")

        logging.info(f"for LdaModel hyperparameters combination -- type: {train_eval_type}, topic: {n_topics}, ALPHA: {alpha_value} and ETA {beta_value}")
        # Submit a future for each scattered data object in the training list
        #if train_eval_type == 'train':
        # Submit a future for each scattered data object in the training list
        for scattered_data in scattered_train_data_futures:
            future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'train')
            train_futures.append(future)
            logging.info(f"The training value is being appended to the train_futures list. Size: {len(train_futures)}")

        # Submit a future for each scattered data object in the evaluation list
        #if train_eval_type == 'eval':
        for scattered_data in scattered_eval_data_futures:
            future = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_data, 'eval')
            eval_futures.append(future)
            logging.info(f"The evaluation value is being appended to the eval_futures list. Size: {len(eval_futures)}")
        #garbage_collection(False, 'client.submit(train_model(...) train and eval)')


        # Map the created futures to their parameters so we can identify them later if needed
        for future in train_futures:
            future_to_params[future] = ('train',n_topics, alpha_value, beta_value)

        # Do the same for eval_futures
        for future in eval_futures:
            future_to_params[future] = ('eval', n_topics, alpha_value, beta_value)

        #train_futures.append(all_train_futures)
        #eval_futures.append(all_eval_futures)
        #print(f"This is the size of the eval_futures {len(eval_futures)}")
        #print(f"this is the eval futures: {eval_futures}\n\n")
            
        # Check if it's time to process futures based on BATCH_SIZE
        #if int(len(train_futures)/3) >= (BATCH_SIZE % 10):
        train_eval_count = train_futures + eval_futures
        if int(len(train_eval_count)) >= (BATCH_SIZE):
            print("In holding pattern until WAIT completes.")
            started = time()
                
            #done, not_done = wait(train_futures + eval_futures, timeout=None)        # Wait for all reattempted futures with an extended timeout (e.g., 120 seconds)

            # Process completed ones after reattempting
            #done_train = [f for f in done if f in train_futures]
            #done_eval = [f for f in done if f in eval_futures]
            # Wait for completion of eval_futures
            done_eval, not_done_eval = wait(eval_futures, timeout=None)  # return_when='FIRST_COMPLETED'
            print(f"This is the size of the done_eval list: {len(done_eval)} and this is the size of the not_done_eval list: {len(not_done_eval)}")

            # Wait for completion of train_futures
            done_train, not_done_train = wait(train_futures, timeout=None)  # return_when='FIRST_COMPLETED'
            print(f"This is the size of the done_train list: {len(done_train)} and this is the size of the not_done_train list: {len(not_done_train)}")

            done = done_train.union(done_eval)
            not_done = not_done_eval.union(not_done_train)
                
            elapsed_time = round(((time() - started) / 60), 2)
            print(f"WAIT completed in {elapsed_time} minutes")
            print(f"This is the size of DONE {len(done)}. And this is the size of NOT_DONE {len(not_done)}\n")
            #print(f"this is the value of done_train {done_train}")

            completed_train_futures = [f for f in done_train]
            print(f"We have completed the TRAIN list comprehension. The size is {len(completed_train_futures)}")
            print(f"This is the length of the TRAIN completed_train_futures var {len(completed_train_futures)}")
            
            completed_eval_futures = [f for f in done_eval]
            print(f"We have completed the EVAL list comprehension. The size is {len(completed_eval_futures)}")
            print(f"This is the length of the EVAL completed_eval_futures var {len(completed_eval_futures)}")

            logging.info(f"This is the size of completed_train_futures {len(completed_train_futures)} and this is the size of completed_eval_futures {len(completed_eval_futures)}")
            process_completed_futures(completed_train_futures, completed_eval_futures, LOG_DIR)
            progress_bar.update(len(done))

            # Handle failed futures using the previously defined function
            for future in not_done:
                failed_future_timer = time()
                print("Handling of failed future method has been initiated.")
                handle_failed_future(future, future_to_params, train_futures,  eval_futures, client)
                elapsed_time = round(((time() - started) / 60), 2)
                print(f"It took {elapsed_time} minutes to handle the failed future.")


            # If no tasks are pending (i.e., all have been processed), consider increasing BATCH_SIZE.
            completed_tasks = 0
            completed_tasks += len(done_train) + len(done_eval)

            # If no tasks are pending (i.e., all have been processed), consider increasing BATCH_SIZE.
            if completed_tasks == len(train_futures) + len(eval_futures):
                BATCH_SIZE = int(math.ceil(BATCH_SIZE * INCREASE_FACTOR)) if int(math.ceil(BATCH_SIZE * INCREASE_FACTOR)) < MAX_BATCH_SIZE else MAX_BATCH_SIZE
                print(f"Increasing batch size to {BATCH_SIZE}")

            # If there are any tasks that were not done, consider decreasing BATCH_SIZE.
            else:
                BATCH_SIZE = max(1, int(BATCH_SIZE * DECREASE_FACTOR)) if max(1, int(BATCH_SIZE * DECREASE_FACTOR)) > 0 else BATCH_SIZE
                print(f"Decreasing batch size to {BATCH_SIZE}")

            # reset lists to empty for next iteration of models
            train_futures.clear()
            eval_futures.clear()

            #defensive programming to ensure utility model lists are empty
            done.clear()
            not_done.clear()
            done_train.clear()
            done_eval.clear()
            not_done_eval.clear()
            not_done_train.clear()
            
    progress_bar.close()

    # After all loops have finished running...
    if len(train_futures) > 0 or len(eval_futures) > 0:
        print("we are in the first IF statement for retry_processing()")
        retry_processing(train_futures, eval_futures, TIMEOUT)


    # Now give one more chance with extended timeout only to those that were incomplete previously
    if len(failed_model_params) > 0:
        print("Retrying incomplete models with extended timeout...")
        
        # Create new lists for retrying futures
        retry_train_futures = []
        retry_eval_futures = []

        # Resubmit tasks only for those that failed in the first attempt
        for params in failed_model_params:
            n_topics, alpha_value, beta_value = params
            
            with performance_report(filename=PERFORMANCE_TRAIN_LOG):
                future_train_retry = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_train_data_futures, 'train')
                future_eval_retry = client.submit(train_model, n_topics, alpha_value, beta_value, scattered_eval_data_futures, 'eval')

            retry_train_futures.append(future_train_retry)
            retry_eval_futures.append(future_eval_retry)

            # Keep track of these new futures as well
            future_to_params[future_train_retry] = params
            future_to_params[future_eval_retry] = params

        # Clear the list of failed model parameters before reattempting
        failed_model_params.clear()

        # Wait for all reattempted futures with an extended timeout (e.g., 120 seconds)
        done, not_done = wait(retry_train_futures + retry_eval_futures ) #, timeout=EXTENDED_TIMEOUT)

        # Process completed ones after reattempting
        process_completed_futures([f for f in done if f in retry_train_futures],
                                [f for f in done if f in retry_eval_futures],
                                LOG_DIR)
        
        progress_bar.update(len(done))

        # Record parameters of still incomplete futures after reattempting for later review
        for future in not_done:
            failed_model_params.append(future_to_params[future])

        # At this point `failed_model_params` contains the parameters of all models that didn't complete even after a retry

    #client.close()
    print("The training and evaluation loop has completed.")

    if len(failed_model_params) > 0:
        # You can now review `failed_model_params` to see which models did not complete successfully.
        logging.error("The following model parameters did not complete even after a second attempt:")
    #    perf_logger.info("The following model parameters did not complete even after a second attempt:")
        for params in failed_model_params:
            logging.error(params)
    #        perf_logger.info(params)
            
client.close()
cluster.close()