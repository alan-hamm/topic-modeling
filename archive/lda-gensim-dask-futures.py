
#%%
import pyLDAvis.gensim  # Library for interactive topic model visualization
import torch  # PyTorch library for deep learning and GPU acceleration
from torch.utils.data import DataLoader  # Provides an iterator over a dataset for efficient batch processing
from tqdm import tqdm  # Creates progress bars to visualize the progress of loops or tasks
from gensim.models import LdaModel  # Implements LDA for topic modeling using the Gensim library
from gensim.corpora import Dictionary  # Represents a collection of text documents as a bag-of-words corpus
from gensim.models import CoherenceModel  # Computes coherence scores for topic models

import pickle
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
import json
import random
import logging
import objgraph
from itertools import chain
import math


#%%

# Dask dashboard throws deprecation warnings w.r.t. Bokeh
import warnings
from bokeh.util.deprecation import BokehDeprecationWarning

# Disable Bokeh deprecation warnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
# Filter out the specific warning message
warnings.filterwarnings("ignore", category=UserWarning, module="distributed.utils_perf")

#BokehDeprecationWarning: 'circle() method with size value' was deprecated in Bokeh 3.4.0 and will be removed, use 'scatter(size=...) instead' instead.
#BokehDeprecationWarning: 'circle() method with size value' was deprecated in Bokeh 3.4.0 and will be removed, use 'scatter(size=...) instead' instead.
#BokehDeprecationWarning: 'square() method' was deprecated in Bokeh 3.4.0 and will be removed, use "scatter(marker='square', ...) instead" instead.


#%%


# Define the range of number of topics for LDA and step size
START_TOPICS = 1
END_TOPICS = 2
STEP_SIZE = 1

# define the decade that is being modelled 
DECADE = '2010s'

# In the case of this machine, since it has an Intel Core i9 processor with 8 physical cores (16 threads with Hyper-Threading), 
# it would be appropriate to set the number of workers in Dask Distributed LocalCluster to 8 or slightly lower to allow some CPU 
# resources for other tasks running on your system.
CORES = 4

# specify the number of passes for Gensim LdaModel
PASSES = 15

# specify the number of iterations
ITERATIONS = 50

# specify the chunk size for LdaModel object
CHUNKSIZE = 4000


#%%


# create folder structure
log_dir = f"C:/_harvester/data/lda-models/{DECADE}_html/log/"
model_dir = f"C:/_harvester/data/lda-models/2010s_html/train-eval-data/"
image_dir = f"C:/_harvester/data/lda-models/{DECADE}_html/visuals/"
train_eval_out = f"C:/_harvester/data/lda-models/{DECADE}_html/train-eval-data/"

# Check if the directories exist and contain data
if os.path.exists(log_dir) and os.path.exists(model_dir) and os.path.exists(image_dir):
    log_files = os.listdir(log_dir)
    model_files = os.listdir(model_dir)
    image_files = os.listdir(image_dir)

    # Check if the directories are not empty
    if log_files or model_files or image_files:
        # Find an available filename for the archive
        counter = 0
        while True:
            archive_file = f"C:/_harvester/data/lda-models/{DECADE}_html/archive{counter:04d}.zip"
            if not os.path.exists(archive_file):
                break
            counter += 1

        # Create the zip file for archiving existing folders
        with zipfile.ZipFile(archive_file, 'w') as zipf:
            # Add log files to the zip file
            for log_file in log_files:
                zipf.write(os.path.join(log_dir, log_file), arcname=os.path.join("log", log_file))
            
            # Add model files to the zip file
            for model_file in model_files:
                zipf.write(os.path.join(model_dir, model_file), arcname=os.path.join("model", model_file))
            
            # Add image files to the zip file
            for image_file in image_files:
                zipf.write(os.path.join(image_dir, image_file), arcname=os.path.join("image", image_file))

        # Remove existing subdirectories after archiving them
        for subdir in [log_dir, model_dir, image_dir]:
            if os.path.exists(subdir):
                subfiles = os.listdir(subdir)
                for subfile in subfiles:
                    filepath = os.path.join(subdir, subfile)
                    if os.path.isdir(filepath):
                        os.rmdir(filepath)

# Create fresh directories for the new run
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(train_eval_out, exist_ok=True)


#%%


# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        # Get the properties of each GPU device
        gpu_properties = torch.cuda.get_device_properties(i)
        
        print(f"\nGPU Device {i} Properties:")
        print(f"Device Name: {gpu_properties.name}")
        print(f"Total Memory: {gpu_properties.total_memory / 1024**3:.2f} GB")
        print(f"Multiprocessor Count: {gpu_properties.multi_processor_count}")
        print(f"CUDA Capability Major Version: {gpu_properties.major}")
        print(f"CUDA Capability Minor Version: {gpu_properties.minor}")
else:
    print("CUDA is not available.")

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# verify if CUDA is being used or the CPU
if device is not None:
    # Check if PyTorch is currently using the GPU
    if torch.backends.cudnn.enabled:
        print("PyTorch is using the GPU.")
        cuda_version = torch.version.cuda
        print("CUDA Version:", cuda_version)
    else:
        print("PyTorch is using the CPU.")
else:
    print("The device is neither using the GPU nor CPU. An error has ocurred.")


    #%%


    # The parameter `alpha` in Latent Dirichlet Allocation (LDA) represents the concentration parameter of the Dirichlet 
# prior distribution for the topic-document distribution.
# It controls the sparsity of the resulting document-topic distributions.

# A lower value of `alpha` leads to sparser distributions, meaning that each document is likely to be associated with fewer topics.
# Conversely, a higher value of `alpha` encourages documents to be associated with more topics, resulting in denser distributions.

# The choice of `alpha` affects the balance between topic diversity and document specificity in LDA modeling.
alpha_values = np.arange(0.01, 1, 0.3).tolist()
alpha_values += ['symmetric', 'asymmetric']

# In Latent Dirichlet Allocation (LDA) topic analysis, the beta parameter represents the concentration 
# parameter of the Dirichlet distribution used to model the topic-word distribution. It controls the 
# sparsity of topics by influencing how likely a given word is to be assigned to a particular topic.

# A higher value of beta encourages topics to have a more uniform distribution over words, resulting in more 
# general and diverse topics. Conversely, a lower value of beta promotes sparser topics with fewer dominant words.

# The choice of beta can impact the interpretability and granularity of the discovered topics in LDA.
beta_values = np.arange(0.01, 1, 0.3).tolist()
beta_values += ['symmetric']


#%%


"""
The data_generator function is defined as a generator. It opens the specified JSON file (filename) 
and iterates over its lines using a for loop. Each line is parsed using json.loads() to convert it 
into a Python object (e.g., dictionary). The yield keyword is used instead of return to create a 
generator that produces one parsed JSON object at a time.

The num_samples variable counts the total number of lines in the JSON file by opening it (open(filename)) 
and iterating over its lines using a generator expression (sum(1 for _ in open(filename))). This gives 
us an estimate of how many samples are present in the dataset.

The num_train_samples variable calculates the desired number of samples for training based on the provided 
train_ratio. It multiplies num_samples by train_ratio, converting it to an integer using int().

Two empty lists, train_data and eval_data, are initialized to store training and evaluation datasets, respectively.

An instance of the `data_generator

"""

def futures_create_lda_datasets(filename, train_ratio):
    # Get the file size in bytes
    file_size = os.path.getsize(filename)

    # Get the last modified timestamp of the file
    last_modified = os.path.getmtime(filename)

    # Print the metadata
    print("\nFile Metadata:")
    print(f"Filename: {filename}")
    print(f"Size: {file_size} bytes")
    print(f"Last Modified: {last_modified}\n")
    
    with open(filename, 'r') as jsonfile:
        data = json.load(jsonfile)
    
    num_samples = len(data)  # Count the total number of samples
    num_train_samples = int(num_samples * train_ratio)  # Calculate the number of samples for training
    
    # Shuffle the data
    random.shuffle(data)

    train_data = data[:num_train_samples]  # Assign a portion of data for training
    eval_data = data[num_train_samples:]  # Assign the remaining data for evaluation
    
    #train_data=flatten_tokens(train_data)
    #eval_data=flatten_tokens(eval_data)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of eval samples: {len(eval_data)}")

    # Create delayed objects for train and eval datasets
    future_train_data = dask.delayed(train_data)
    future_eval_data = dask.delayed(eval_data)

    return future_train_data, future_eval_data


#%%

"""
This method trains a Latent Dirichlet Allocation (LDA) model using the Gensim library. Here is a breakdown of the steps involved:

    (1)The method takes in parameters such as the number of topics (n_topics), alpha and beta hyperparameters, data (a list of documents), 
        and train_eval (a boolean indicating whether it's training or evaluation).

    (2)If train_eval is True, a logging configuration is set up to log training information to a file named "train-model.log". 
        Otherwise, it logs to "eval-model.log".

    (3) Two empty lists, combined_corpus and combined_text, are initialized to store the combined corpus and text.

    (4) The number of passes for training the LDA model is set to 11.

    (5) A loop iterates over each document in the data list. Inside the loop:
            - A Gensim Dictionary object is created from the current document.
            - The document is converted into a bag-of-words representation using doc2bow().
            - A PerplexityMetric object is created to track perplexity during training.
            - If combined_text is empty, indicating that it's the first iteration:
                The initial LDA model is trained using LdaModel() with parameters such as corpus, 
                id2word (the dictionary), num_topics, alpha, beta, random_state, passes, iterations, chunksize, and per_word_topics.

            - Otherwise:
                The existing LDA model is updated with new data using lda_model_gensim.update(corpus).
                The current document's text and corpus are added to combined_text and combined_corpus respectively.

    (6) Logging is shut down.

    (7) Finally, the trained LDA model (lda_model_gensim), combined_corpus, and combined_text are returned.
"""

def train_model(n_topics: int, alpha: list, beta: list, data: list, chunksize=2000):
    combined_corpus = []  # Initialize list to store combined corpus
    combined_text = []
    
    # Convert the Delayed object to a Dask Bag and compute it to get the actual data
    streaming_documents = data.compute()
    
    # Load or create a dictionary outside the loop to track word IDs across batches
    dictionary_global = Dictionary()

    num_documents = len(streaming_documents)
    
    model_data = {
        'lda_model': [], # lda_model_gensim,
        'corpus': [], # combined_corpus,
        'text': [], #combined_text,
        'convergence': [], #convergence_score,
        'perplexity': [], # perplexity_score,
        'coherence': [], #coherence_score,
        'dictionary': [], #dictionary_global,
        'topics': [], #n_topics,
        'alpha': [], #alpha,
        'beta': []  #beta ,
    }

    batch_size = chunksize  # Number of documents to process per iteration
    
    for start_index in range(0, num_documents, batch_size):
        end_index = min(start_index + batch_size, num_documents)
        
        batch_documents = streaming_documents[start_index:end_index]

        for texts_out in batch_documents:
            dictionary_global.add_documents([texts_out])
            corpus_single_doc = [dictionary_global.doc2bow(texts_out)]
            
            lda_model_gensim = LdaModel(corpus=corpus_single_doc,
                                        id2word=dictionary_global,
                                        num_topics=n_topics,
                                        alpha=alpha,
                                        eta=beta,
                                        random_state=75,
                                        passes=PASSES,
                                        iterations=ITERATIONS,
                                        chunksize=CHUNKSIZE,
                                        per_word_topics=True)

            save_model_with_dynamic_path(lda_model_gensim, n_topics, alpha, beta).compute()
            convergence_score = lda_model_gensim.bound(corpus_single_doc)
            perplexity_score = lda_model_gensim.log_perplexity(corpus_single_doc)

            coherence_model = CoherenceModel(model=lda_model_gensim, texts=texts_out, dictionary=dictionary_global, coherence='c_v')
            coherence_score = coherence_model.get_coherence()

            combined_text.extend(texts_out)
            combined_corpus.extend(corpus_single_doc)

            model_data['lda_model'].append(lda_model_gensim)
            model_data['corpus'].append(combined_corpus.copy())
            model_data['text'].append(combined_text.copy())
            model_data['convergence'].append(convergence_score)
            model_data['perplexity'].append(perplexity_score)
            model_data['coherence'].append(coherence_score)
            model_data['dictionary'].append(dictionary_global.copy())
            model_data['topics'].append(n_topics)
            model_data['alpha'].append(alpha.copy())
            model_data['beta'].append(beta.copy())

    # Verify that combined_text contains all the original text
    original_tokens = sum((len(doc) for doc in streaming_documents), 0)
    assert len(combined_text) == original_tokens, "Combined text does not contain all the original text"

    return (lda_model_gensim, \
            combined_corpus, \
            combined_text , \
            convergence_score, \
            perplexity_score, \
            coherence_score, \
            dictionary_global, \
            n_topics, \
            alpha, \
            beta)
            

#%%

if __name__=="__main__":
    import dask   # Parallel computing library that scales Python workflows across multiple cores or machines 
    from dask.distributed import Client, LocalCluster, wait   # Distributed computing framework that extends Dask functionality 
    from dask.diagnostics import ProgressBar   # Visualizes progress of Dask computations
    from dask.distributed import progress
    from dask.delayed import Delayed # Decorator for creating delayed objects in Dask computations
    from dask.distributed import as_completed
    import dask.bag as db
    from dask.bag import Bag
    from dask import delayed
    import dask.config
    from dask.distributed import wait

    # Specify the local directory path
    DASK_DIR = '/_harvester/tmp-dask-out'

    # Deploy a Single-Machine Multi-GPU Cluster
    # https://medium.com/@aryan.gupta18/end-to-end-recommender-systems-with-merlin-part-1-89fabe2fa05b
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device IDs
    protocol = "tcp"  # "tcp" or "ucx"
    num_gpus = 1
    NUM_GPUS=[0]
    visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Select devices to place workers
    device_limit_frac = 0.7  # Spill GPU-Worker memory to host at this limit.
    device_pool_frac = 0.8
    part_mem_frac = 0.15

    # Manually specify the total device memory size (in bytes)
    device_size = 10 * 1024 * 1024 * 1024  # GPU has 12GB but setting at 10GB
            
    ram_memory_limit = "100GB" # Set the RAM memory limit (per worker)
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)

    cluster = LocalCluster(
            n_workers=4,
            threads_per_worker=2,
            processes=False,
            memory_limit=ram_memory_limit,
            local_directory=DASK_DIR,
            dashboard_address=None,
            #dashboard_address=":8787",
            #protocol="tcp",
    )


    # Create the distributed client
    client = Client(cluster)

    # Get information about workers from scheduler
    workers_info = client.scheduler_info()["workers"]

    # Iterate over workers and set their memory limits
    for worker_id, worker_info in workers_info.items():
        worker_info["memory_limit"] = ram_memory_limit

    # Verify that memory limits have been set correctly
    #for worker_id, worker_info in workers_info.items():
    #    print(f"Worker {worker_id}: Memory Limit - {worker_info['memory_limit']}")

    # Check if the Dask client is connected to a scheduler:
    if client.status == "running":
        print("Dask client is connected to a scheduler.")
        # Scatter the embedding vectors across Dask workers
    else:
        print("Dask client is not connected to a scheduler.")

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
        print("Dask workers are running.")
    else:
        print("No Dask workers are running.")


        
    # Load data from the JSON file
    filename = "C:/_harvester/data/tokenized-sentences/10s/tokenized_sents-w-bigrams.json"
    train_ratio = 0.8

    # create training and evaluation data
    print("Creating training and evaluation samples...")
    started = time()
    future_train_data, future_eval_data = futures_create_lda_datasets(filename, train_ratio)
    print(f"Completed creation of training and evaluation samples in {round((time()- started)/60,2)} minutes.\n")

    # Scatter the training data
    print(f"Scattering training samples...")
    started = time()
    train_data_scattered = client.scatter(future_train_data)
    #future_train_data_scattered = client.scatter(future_train_data)
    print(f"Training data scattered successfully in {round((time()- started)/60,2)} minutes.\n")

    # Scatter the eval data
    print(f"Scattering training samples...")
    started = time()
    eval_data_scattered = client.scatter(future_eval_data)
    #future_eval_data_scattered = client.scatter(future_eval_data)
    print(f"Evaluation data scattered successfully in {round((time()- started)/60,2)} minutes.\n")


    train_results = []  # List to store delayed objects for training
    eval_results = []  # List to store delayed objects for evaluation
    train_dict_dir = 'C:/_harvester/data/lda-models/temp_dict_out/train.pkl'
    eval_dict_dir = 'C:/_harvester/data/lda-models/temp_dict_out/eval.pkl'
    # Iterate over the range of topics from START_TOPICS to END_TOPICS with a step size of STEP_SIZE

    for n_topics in range(START_TOPICS, END_TOPICS + 1, STEP_SIZE):
        total_combinations = len(alpha_values) * len(beta_values)
        pbar = tqdm(total=total_combinations, desc=f"Scheduling model training and evaluation for {n_topics} topics")
        
        for alpha, beta in itertools.product(alpha_values, beta_values):
            #print(f"We are in the future: topics{n_topics} alpha{alpha} and beta{beta}")
            
            # Gather results immediately by computing the Future objects
            #train_data_local = client.gather(train_data_scattered)
            #eval_data_local = client.gather(eval_data_scattered)

            # Call train_model with local data
            result_train = dask.delayed(train_model)(n_topics, alpha, beta, train_data_scattered)
            result_eval = dask.delayed(train_model)(n_topics, alpha, beta, eval_data_scattered)

            # Append results to respective lists
            train_results.append(result_train)
            eval_results.append(result_eval)
            
            pbar.update(1)
        
        pbar.close()


    """
        training data
    """
        # Dictionary to hold the metrics that are generated
    train_metrics_csv = {
        'n_topics': [],
        'alpha': [],
        'beta': [],
        'cv_score': [],
        'convergence_score': [],
        'log_perplexity': [],
        'time_to_complete': []
    }

    
    # Iterate over the results of lda_models_train and combinations of alpha and beta values
    # Set the scheduler to 'distributed' using dask.config.set()
    #print("Modeling the training data...")
    started = time()
    #with dask.config.set(scheduler='distributed'):
    lda_models_train = None # Define lda_models_train with a default value
    try:
        # Compute the delayed objects in train_results using dask.compute()
        print("Creating training LDA models...")
        
        with ProgressBar():
            train_results = []
            eval_results = []

            for n_topics in range(START_TOPICS, END_TOPICS + 1, STEP_SIZE):
                for alpha_value in alpha_values:
                    for beta_value in beta_values:
                        result_train = dask.delayed(train_model)(n_topics, alpha_value, beta_value, train_data_scattered)
                        result_eval = dask.delayed(train_model)(n_topics, alpha_value, beta_value, eval_data_scattered)
                        train_results.append(result_train)
                        eval_results.append(result_eval)

            # Persist all training tasks as futures
            futures_train = client.persist(train_results)
            futures_eval = client.persist(eval_results)

            # Wait for all futures to complete their computations
            wait(futures_train)
            wait(futures_eval)

            # Gather all trained models back to the client
            trained_models = client.gather(futures_train)
            evaluated_models = client.gather(futures_eval)

            # Save each trained model with specific filenames based on its parameters
            for model_data in trained_models:
                n_topics, alpha, beta, lda_model = model_data
                
                # Normalize alpha and beta values into strings suitable for filenames
                alpha_str = '_'.join(map(str,alpha)) if isinstance(alpha,list) else str(alpha)
                beta_str = '_'.join(map(str,beta)) if isinstance(beta,list) else str(beta)
                
                # Construct a unique filename for each model using its parameters
                filename = f"lda_model_topics{n_topics}_alpha{alpha_str}_beta{beta_str}.model"
                
                # Ensure that any special characters are removed or replaced in filename components
                filename = filename.replace('.', 'p')  # Example: replace '.' with 'p'
                
                filepath = os.path.join(model_dir, filename)
                
                # Save each model directly without using dask.delayed
                lda_model.save(filepath)                  
                
                    
                # Log metrics to a file for training data
                #pd.DataFrame(model).to_csv(f'C:/_harvester/data/lda-models/{DECADE}_html/log/train-lda-tuning-results.csv', index=False) 
            print(f"Creation of training LDA models completed successfully in {round((time()- started)/60,2)} minutes.\n")
    except Exception as e:
        for result in eval_results:
            if not isinstance(result, Delayed):
                print("Invalid element found in train_results:", result)
        print(e)
 
    # Close the Dask client and cluster when done
    client.close()
    #cluster.close(timeout=60)
    cluster.close()
    #%%