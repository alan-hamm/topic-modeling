"""
This code was written using CDC AI Chatbot. A variety of prompts were used, including questions and prompts to 
    correct bugs, memory issues(ie too little resources available), generate comments, etc.

maintenance: alan hamm(pqn7)
apr 2024
"""

#%%
import torch  # PyTorch library for deep learning and GPU acceleration
from torch.utils.data import DataLoader  # Provides an iterator over a dataset for efficient batch processing
from tqdm import tqdm  # Creates progress bars to visualize the progress of loops or tasks
from sklearn.feature_extraction.text import CountVectorizer  # Converts text documents into numerical representations
from sklearn.decomposition import LatentDirichletAllocation  # Implements Latent Dirichlet Allocation (LDA) for topic modeling
from gensim.models import LdaModel  # Implements LDA for topic modeling using the Gensim library
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary  # Represents a collection of text documents as a bag-of-words corpus
from gensim.models import CoherenceModel
import gensim

import os  # Provides functions for interacting with the operating system, such as creating directories
import pickle  # Allows objects to be serialized and deserialized to/from disk
import itertools  # Provides various functions for efficient iteration and combination of elements
import numpy as np  # Library for numerical computing in Python, used for array operations and calculations
from time import time  # Measures the execution time of code snippets or functions
import pprint as pp  # Pretty-printing library, used here to format output in a readable way
import multiprocessing
import pandas as pd

from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import triu

import pyLDAvis

import dask
import dask
from dask.distributed import Client, LocalCluster #, LocalCUDACluster
from dask.diagnostics import ProgressBar
import dask.bag as db
import torch
import pickle
import itertools
from gensim.models import Word2Vec
import cupy as cp
import webbrowser
from torchtext.vocab import GloVe
from gensim.models import KeyedVectors


#%%
# Dask dashboard throws deprecation warnings w.r.t. Bokeh
import warnings
from bokeh.util.deprecation import BokehDeprecationWarning

# Disable Bokeh deprecation warnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)

#BokehDeprecationWarning: 'circle() method with size value' was deprecated in Bokeh 3.4.0 and will be removed, use 'scatter(size=...) instead' instead.
#BokehDeprecationWarning: 'circle() method with size value' was deprecated in Bokeh 3.4.0 and will be removed, use 'scatter(size=...) instead' instead.
#BokehDeprecationWarning: 'square() method' was deprecated in Bokeh 3.4.0 and will be removed, use "scatter(marker='square', ...) instead" instead.


#%%

# Define the range of number of topics for LDA and step size
start_topics = 74
end_topics = 82
step_size = 2

# Specify output directories for log file, model outputs, and images generated.
log_dir = "C:/_harvester/data/lda-models/2010s_html.json/"
model_dir = "C:/_harvester/data/lda-models/2010s_html.json/lda-models/"
image_dir = "C:/_harvester/data/lda-models/2010s_html.json/visuals/"

# Create directories if they don't exist.
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

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

cores = multiprocessing.cpu_count() - 1 # Count the number of cores in a computer


#%%
# The parameter `alpha` in Latent Dirichlet Allocation (LDA) represents the concentration parameter of the Dirichlet 
# prior distribution for the topic-document distribution.
# It controls the sparsity of the resulting document-topic distributions.

# A lower value of `alpha` leads to sparser distributions, meaning that each document is likely to be associated with fewer topics.
# Conversely, a higher value of `alpha` encourages documents to be associated with more topics, resulting in denser distributions.

# The choice of `alpha` affects the balance between topic diversity and document specificity in LDA modeling.
alpha_values = np.arange(0.01, 1, 0.3).tolist()


#%%
# In Latent Dirichlet Allocation (LDA) topic analysis, the beta parameter represents the concentration 
# parameter of the Dirichlet distribution used to model the topic-word distribution. It controls the 
# sparsity of topics by influencing how likely a given word is to be assigned to a particular topic.

# A higher value of beta encourages topics to have a more uniform distribution over words, resulting in more 
# general and diverse topics. Conversely, a lower value of beta promotes sparser topics with fewer dominant words.

# The choice of beta can impact the interpretability and granularity of the discovered topics in LDA.
beta_values = np.arange(0.01, 1, 0.3).tolist()


#%%

gamma_threshold_values = np.arange(0.001, 0.011, 0.001).tolist()


#%%

# Define your dataset as a list of a list of tokenized sentences or load data from a file
def get_texts_out(year):
    year = int(year)
    with open(f"C:/_harvester/data/tokenized-sentences/10s/{year}-tokenized_sents-w-bigrams.pkl", "rb") as fp:
        texts_out = pickle.load(fp)

    return texts_out

#pp.pprint(get_texts_out(2010))


#%%

from typing import List, Optional
def coherence_score(X: List[List[str]], topics: List[int], metric: str = 'c_v', vectorizer: Optional[str] = None, glove: Optional[GloVe] = None) -> float:
    """
    Compute the coherence score for a given set of topics and documents.

    Args:
        X (list): List of documents.
        topics (list): List of topic assignments for each document.
        metric (str, optional): Coherence metric to use. Defaults to 'c_v'.
        vectorizer (str, optional): Vectorizer to use. Defaults to None.

    Returns:
        float: Coherence score.

    """
    if vectorizer == 'glove':
        # Load pre-trained GloVe embeddings
        # load the scattered embedding vectors from across Dask workers
        #glove = GloVe(vectors=embedding_vectors)

        # Move the embeddings to the GPU device
        #glove.vectors = glove.vectors.to(device)

        # Convert X to a list of documents
        documents = [list(doc) for doc in X]

        # Convert documents into numerical representations using GloVe
        document_vectors = []
        
        for doc in documents:
            doc_vector = [glove[word] for word in doc]
            document_vectors.append(doc_vector)
        
        X = document_vectors

    # Create a dictionary and corpus from the documents
    dictionary = Dictionary(X)
    corpus = [dictionary.doc2bow(doc) for doc in X]

    # Create a topic model using the given topics
    topic_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=len(set(topics)), random_state=42)

    # Compute the coherence score using the CoherenceModel
    coherence_model = CoherenceModel(model=topic_model, texts=X, dictionary=dictionary, coherence=metric)

    return coherence_model.get_coherence()


#%%


if __name__=="__main__":
    # Create a multiprocessing context using the "spawn" method
    # This method is recommended for certain platforms, such as Windows or Jupyter Notebook, to avoid conflicts
    ctx = multiprocessing.get_context("spawn")

    # Create a Pool of worker processes using the multiprocessing context
    # The number of worker processes is cores - 1
    # This ensures that one CPU core is left available for other tasks or system operations
    pool = ctx.Pool(cores - 1)

    # check to see if GloVe embeddings are saved
    #def is_folder_empty(folder_path):
    #    return len(os.listdir(folder_path)) == 0

    # initialize Dask cluster to None
    cluster = None

    # flag variable to control how embedding is scattered across Dask
    DASK_RUNNING = False

    # Load the saved embedding vectors
    # https://nlp.stanford.edu/projects/glove/
    import torchtext.vocab as vocab
    glove = vocab.Vectors('glove.840B.300d.txt', 'C:/_harvester/GloVe/')
    # Get the embedding vectors and vocabulary
    embedding_vectors_ = glove.vectors
    #vocabulary = glove.stoi
    # Move the embeddings to the GPU device
    glove.vectors = glove.vectors.to(device)

    # Verify if CUDA is being used by checking the device type
    if glove.vectors.device.type == "cuda":
        print("CUDA is being used by GloVe.")
    else:
        print("CUDA is not being used by GloVe. Using CPU instead.")

    # Specify the local directory path
    DASK_DIR = '/_harvester/tmp-dask-out'

    # specify Dask dashboard port
    #DASHBOARD_PORT = "60481"
    """
    # Set the GPU memory limit
    gpu_memory_limit = "10GB"
    # Set the CUDA_VISIBLE_DEVICES environment variable to specify which GPUs to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Specify GPU device IDs
    # Create a Dask local cluster with the specified local directory and GPU memory limit
    #cluster = LocalCluster(local_directory=DASK_DIR, device_memory_limit=gpu_memory_limit)
    cluster = LocalCluster(local_directory=DASK_DIR)
    client = Client(cluster)
    """
    # Deploy a Single-Machine Multi-GPU Cluster
    # https://medium.com/@aryan.gupta18/end-to-end-recommender-systems-with-merlin-part-1-89fabe2fa05b
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device IDs
    protocol = "tcp"  # "tcp" or "ucx"
    num_gpus = 1
    NUM_GPUS=[0]
    cores = multiprocessing.cpu_count() - 1 # Count the number of cores in a computer
    visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Select devices to place workers
    device_limit_frac = 0.7  # Spill GPU-Worker memory to host at this limit.
    device_pool_frac = 0.8
    part_mem_frac = 0.15

    # Manually specify the total device memory size (in bytes)
    device_size = 10 * 1024 * 1024 * 1024  # GPU has 12GB but setting at 10GB
            
    ram_memory_limit = "75GB" # Set the RAM memory limit (per worker)
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)

    if cluster is None:
        cluster = LocalCluster(
            n_workers=cores,
            threads_per_worker=1,
            #processes=False,
            memory_limit=ram_memory_limit,
            local_directory=DASK_DIR,
            dashboard_address=":8787",
            protocol="tcp",
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

    # verify that Dask is being used in your code, you can check the following:
    # Check if the Dask client is connected to a scheduler:
    if client.status == "running":
        print("Dask client is connected to a scheduler.")
        # Scatter the embedding vectors across Dask workers
        glove = client.scatter(embedding_vectors_)
        print("Embedded vectors are scattered across Dask workers.")
    else:
        print("Dask client is not connected to a scheduler.")

    # Check if Dask workers are running:
    if len(client.scheduler_info()["workers"]) > 0:
        print("Dask workers are running.")
    else:
        print("No Dask workers are running.")

    # dictionary to hold the metrics that are generated
    metrics_csv = {
            'n_topics': [],
            'alpha': [],
            'beta': [],
            'gamma': [],
            'median_cv': [],
            'convergence_score': [],
            'log_perplexity': [],
            'time_to_complete': []
            }
    
    for n_topics in range(start_topics, end_topics + 1, step_size):
        ###########################
        #   MODEL WITH GENSIM     #
        ###########################
        for alpha, beta, gamma_threshold in itertools.product(alpha_values, beta_values, gamma_threshold_values):
            
            # initialize timer. this is used to output to the metrics_csv dictionary.
            started = time()

            dictionary = Dictionary(get_texts_out(2010))
            corpus10 = [dictionary.doc2bow(doc) for doc in get_texts_out(2010)]

            passes = 11  # Number of passes
            total_iterations = passes * int(len(corpus10) / cores + 1)

            with tqdm(total=total_iterations, desc=f"Training LDA model with {n_topics} topics.") as pbar:
                lda_model_gensim = LdaMulticore(corpus=corpus10, 
                                                id2word=dictionary, 
                                                num_topics=n_topics,
                                                alpha=alpha, 
                                                eta=beta, 
                                                random_state=75,
                                                passes=passes,
                                                workers=cores,
                                                chunksize=int(len(corpus10)/cores+1),
                                                gamma_threshold=gamma_threshold, 
                                                per_word_topics=True)
                
                for i in range(passes):
                    lda_model_gensim.update(corpus10)
                    pbar.update(int(len(corpus10) / cores + 1))

            def get_texts(year):
                # Define your implementation for getting texts out based on year
                # Load data from disk or any other source as needed
                
                # Example implementation assuming you have a pickle file:
                filename = f"C:/_harvester/data/tokenized-sentences/10s/{year}-tokenized_sents-w-bigrams.pkl"
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                return list(itertools.chain.from_iterable(data))

            def process_year(year):
                def process_sentence(sentence):
                    # Convert tokens to numerical representations (word embeddings)
                    numerical_representations = [glove[token] for token in sentence]

                    return numerical_representations

                corpus_data_year = get_texts(year)  # Get tokenized sentences for one year

                processed_sentences_results = (process_sentence(sentence) for sentence in corpus_data_year)

                # Compute the results asynchronously
                for _ in processed_sentences_results:
                    pass

                corpus = [dictionary.doc2bow(doc) for doc in processed_sentences_results]

                return corpus  # Return the generated corpus


            # Call process_year function for each year from 2010 to 2019
            # https://docs.dask.org/en/latest/delayed-best-practices.html
            years = range(2010, 2020)
            # Create a progress bar using tqdm
            #with tqdm(total=len(years), desc="Vectorizing sentences into numerical representations.") as pbar:
            results = []
            counter = 0
            #    for year in years:
            #        result = dask.delayed(process_year)(year)
            #        results.append(result)
            #        pbar.set_postfix({"Year": year, "Status": "Processing"})
            #        pbar.update(1)
            #for year in years:
            #    result = dask.delayed(process_year)(year)
            #    results.append(result)
            #    counter += 1
            #    print(f"Vectorizing sentences for year {year} ({counter}/{len(years)})")
            for year in years:
                result = dask.delayed(process_year)(year)
                results.append(result)

            # Compute delayed computations and store results in corpus_data_list
            with ProgressBar():
                corpus_data_list = dask.compute(*results)

            # Pause and wait for computations to complete
            #client.gather(corpus_data_list)

            # Combine individual corpora into one corpus
            corpus = [doc for sublist in corpus_data_list for doc in sublist]

            #pp.pprint(f"Gensim LDA model for {n_topics} topics completed in {round((time() - started) / 60, 2)} minutes.")

            #convergence_list = list()
            convergence_score = lda_model_gensim.bound(corpus)

            #perplexity_list = list()
            perplexity_score = lda_model_gensim.log_perplexity(corpus)

            # Get topic-word distributions from trained Gensim LDA model
            topic_word_distributions_gensim = lda_model_gensim.get_topics()
        
            #c_v_score_gensim = 0
            c_v_scores = []
            pbar_coherence = tqdm(total=len(corpus), desc="Calculating Coherence Value - Gensim")
            
            whole_dict = gensim.corpora.Dictionary(corpus)
            for doc in corpus:
                bow = dictionary.doc2bow(doc)
                c_v_scores.append(coherence_score(X=corpus, topics=lda_model_gensim.get_document_topics(bow), 
                                                  vectorizer='glove', glove=glove))
                pbar_coherence.update(1)
                    
            pbar_coherence.close()
                
            #c_v_score_gensim /= len(corpus)
            c_v_score_gensim = np.median(c_v_scores)

            #print(f"Median Coherence Value (c_v) - Gensim: {c_v_score_gensim}")

            # Save the best Gensim LDA model
            best_model_gensim_filename = os.path.join(model_dir, f"best_model_gensim_{n_topics}_topics.model")
            lda_model_gensim.save(best_model_gensim_filename)

            # Generate and save a visualization for the best Gensim LDA model
            #vis_data = pyLDAvis.gensim.prepare(lda_model_gensim, corpus, whole_dict)
            #vis_html_filename = os.path.join(image_dir, f"lda_visualization_{n_topics}_topics.html")
            #pyLDAvis.save_html(vis_data, vis_html_filename)

            # calculate time to complete this run
            time_to_complete = round((time() - started) / 60, 2)

            # add metrics to dictionary
            metrics_csv['n_topics'].append(n_topics)
            metrics_csv['alpha'].append(alpha)
            metrics_csv['beta'].append(beta)
            metrics_csv['gamma'].append(gamma_threshold)
            metrics_csv['median_cv'].append(c_v_score_gensim)
            metrics_csv['convergence_score'].append(convergence_score)
            metrics_csv['log_perplexity'].append(perplexity_score)
            metrics_csv['time_to_complete'].append(time_to_complete)

            # Log metrics to a file
            log_filename_txt = os.path.join(log_dir, "lda_metrics.txt")

            with open(log_filename_txt, 'a') as log_file:
                log_file.write(f"Number of Topics: {n_topics}  |  ")
                log_file.write(f"Alpha: {alpha}  |  ")
                log_file.write(f"Beta: {beta}  |  ")
                log_file.write(f"Gamma Threshold: {gamma_threshold}  |  ")
                log_file.write(f"Median Coherence Value (c_v) - Gensim: {c_v_score_gensim}  |  ")
                log_file.write(f"Convergence Score - Gensim: {convergence_score}  |  ")
                log_file.write(f"Log Perplexity - Gensim: {perplexity_score}  |  ")
                log_file.write(f"Time to Complete: {time_to_complete}\n")
            

    pd.DataFrame(metrics_csv).to_pickle('C:/_harvester/data/lda-models/lda-pytorch-2010s/2010s-lda_tuning_results.pkl')
    pd.DataFrame(metrics_csv).to_csv('C:/_harvester/data/lda-models/lda-pytorch-2010s/2010s-lda_tuning_results.csv', index=False)   

    # Close the Dask client and cluster when done
    client.close()
    cluster.close(timeout=60)
    cluster = None
#%%