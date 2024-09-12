#%%
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

with open(r"C:\_harvester\data\lda-models\2010s_html\visuals\wordcloud-09082024.txt",'r') as wordc:
    data = wordc.read()
    print(data)

# %%
plt.figure(figsize=(12,12))
wc = WordCloud().generate(data)
plt.axis("off")
plt.imshow(wc)
# %%
