import numpy as np 
import pandas as pd 
from gensim.models import KeyedVectors
from keras.applications.vgg19 import VGG19
from keras.models import Model
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
import h5py
from multiprocessing import Pool
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg19 import preprocess_input

# Feature Extraction Hyperparams - Images
imgShape         = (224, 224, 3)
featureLayerName = "fc1"  # We will extract the activations from the first fully connected layer.
batchSize        = 150  # Number of images we process at once.
nProcesses       = 15 # Number of threads we will use to process the data.

# Feature Extraction Hyperparams - Text
vocab_len    = 20000
title_maxlen = 50
desc_maxlen  = 1000
embed_size   = 300 # Set by the word2vec corpus, don't change. 

# Image Path
imagePath_Train = "./input/avito-demand-prediction/train_jpg/data/competition_files/train_jpg/"
imagePath_Test  = "./input/avito-demand-prediction/train_jpg/data/competition_files/test_jpg/"
imageExt        = ".jpg"

# Fasttext Word2Vec Path
russVecPath = 'input/fasttext-russian-2m/wiki.ru.vec'


def readAndReshape(img):
    return preprocess_input(resize(imread(img), imgShape))

def imageFeatures(df,vgg,fc1_features,threads,path,title):
    onlyImages = df["image"]

    hf = h5py.File('AvitoVGGFeatures_'+title+'.h5', 'w')
    nBatches = int(np.floor(trainDF.shape[0] / batchSize))

    dataIdx = 0
    dataOut = hf.create_dataset("vgg", data=np.zeros(shape=(onlyImages.shape[0],4096)))
    
    for batchIndex in range(nBatches):
        sampleFrame = onlyImages[batchIndex * batchSize:(batchIndex + 1) * batchSize]
        subBatchSize = sampleFrame.shape[0]

        imgURIs = sampleFrame.map(lambda x: path + x + imageExt).as_matrix()
        imgReads = np.hstack(threads.map(readAndReshape,imgURIs)).reshape((subBatchSize, imgShape[0], imgShape[1], imgShape[2]))
        outFeatures = fc1_features.predict(imgReads)

        dataOut[dataIdx:dataIdx+subBatchSize] = outFeatures
        dataIdx += subBatchSize

    hf.close()

def makeEmbeddingMatrix(ru_model,tokenizer):
    def getEmbedding(x):
        if x in ru_model:
            return ru_model[x]
        else:
            return None

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    for word, i in word_index.items():
        embedding_vector = getEmbedding(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    hf_emb = h5py.File('AvitoVecEmbeddingsMat.h5', 'w')
    hf_emb.create_dataset("embedding_matrix", data=embedding_matrix)

    hf_emb.close()

def textTokens(df,tokenizer,title):
    corpus  = df["description"]

    hf = h5py.File('AvitoDescTokens_'+title+'.h5', 'w')
    nBatches = int(np.floor(corpus.shape[0] / batchSize))

    dataIdx = 0
    dataOut = hf.create_dataset("tokens", data=np.zeros(shape=(corpus.shape[0],desc_maxlen)))

    for batchIndex in range(nBatches):
        sampleFrame = corpus[batchIndex * batchSize:(batchIndex + 1) * batchSize]
        subBatchSize = sampleFrame.shape[0]

        descTokens =  pad_sequences(tokenizer.texts_to_sequences(sampleFrame),maxlen = desc_maxlen )

        dataOut[dataIdx:dataIdx+subBatchSize] = descTokens
        dataIdx += subBatchSize
    
    hf.close()

if __name__ == '__main__':
    trainDF = pd.read_csv("input/avito-demand-prediction/train.csv")
    #testDF = pd.read_csv("input/avito-demand-prediction/test.csv")[0:100]

    # Corrupted images in the training set.
    badImgs = ["4f029e2a00e892aa2cac27d98b52ef8b13d91471f613c8d3c38e3f29d4da0b0c",
               "b98b291bd04c3d92165ca515e00468fd9756af9a8f1df42505deed1dcfb5d7ae",
               "60d310a42e87cdf799afcd89dc1b11ae3fdc3d0233747ec7ef78d82c87002e83",
               "8513a91e55670c709069b5f85e12a59095b802877715903abef16b7a6f306e58"]

    trainDF[trainDF["image"].isin(badImgs)] = np.nan
    trainDF = trainDF[(~trainDF["image"].isnull()) & (~trainDF["description"].isnull())][0:100000]

    # Feature Extraction Setup - Images
    vgg = VGG19(weights='imagenet', input_shape=imgShape, classes=1000)
    fc1_features = Model(vgg.input, vgg.get_layer(featureLayerName).output)

    threads = Pool(nProcesses)

    # Feature Extraction - Images
    imageFeatures(trainDF,vgg,fc1_features,threads,imagePath_Train,"Train")
    #imageFeatures(testDF,vgg,fc1_features,threads,imagePath_Test,"Test")

    # Feature Extraction Setup - Text
    #tokenizer = Tokenizer(num_words=vocab_len)
    #tokenizer.fit_on_texts(trainDF["description"]) 
    #ru_model = KeyedVectors.load_word2vec_format(russVecPath)

    # Feature Extraction - Text
    #textTokens(trainDF,tokenizer,"Train")
    #textTokens(testDF,tokenizer,"Test")
    #makeEmbeddingMatrix(ru_model,tokenizer)

