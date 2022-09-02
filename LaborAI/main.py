
import gensim
from collections import defaultdict
from gensim import corpora
from ekphrasis.classes.segmenter import Segmenter
from gensim.models import Phrases
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.util import ngrams
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from transformers import BertTokenizer, BertModel
import torch
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import math
from scipy.spatial.distance import cosine
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_mutual_info_score

#Only used if hashtags need to be exported for analytical usages
def export_hashtags():
  a = pd.Series([item for sublist in dataset.hashtags for item in sublist])
  df = a.value_counts().sort_index().rename_axis('hashtag').reset_index(name='times')
  df = df.nlargest(3000, "times")
  df = df.reset_index()
  df = df[["hashtag", "times"]]
  df.to_json("hashtags_counts.json")
  files.download("hashtags_counts.json")


def preprocess():
    ds = dataset[dataset["quoted_tweet_id"].isnull()]  # 198954 left

    ds = ds.tail(5000)
    n = len(ds) % 16
    #ds = ds.drop(ds.tail(n).index, inplace = True)
    ds = ds.iloc[:-n]
    ds = ds.reset_index()

    ds = ds.drop(
        columns=["index", "date", "quoted_tweet_id", "user_id", "id", "reply_count", "quote_count", "retweet_count",
                 "in_reply_to_tweet_id", "like_count"])
    ds = ds.replace({"text": {"\n": " ", "ö": "oe", "ü": "ue", "ä": "ae", "ß": "ss", "Ö": "Oe", "Ä": "Ae", "Ü": "Ue"}})
    ds["text2"] = ds["text"].apply(
        lambda x: " ".join(re.sub("(@[A-Za-z0-9_]+)|(#[A-Za-z0-9_]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))
    ds["text2"] = ds["text2"].apply(lambda x: " ".join([word for word in x.split() if word not in (stop)]))
    ds["text2"] = ds["text2"].apply(lambda x: " ".join([word.lower() for word in x.split()]))
    ds["text3"] = ds["text2"]

    from ekphrasis.classes.preprocessor import TextPreProcessor
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.dicts.emoticons import emoticons

    text_processor = TextPreProcessor(
        normalize=[],
        # terms that will be annotated
        annotate={},
        fix_html=False,  # fix HTML tokens

        # corpus from which the word statistics are going to be used for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used for spell correction
        corrector="twitter",

        unpack_hashtags=False,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        # dicts=[emoticons]
    )

    ds["text3"] = ds["text3"].apply(lambda x: " ".join(text_processor.pre_process_doc(x)))

    # segmenter using the word statistics from Twitter
    seg_tw = Segmenter(corpus="twitter")

    ds["text3"] = ds["text3"].apply(lambda x: " ".join([seg_tw.segment(word) for word in x.split()]))

    nltk.download('omw-1.4')
    # Tokenize and remove single letters
    ds["gensimtext"] = ds["text3"].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    ds["gensimtext"] = ds["gensimtext"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

    ds = ds.drop(columns=["text", "text2", "text3"])

    return ds


def tokenize_bert():
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    # logging.basicConfig(level=logging.INFO)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []
    input_ids_h = []
    attention_masks_h = []

    # For every sentence...
    for i in ds.index:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        sent = ds["gensimtext"][i]
        hasht = ds["hashtags"][i]
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        encoded_dict_h = tokenizer.encode_plus(
            hasht,  # Hashtags to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids_h.append(encoded_dict_h['input_ids'])
        attention_masks_h.append(encoded_dict_h['attention_mask'])

    # Convert the lists into tensors.
    tokens_tens = torch.cat(input_ids, dim=0)
    segments_tens = torch.cat(attention_masks, dim=0)

    tokens_tens_h = torch.cat(input_ids_h, dim=0)
    segments_tens_h = torch.cat(attention_masks_h, dim=0)
    return tokens_tens, segments_tens, tokens_tens_h, segments_tens_h


def predict(dataloader, predictions):
    pca = PCA(n_components=16)
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
            # hidden_states = outputs[2]
        principalComponents = []
        #print(len(outputs), len(outputs[0]), len(outputs[0][0]), len(outputs[0][0][0]))
        for i in range(0, len(outputs[0])):
            principalComponents.append(pca.fit_transform(outputs[0][i]))

        logits = torch.tensor(principalComponents)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # hidden_states = hidden_states.detach().cpu().numpy()

        # Store predictions and true labels
        predictions.append(torch.tensor(logits))
        # predictions.append(hidden_states)

    return predictions


def create_wordconversion():
    res = []
    wordconv = {}
    i = 0
    for sent in ds.tokens:
        for x in sent:
            if x not in wordconv:
                value = wordconv[x] = i
                i += 1
            else:
                value = wordconv[x]
            res.append(value)
    return wordconv


def create_w_w_matrice():
    worddict = {}
    words = {}
    wwsum = 0

    for isent in range(0, len(ds.tokens)):
        sent = ds.tokens[isent]
        #max = 31 if len(ds.tokens[isent]) > 31 else len(ds.tokens[isent])
        for i in range(0, len(ds.tokens[isent])):
            word = wordconversion[sent[i]]
            if worddict.get(word) is None:
                worddict[word] = {}
            if i - 1 >= 0:
                left = wordconversion[sent[i - 1]]
                wwsum += 1
                if worddict[word].get(left) is None:
                    worddict[word][left] = 1
                else:
                    worddict[word][left] += 1

            if i + 1 < len(sent):
                right = wordconversion[sent[i + 1]]
                wwsum += 1
                if worddict[word].get(right) is None:
                    worddict[word][right] = 1
                else:
                    worddict[word][right] += 1

    factor_ww = {}
    for key in worddict:
        for k in worddict[key]:
            if factor_ww.get(key) is None:
                factor_ww[key] = {}
            # factor_ww[key][k] = math.log(worddict[key][k] * wwsum / sum(worddict[key].values()) / sum(worddict[k].values()))
            factor_ww[key][k] = worddict[key][k] / wwsum

    return factor_ww, worddict


def create_hashconversion():
    res = []
    hashconv = {}
    i = 0
    for sent in ds.hashtags:
        for x in sent:
            if x not in hashconv:
                value = hashconv[x] = i
                i += 1
            else:
                value = hashconv[x]
            res.append(value)
    return hashconv


def create_h_h_matrice():
    hhsum = 0
    hashdict = {}

    for sent in ds.hashtags:
        #max = 15 if len(sent) > 15 else len(sent)
        for i in range(0, len(sent)):
            hasht = hashconversion[sent[i]]

            if hashdict.get(hasht) is None:
                hashdict[hasht] = {}
            #maxi = 15 if len(sent) > 15 else len(sent)
            for j in range(0, len(sent)):
                # hhsum += 1
                left = hashconversion[sent[j]]
                if hashdict[hasht].get(left) is None and left != hasht:
                    hashdict[hasht][left] = 1
                    hhsum += 1
                elif left != hasht:
                    hashdict[hasht][left] += 1
                    hhsum += 1

    factor_hh = {}

    for key in hashdict:
        for k in hashdict[key]:
            if factor_hh.get(key) is None:
                factor_hh[key] = {}
            # factor_hh[key][k] = math.log(hashdict[key][k] * wwsum / sum(hashdict[key].values()) * sum(hashdict[k].values()))
            factor_hh[key][k] = hashdict[key][k] / hhsum
    return factor_hh, hashdict


def create_h_t_matrice():
    hashtweetdict = {}
    htsum = 0

    for i in ds.index:
        #max = 15 if len(ds.hashtags[i]) > 15 else len(ds.hashtags[i])
        for j in range(0, len(ds.hashtags[i])):
            hasht = hashconversion[ds.hashtags[i][j]]
            if hashtweetdict.get(hasht) is None:
                hashtweetdict[hasht] = {}
            htsum += 1
            hashtweetdict[hasht][j] = 1

    factor_ht = {}

    for key in hashtweetdict:
        for k in hashtweetdict[key]:
            if factor_ht.get(key) is None:
                factor_ht[key] = {}
            factor_ht[key][k] = hashtweetdict[key][k] / htsum
    return factor_ht, hashtweetdict


def create_t_w_matrice():
    tweetworddict = {}
    twsum = 0

    for i in ds.index:
        tweetworddict[i] = {}
        #max = 31 if len(ds.tokens[i]) > 31 else len(ds.tokens[i])
        for j in range(0, len(ds.tokens[i])):
            word = wordconversion[ds.tokens[i][j]]
            if tweetworddict[i].get(word) is None:
                tweetworddict[i][word] = 1
            else:
                tweetworddict[i][word] += 1
            twsum += 1

    factor_tw = {}

    for key in tweetworddict:
        for k in tweetworddict[key]:
            if factor_tw.get(key) is None:
                factor_tw[key] = {}
            factor_tw[key][k] = tweetworddict[key][k] / twsum
    return factor_tw, tweetworddict


def writeindexandlabel(name, chosen, choice):
  with open(name + '_label.txt', 'w') as writefile:
    writefile.write(str(len(chosen)) + "\n")
    for i in chosen:
      writefile.write(str(i) + "\n")

  with open(name + '_index.txt', 'w') as writefile:
    writefile.write(str(len(chosen)) + "\n")
    for i in choice:
      writefile.write(str(i) + "\n")


def import_hashtags():
    hash_csv = pd.read_csv("top_hashtags_all_new.csv", sep=";",
                           names=["hashtag", "category", "category2", 1, 2, 3, 4, 5, 6, 7, 8], header=None)
    hash_csv = hash_csv.drop([1, 2, 3, 4, 5, 6, 7, 8], axis=1)
    hash_csv.hashtag = hash_csv.hashtag.apply(
        lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))
    import collections
    found_hashtags1 = {}
    found_hashtags2 = {}
    topics_hashtags1 = {}
    topics_hashtags2 = {}
    j = 1
    k = 1
    for i in hash_csv.index:
        if topics_hashtags1.get(hash_csv.category[i]) == None:
            topics_hashtags1[hash_csv.category[i]] = j
            j += 1
        if topics_hashtags2.get(hash_csv.category2[i]) == None:
            topics_hashtags2[hash_csv.category2[i]] = k
            k += 1
        if hashconversion.get(hash_csv.hashtag[i]) != None:
            found_hashtags1[hashconversion[hash_csv.hashtag[i]]] = topics_hashtags1.get(hash_csv.category[i])
            found_hashtags2[hashconversion[hash_csv.hashtag[i]]] = topics_hashtags2.get(hash_csv.category2[i])

    found_hashtags1 = collections.OrderedDict(sorted(found_hashtags1.items()))

    writeindexandlabel("hashtag", found_hashtags1.values(), found_hashtags1.keys())
    writeindexandlabel("hashtag2", found_hashtags2.values(), found_hashtags2.keys())
    return found_hashtags1, found_hashtags2, topics_hashtags1, topics_hashtags2


def bertembedding(tweetind, wordind, wordind2, mode):
  if mode == "tw":
    d = 1/(1 + math.exp(- cosine(sentence_embedding[tweetind], token_vecs_sum[tweetind][wordind])))
  elif mode == "ww":
    d = 1/(1 + math.exp(- cosine(token_vecs_sum[tweetind][wordind], token_vecs_sum[tweetind][wordind2])))
  elif mode == "ht":
    d = 1/(1 + math.exp(- cosine(token_vecs_sum_h[tweetind][wordind], sentence_embedding[tweetind])))
  else:
    d = 1/(1 + math.exp(- cosine(token_vecs_sum_h[tweetind][wordind], token_vecs_sum_h[tweetind][wordind2])))

  return d


def create_bert_matrices(worddict, hashdict, hashtweetdict, tweetworddict):
    tweetworddictbert = {}

    for i in ds.index:
        tweetworddictbert[i] = {}
        #max = 31 if len(ds.tokens[i]) > 31 else len(ds.tokens[i])
        for j in range(0, len(ds.tokens[i])):
            word = wordconversion[ds.tokens[i][j]]
            if tweetworddictbert[i].get(word) is None:
                tweetworddictbert[i][word] = bertembedding(i, j, 0, "tw") / tweetworddict[i][word]
            else:
                tweetworddictbert[i][word] = tweetworddictbert[i][word] + (
                            bertembedding(i, j, 0, "tw") / tweetworddict[i][word])

    wordworddictbert = {}
    for i in ds.index:
        #max = 31 if len(ds.tokens[i]) > 31 else len(ds.tokens[i])
        for j in range(0, len(ds.tokens[i])):
            word = wordconversion[ds.tokens[i][j]]
            if wordworddictbert.get(word) is None:
                wordworddictbert[word] = {}
            if j - 1 >= 0:
                left = wordconversion[ds.tokens[i][j - 1]]
                if wordworddictbert[word].get(left) is None:
                    wordworddictbert[word][left] = bertembedding(i, j, j - 1, "ww") / worddict[word][left]
                else:
                    wordworddictbert[word][left] = wordworddictbert[word][left] + (
                                bertembedding(i, j, j - 1, "ww") / worddict[word][left])
            if j + 1 < len(ds.tokens[i]):
                right = wordconversion[ds.tokens[i][j + 1]]
                if wordworddictbert[word].get(right) is None:
                    wordworddictbert[word][right] = bertembedding(i, j, j + 1, "ww") / worddict[word][right]
                else:
                    wordworddictbert[word][right] = wordworddictbert[word][right] + (
                                bertembedding(i, j, j + 1, "ww") / worddict[word][right])

    hashtweetdictbert = {}
    for i in ds.index:
        #max = 15 if len(ds.hashtags[i]) > 15 else len(ds.hashtags[i])
        for j in range(0, len(ds.hashtags[i])):
            hash = hashconversion[ds.hashtags[i][j]]
            if hashtweetdictbert.get(hash) is None:
                hashtweetdictbert[hash] = {}
            # htsum += 1
            hashtweetdictbert[hash][j] = bertembedding(i, j, 0, "ht")

    hashhashdictbert = {}
    for k in ds.index:
        sent = ds.hashtags[k]
        #max = 15 if len(ds.hashtags[k]) > 15 else len(ds.hashtags[k])
        for i in range(0, len(ds.hashtags[k])):
            hash = hashconversion[sent[i]]
            if hashhashdictbert.get(hash) is None:
                hashhashdictbert[hash] = {}
            #maxi = 15 if len(ds.hashtags[k]) > 15 else len(ds.hashtags[k])
            for j in range(0, len(ds.hashtags[k])):
                left = hashconversion[sent[j]]
                if hashhashdictbert[hash].get(left) is None and left != hash:
                    hashhashdictbert[hash][left] = bertembedding(k, i, j, "hh") / hashdict[hash][left]
                elif left != hash:
                    hashhashdictbert[hash][left] = hashhashdictbert[hash][left] + (
                                bertembedding(k, i, j, "hh") / hashdict[hash][left])

    return tweetworddictbert, wordworddictbert, hashtweetdictbert, hashhashdictbert


def writematrix(name, dictina, factor, a, b):
  with open(name, 'w') as writefile:
    writefile.write(str(a) + " " + str(b) + "\n")
    index = sorted(dictina)
    for key in index:
      line = str(key) + " " + str(len(dictina[key]))
      index2 = sorted(dictina[key])
      for keykey in index2:
        line += " " + str(keykey) + ":" + str(factor[key][keykey])
      writefile.write(line + "\n")


def writematrixbert(name, dictina, factor, a, b, bert):
  with open(name, 'w') as writefile:
    writefile.write(str(a) + " " + str(b) + "\n")
    index = sorted(dictina)
    for key in index:
      line = str(key) + " " + str(len(dictina[key]))
      index2 = sorted(dictina[key])
      for keykey in index2:
        line += " " + str(keykey) + ":" + str(bert[key][keykey])
      writefile.write(line + "\n")


def prep_data_pca(data):
    global chosen
    for column in data:
        data[column] = data[column].apply(lambda x: float(x))

    data = data.transpose()
    data[16] = chosen

    features = list(range(0, 16))
    # Separating out the features
    x = data.loc[:, features].values
    # Separating out the target
    y = data.loc[:, [16]].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=[1, 2])
    finalDf = pd.concat([principalDf, data[[16]]], axis=1)
    print(pca.explained_variance_ratio_)
    return finalDf


def prep_data_tsne(data):
    global chosen
    data = data.transpose()
    tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)

    x = pd.DataFrame(data=tsne, columns=[1, 2])
    x[16] = pd.DataFrame(chosen)
    finalDf = pd.concat([x, x[16]], axis=1)
    return x


def create_subplot(fig, data):
  global targets, colors, annotations
  for i in range(0, len(data)):
    ax = fig.add_subplot(len(data), 2, i + 1)
    if i%2==0:
      ax.set_title('2 component PCA', fontsize = 20)
    else:
      ax.set_title('2 component TSNE', fontsize = 20)
    for target, color in zip(targets, colors):
        indicesToKeep = data[i][16] == target
        ax.scatter(data[i].loc[indicesToKeep, 1]
                  , data[i].loc[indicesToKeep, 2]
                  , c = color
                  #, cmap = "tab20c"
                  , s = 20)
    #for j, label in enumerate(annotations):
      #plt.annotate(label, (data[i][1][j], data[i][2][j]))
  return fig


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def eval_sil(data, labels):
  return metrics.silhouette_score(data, labels, metric='euclidean')


def eval_ami(labels_pred, labels_true):
  return adjusted_mutual_info_score(labels_pred, labels_true)


def eval(data, start, end):
  sil = []
  ami = []
  for i in range(start, end):
    kmeans_model = KMeans(n_clusters=i, random_state=1).fit(data[[1, 2]])
    labels = kmeans_model.labels_
    sil.append(eval_sil(data[[1, 2]], labels))
    ami.append(eval_ami(data[16], labels))
  frame = [sil, ami, list(range(start, end))]
  frame = np.array(frame).transpose()
  ret = pd.DataFrame(frame, columns = ["sil", "ami","cluster"])
  return ret


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('stopwords')

    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words("english")

    dataset = pd.read_json("hashtags-en-tweets.jsonl.gz", compression="gzip", lines=True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    ds = preprocess()
    tokens_tensor, segments_tensors, tokens_tensor_h, segments_tensors_h = tokenize_bert()
    '''
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)

    tf.compat.v1.enable_eager_execution()

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Set the batch size.
    batch_size = 16

    # Create the DataLoader.
    prediction_data = TensorDataset(tokens_tensor, segments_tensors)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    prediction_data = TensorDataset(tokens_tensor_h, segments_tensors_h)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader_h = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Tracking variables
    predictions, predictions_h = [], []

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    predictions = predict(prediction_dataloader, predictions)
    predictions_h = predict(prediction_dataloader_h, predictions_h)

    print('    DONE.')

    token_embeddings = torch.stack(predictions, dim=0)
    predictions = 0
    token_embeddings_h = torch.stack(predictions_h, dim=0)
    predictions_h = 0
    token_vecs_sum = token_embeddings.reshape(-1, 64, 16)
    token_embeddings = 0
    token_vecs_sum_h = token_embeddings_h.reshape(-1, 32, 16)
    token_embeddings_h = 0

    # `hidden_states` has shape [13 x 1 x 32 x 768]
    sentence_embedding = []
    # `token_vecs` is a tensor with shape [32 x 768]
    for i in range(0, len(token_vecs_sum)):
        token_vecs = token_vecs_sum[i]
        # Calculate the average of all 32 token vectors.
        sentence = torch.mean(token_vecs, dim=0)
        sentence_embedding.append(sentence)

    print('Shape is: %d x %d' % (len(sentence_embedding), len(sentence_embedding[0])))
    '''
    '''
    Creating Matrices, factor matrices are co-occurence matrices
    '''
    ds.tokens = [sent.split() for sent in ds.gensimtext]
    wordconversion = create_wordconversion()
    hashconversion = create_hashconversion()

    factor_w_w, worddict = create_w_w_matrice()
    factor_h_h, hashdict = create_h_h_matrice()
    factor_h_t, hashtweetdict = create_h_t_matrice()
    factor_t_w, tweetworddict = create_t_w_matrice()
    #print(ds)
    '''
    Topics for tweets
    Maybe more annotated sentences will improve results
    '''
    tweet_choice = [0, 2, 4, 8, 10, 11, 13, 14, 17, 18, 21, 22, 24, 28, 33, 34, 35, 36, 44, 45, 48, 49, 50, 52, 53, 54,
                    55, 63, 64, 71, 74, 75, 82, 83, 85, 86, 87, 90, 91, 92, 93, 95, 96, 97, 99, 100, 101, 105, 106,
                    107, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124, 130, 137, 144, 145, 153, 154, 156]
    tweet_topics = [1, 1, 1, 10, 9, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 14, 2, 2, 11, 9, 9, 9, 9, 9, 2, 15,
                    15, 8, 9, 9, 10, 10, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 3, 3, 2, 3, 1, 4, 6, 6, 2, 6, 6, 6, 6, 7, 7,
                    4, 4, 4, 14, 13, 13, 16, 2, 7]

    writeindexandlabel("tweet", tweet_topics, tweet_choice)

    found_hashtags, found2_hashtags, topics_hashtags, topics2_hashtags = import_hashtags()
    #t_w_dictbert, w_w_dictbert, h_t_dictbert, h_h_dictbert = create_bert_matrices(worddict, hashdict, hashtweetdict,
                                                                                  #tweetworddict)
    '''
    In current mode only the bert embedding will be used for Matrice writing. 
    To use only embedding, use function writematrix()
    '''
    '''
    writematrixbert("M_t_w.txt", tweetworddict, factor_t_w, len(tweetworddict), len(worddict), t_w_dictbert)
    writematrixbert("M_w_w.txt", worddict, factor_w_w, len(worddict), len(worddict), w_w_dictbert)
    writematrixbert("M_h_t.txt", hashtweetdict, factor_h_t, len(hashtweetdict), len(ds.index),  h_t_dictbert)
    writematrixbert("M_h_h.txt", hashdict, factor_h_h, len(hashdict), len(hashdict), h_h_dictbert)
    '''
    writematrix("M_t_w.txt", tweetworddict, factor_t_w, len(tweetworddict), len(worddict))
    writematrix("M_w_w.txt", worddict, factor_w_w, len(worddict), len(worddict))
    writematrix("M_h_t.txt", hashtweetdict, factor_h_t, len(hashtweetdict), len(ds.index))
    writematrix("M_h_h.txt", hashdict, factor_h_h, len(hashdict), len(hashdict))

    '''
    Creating Plots
    '''
    name = input ("Include Data now! ")

    dat = pd.read_csv("Test/W_h_5_5_0.txt", sep=" ", header=None)
    dat2 = pd.read_csv("Test/W_h_5_10_0.txt", sep=" ", header=None)
    dat3 = pd.read_csv("Test/W_h_5_5_1.txt", sep=" ", header=None)
    dat4 = pd.read_csv("Test/W_h_5_10_1.txt", sep=" ", header=None)

    chosen = []

    for i in range(0, len(dat.columns)):
        if found_hashtags.get(i) == None:
            chosen.append(len(topics_hashtags) + 1)
        else:
            chosen.append(found_hashtags.get(i))

    annotations = list(range(0, 50))

    data = prep_data_pca(dat)
    data2 = prep_data_pca(dat2)
    data_tsne = prep_data_tsne(dat)
    data2_tsne = prep_data_tsne(dat2)
    fig = plt.figure(figsize=(16, 16))
    # targets = list(range(1, 5))
    #{'ukraine', 'society', 'sci_tech', 'middle_east', 'football'}
    targets = [topics_hashtags['ukraine'], topics_hashtags['society'], topics_hashtags['sci_tech'], topics_hashtags['middle_east'], topics_hashtags['football']]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'limegreen', 'grey', 'darkorange', 'fuchsia', 'khaki']

    fig = create_subplot(fig, [data, data_tsne, data2, data2_tsne])

    target_names = []
    for i in targets:
        a = get_keys_from_value(topics_hashtags, i)
        a = ' '.join(a)
        target_names.append(a)

    plt.legend(target_names)
    plt.show()



    #2 Plot

    chosen = []

    for i in range(0, len(dat3.columns)):
        if found2_hashtags.get(i) == None:
            chosen.append(len(topics2_hashtags) + 1)
        else:
            chosen.append(found2_hashtags.get(i))

    annotations = list(range(0, 50))

    data3 = prep_data_pca(dat3)
    data4 = prep_data_pca(dat4)
    data3_tsne = prep_data_tsne(dat3)
    data4_tsne = prep_data_tsne(dat4)
    fig2 = plt.figure(figsize=(16, 16))
    # targets = list(range(1, 5))
    # {'news_politics', 'sport', 'entertainment', 'culture', 'celebrity'}
    targets = [topics2_hashtags['news_politics'], topics2_hashtags['sport'], topics2_hashtags['entertainment'],
               topics2_hashtags['culture'], topics2_hashtags['celebrity']]

    fig2 = create_subplot(fig2, [data3, data3_tsne, data4, data4_tsne])

    target_names = []
    for i in targets:
        a = get_keys_from_value(topics2_hashtags, i)
        a = ' '.join(a)
        target_names.append(a)

    plt.legend(target_names)
    plt.show()

    x = eval(data, 2, 25)
    print(x)
    x = eval(data_tsne, 2, 25)
    print(x)
    x = eval(data4, 2, 25)
    print(x)
    x = eval(data4_tsne, 2, 25)
    print(x)