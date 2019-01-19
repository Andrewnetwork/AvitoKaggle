def wv(x, embedSize):
    if x in model.wv:
        return model.wv[x]
    else:
        return np.zeros(embedSize)


def pad(x, maxLen, embedSize):
    nPad = maxLen - len(x)
    for i in range(nPad):
        x.append(np.zeros(embedSize))
    return x

embeddingSize = 128
contextWindowSize = 5
maxLen = 20
model = Word2Vec(trainDF["description"].fillna("").map(lambda x: x.split(" ")), size=embeddingSize, window=contextWindowSize, min_count=5, workers=4)
train_desc = []
for desc in trainDF["description"].fillna(""):
    res = desc.split()[0:maxLen]
    out = pad(list(map(lambda x: wv(x, embeddingSize), res)), maxLen, embeddingSize)
    train_desc.append(out)

test_desc = []
for desc in testDF["description"].fillna(""):
    res = desc.split()[0:maxLen]
    out = pad(list(map(lambda x: wv(x, embeddingSize), res)), maxLen, embeddingSize)
    test_desc.append(out)

train_desc = np.array(train_desc).reshape(len(train_desc),maxLen*embeddingSize)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(train_desc)

test_desc = np.array(test_desc).reshape(len(test_desc),maxLen*embeddingSize)
distances, indices = nbrs.kneighbors(test_desc)

probs = []
for idxs in indices:
    probs.append(trainDF.iloc[idxs[1]]["deal_probability"])

data_to_submit = pd.DataFrame.from_items([
    ('item_id', testDF["item_id"]),
    ('deal_probability', pd.Series(probs))])

data_to_submit.to_csv('csv_to_submit.csv', index=False)

cnt = 0
res = 0
for idxs in indices:
    res += trainDF.iloc[idxs[1]]["category_name"] == testDF.iloc[idxs[0]]["category_name"]
    cnt += 1

print(res / cnt)

idx =9
print(trainDF["description"].iloc[indices[idx][1]])
print("\n----\n")
print(testDF["description"].iloc[indices[idx][0]])

for key in tokenizer.word_index:
    if key in ru_model:
        tokenizer.word_index[key] = ru_model[key]
    else:
        tokenizer.word_index[key] = np.zeros(emeddingDim)

 idx = 0
        for idxName in sampleFrame.index:
            #hf.create_dataset(str(idxName), data=outFeatures[idx])
            dataOut[dataIdx] = outFeatures[idx]
            dataIdx +=1
            idx += 1