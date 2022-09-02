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
                  , s = 50)
    #for j, label in enumerate(annotations):
      #plt.annotate(label, (data[i][1][j], data[i][2][j]))
  return fig


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


if __name__ == '__main__':
