from tensorflow import keras
import numpy as np
from keras import layers
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import shap

df = pd.read_csv('phoibledat.csv')
maindat = df
obs = np.asarray(df['ID'])
obs = obs.astype('str')
colnames = list(df.iloc[:, 2:39])
df = np.asarray(df.iloc[:, 2:39])

enc = OneHotEncoder()
enc.fit(obs.reshape(-1, 1))
obs = enc.transform(obs.reshape(-1, 1)).toarray()
dropout = .25
for run in range(100):
    randlist = np.random.randint(0, np.shape(obs)[0], size=int(np.round(np.shape(obs)[0] * 0.2)))

    df_val = df[randlist]
    df_train = df[[z for z in range(np.shape(obs)[0]) if not z in randlist]]

    input_enc = keras.Input(shape=(37,))

    encoded_x = layers.Dense(128, activation='relu')(input_enc)
    encoded_x = layers.Dropout(dropout)(encoded_x)
    encoded_x = layers.Dense(64, activation='relu')(encoded_x)
    encoded_x = layers.Dropout(dropout)(encoded_x)
    encoded_x = layers.Dense(32, activation='relu')(encoded_x)
    encoded_x = layers.Dropout(dropout)(encoded_x)
    encoded_x = layers.Dense(1, activation='relu', kernel_initializer='one',
                             bias_initializer='zero')(encoded_x)
    encoded = keras.Model(inputs=input_enc, outputs=encoded_x)

    encoded_x = layers.BatchNormalization()(encoded_x)

    decoded = layers.Dense(32, activation='relu')(encoded_x)
    decoded = layers.Dropout(dropout)(decoded)
    decoded = layers.Dense(64, activation='relu')(decoded)
    decoded = layers.Dropout(dropout)(decoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dropout(dropout)(decoded)

    decoded = layers.Dense(37, activation='relu')(decoded)

    autoencoder = keras.Model(inputs=encoded.input, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(df_train, df_train,
                    epochs=30,
                    batch_size=16,
                    validation_data=(df_val, df_val)
                    )
    preds = encoded.predict(df)
    hiddenpreds = pd.DataFrame(preds, columns=['estimate'])
    hiddenpreds['ID'] = [x+1 for x in range(len(preds))]
    hiddenpreds['true'] = np.sum(df, axis=1)

    shap.initjs()
    explainerModel = shap.DeepExplainer(encoded, df)
    shap_values_Model = explainerModel.shap_values(df)

    shap_v = pd.DataFrame(shap_values_Model[:, :, 0])
    feature_list = colnames
    shap_v.columns = feature_list
    df_v = pd.DataFrame(df, columns=colnames).copy().reset_index().drop('index', axis=1)

    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)


    weighted_shap = np.asarray(shap_v) / np.asarray(df_v)
    weighted_shap[weighted_shap == -np.inf] = np.nan
    weighted_shap[weighted_shap == np.inf] = np.nan

    if run == 0:
        raw_shap = pd.DataFrame(shap_values_Model[0].mean(0).reshape(1, -1), columns=colnames)
        res_df = pd.DataFrame(np.abs(shap_values_Model[0]).mean(0).reshape(1, -1), columns=colnames)
        weighted_shap_df = pd.DataFrame(np.nanmean(np.abs(weighted_shap), axis=0).reshape(1, -1), columns=colnames)
        hiddenpreds.to_csv(
            'newlatentpreds.csv', mode='a',
            index=False, header=False)
        corr_df = pd.DataFrame(np.asarray(corr_list).reshape(1, -1), columns=colnames)

    else:
        raw_shap_temp = pd.DataFrame(shap_values_Model[0].mean(0).reshape(1, -1), columns=colnames)
        raw_shap = pd.concat([raw_shap, raw_shap_temp], ignore_index=True)
        weighted_shap_df_temp = pd.DataFrame(np.nanmean(np.abs(weighted_shap), axis=0).reshape(1, -1), columns=colnames)
        weighted_shap_df = pd.concat([weighted_shap_df, weighted_shap_df_temp], ignore_index=True)
        res_df_temp = pd.DataFrame(np.abs(shap_values_Model[0]).mean(0).reshape(1, -1), columns=colnames)
        res_df = pd.concat([res_df, res_df_temp], ignore_index=True)

        hiddenpreds.to_csv(
            'latentpreds.csv', mode='a',
            index=False, header=False)
        corr_df = pd.concat((corr_df, pd.DataFrame(np.asarray(corr_list).reshape(1, -1), columns=colnames)))

res_df.to_csv('shap_dist.csv')
raw_shap.to_csv('raw_shap.csv')
corr_df.to_csv('corr_df.csv')
weighted_shap_df.to_csv('weighted_shap_dist.csv')
