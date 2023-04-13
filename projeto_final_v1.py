import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

from pycaret.classification import *

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline


def load_data():
    df = pd.read_csv("Data/kobe_dataset.csv")

    df.dropna(inplace=True)

    df = df[['lat', 'lon', 
                'minutes_remaining', 
                'period', 'playoffs', 
                'shot_distance', 'shot_made_flag', 'shot_type']]

    df.columns = df.columns.astype(str)  

    return df

def conform_data(df, shot_type):
    # Filtrando os dados onde o valor de shot_type for igual à 2PT Field Goal
    data_filtered = df[df['shot_type'] == shot_type]

    data_filtered.drop('shot_type',axis=1, inplace=True)

    data_filtered.columns = data_filtered.columns.astype(str)  

    return data_filtered

def train_test_spliting(data_filtered, test_size, random_state):
    # Dividindo os dados em treino e teste deve ser feita de maneira aleatória, independente e estratificada para garantir que ambos os conjuntos representem 
    # a mesma distribuição garantindo que as proporções de classes nas amostras de treino e teste sejam semelhantes às proporções de classes na população original.
    X_train, X_test, y_train, y_test = train_test_split(data_filtered.drop(['shot_made_flag'],axis=1), data_filtered['shot_made_flag'],
                                                         test_size=test_size, random_state=random_state, stratify = data_filtered['shot_made_flag'])

    # Juntando os dados de treino e teste
    data_train = pd.concat([X_train, y_train], axis=1)

    data_test = pd.concat([X_test, y_test], axis=1)
    
    # Salvando os dados processados
    data_train.columns = data_train.columns.astype(str)  

    data_test.columns = data_test.columns.astype(str) 

    return data_train, data_test

# # Inicialização do Mlflow 
# mlflow.set_tracking_uri("http://127.0.0.1:5000/#/models")
# # # Obtém o ID da run atual
mlflow.active_run()

# # # Exibe o ID da run
# print("ID da run: ", run_id)

# Definindo o nome do experimento 
mlflow.set_experiment("Kobe Bryant shot-selection - 2PT Experiment")

# Definindo os parametros da primeira rodada PreparacaoDados
random_state = 13
test_size = 0.2
shot_type = '2PT Field Goal'
target = 'shot_made_flag'

# Iniciando uma run do MlFlow para o pipeline de preparação de dados
with mlflow.start_run(run_name='PreparacaoDados_2PT'):
    
    # Carregando os dados
    df = load_data()
    data_filtered = conform_data(df, shot_type)

    data_filtered.to_parquet('Data/processed/data_filtered.parquet')
    mlflow.log_artifact('Data/processed/data_filtered.parquet')

    data_train, data_test = train_test_spliting(data_filtered, test_size, random_state)

    data_train.to_parquet('Data/operalization/base_train.parquet')
    mlflow.log_artifact('Data/operalization/base_train.parquet')

    data_test.to_parquet('Data/operalization/base_test.parquet')
    mlflow.log_artifact('Data/operalization/base_test.parquet')

    # Registrando os parâmetros do pipeline
    mlflow.log_metric('num_features', (data_test.shape[1]))
    mlflow.log_param('prop_test_size', test_size)
    mlflow.log_param('shot type', shot_type)
    
    # Registrando métricas do pipeline
    mlflow.log_metric('train_size', len(data_train))
    mlflow.log_metric('test_size', len(data_test))

mlflow.end_run()

    
# Iniciando uma run do MlFlow para o pipeline de treinamento do modelo de REGRESSÃO LOGÍSTICA
with mlflow.start_run(run_name='Treinamento_lr_2PT'):

    # Carregando os dados processados
    data_train = pd.read_parquet('Data/operalization/base_train.parquet')
    data_test = pd.read_parquet('Data/operalization/base_test.parquet')
    
    data_train[target] =data_train[target].astype(int)

    X_train = data_train.drop(target,axis=1)
    X_test = data_test.drop(target,axis=1)

    y_train = data_train.pop(target)
    y_test = data_test.pop(target)
    
    # Configurando o ambiente do PyCaret
    setup(data=X_train, target=y_train)
    
    # Criando o modelo de regressão logística com o PyCaret
    lr = create_model('lr')
    
    # Treinando o modelo com o conjunto de treino
    lr_fit = tune_model(lr)
    
    # Avaliando o modelo com o conjunto de teste    
    y_pred_lr = predict_model(lr_fit, data=X_test)['prediction_label']
    f1_score_lr = f1_score(y_test, y_pred_lr)
    
    # Calculando a função de custo log loss e registrando no Mlflow
    log_loss_value_lr = log_loss(y_test, y_pred_lr)

    # Salvando o modelo treinado
    save_model(lr, 'modelo_treinado_LR')
    mlflow.log_artifact('modelo_treinado_LR.pkl')

    # Registrando os parâmetros do modelo
    mlflow.set_tag('model', 'Logistic Regression')
    mlflow.set_tag('algorithm', 'PyCaret')
    mlflow.set_tag('random_state',random_state)

    mlflow.log_metric('num_features', (data_test.shape[1]))
    mlflow.log_param('prop_test_size', test_size)
    mlflow.log_param('shot type', shot_type)

    mlflow.log_metric('log_loss', log_loss_value_lr)
    mlflow.log_metric('f1_score', f1_score_lr)

mlflow.end_run()

# Iniciando uma run do MlFlow para o pipeline de treinamento do modelo de CLASSIFICAÇÃO
with mlflow.start_run(run_name='Treinamento_clf_2PT'):

    # Carregando os dados processados
    data_train = pd.read_parquet('Data/operalization/base_train.parquet')
    data_test = pd.read_parquet('Data/operalization/base_test.parquet')
    
    data_train[target] =data_train[target].astype(int)

    X_train = data_train.drop(target,axis=1)
    X_test = data_test.drop(target,axis=1)

    y_train = data_train.pop(target)
    y_test = data_test.pop(target)
    
    # Configurando o ambiente do PyCaret
    setup(data=X_train, target=y_train)
    
    # Treinando um modelo de classificação (árvore de decisão)
    clf = create_model('dt')
    clf.fit(X_train, y_train)
    
    # Avaliando o modelo de classificação com a métrica F1-score na base de teste
    y_pred_clf = clf.predict(X_test)
    f1_score_clf = f1_score(y_test, y_pred_clf)
    log_loss_value_clf = log_loss(y_test, y_pred_clf)

    # Registrando o modelo de classificação treinado
    save_model(clf, 'modelo_treinado_CLF')
    mlflow.log_artifact('modelo_treinado_CLF.pkl')

    # Registrando os parâmetros do modelo
    mlflow.set_tag('model', 'Classification DT')
    mlflow.set_tag('algorithm', 'PyCaret')
    mlflow.set_tag('random_state',random_state)

    mlflow.log_metric('num_features', (data_test.shape[1]))
    mlflow.log_param('prop_test_size', test_size)
    mlflow.log_param('shot type', shot_type)

    mlflow.log_metric('log_loss', log_loss_value_clf)
    mlflow.log_metric('f1_score', f1_score_clf)

mlflow.end_run()

mlflow.set_experiment("Kobe Bryant shot-selection - 3PT Experiment")

# Definindo os parametros da primeira rodada PreparacaoDados
random_state = 13
test_size = 0.2
second_shot_type = '3PT Field Goal'
target = 'shot_made_flag'


# # Iniciando uma run do MlFlow para o pipeline de preparação de dados
# with mlflow.start_run(run_name='PreparacaoDados_3PT'):
    
#     # Carregando os dados
#     df = load_data()
#     data_filtered = conform_data(df, second_shot_type)

#     data_filtered.to_parquet('Data/processed/data_filtered.parquet')
#     mlflow.log_artifact('Data/processed/data_filtered.parquet')

#     # data_train, data_test = train_test_spliting(data_filtered, test_size, random_state)

#     # data_train.to_parquet('Data/operalization/base_train.parquet')
#     # mlflow.log_artifact('Data/operalization/base_train.parquet')

#     # data_test.to_parquet('Data/operalization/base_test.parquet')
#     # mlflow.log_artifact('Data/operalization/base_test.parquet')

#     # # Registrando os parâmetros do pipeline
#     # mlflow.log_metric('num_features', (data_test.shape[1]))
#     # mlflow.log_param('prop_test_size', test_size)
#     mlflow.log_param('shot type', second_shot_type)

    
#     # # Registrando métricas do pipeline
#     # mlflow.log_metric('train_size', len(data_train))
#     # mlflow.log_metric('test_size', len(data_test))

# mlflow.end_run()



# Iniciando uma run do MlFlow para o pipeline de treinamento do modelo de CLASSIFICAÇÃO
with mlflow.start_run(run_name='Treinamento_clf_3PT'):

    # Carregando os dados
    df = load_data()
    data_filtered = conform_data(df, second_shot_type)

    mlflow.log_param('shot type', second_shot_type)
    
    # Aplicando o modelo treinado para os novos dados
    X = data_filtered.drop('shot_made_flag',axis=1)
    y = data_filtered['shot_made_flag']

    y_pred_clf = clf.predict(X)

    # Avaliando o modelo de classificação com a métrica F1-score na base de teste
    f1_score_clf = f1_score(y, y_pred_clf)
    log_loss_value_clf = log_loss(y, y_pred_clf)

    # Registrando o modelo de classificação treinado
    save_model(clf, 'modelo_treinado_CLF')
    mlflow.log_artifact('modelo_treinado_CLF.pkl')

    # Registrando os parâmetros do modelo
    mlflow.set_tag('model', 'Classification DT')
    mlflow.set_tag('algorithm', 'PyCaret')
    mlflow.set_tag('random_state',random_state)

    mlflow.log_metric('num_features', (data_filtered.drop('shot_made_flag',axis=1).shape[1]))
    mlflow.log_param('prop_test_size', test_size)
    mlflow.log_param('shot type', second_shot_type)

    mlflow.log_metric('log_loss', log_loss_value_clf)
    mlflow.log_metric('f1_score', f1_score_clf)


mlflow.end_run()

    
