import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Título do dashboard
st.title('Monitoramento do Modelo com MLflow')

# Seção de experimento
st.header('Kobe Bryant shot-selection')

# Carrega experimento do MLflow
opcoes = ['Kobe Bryant shot-selection - 3PT Experiment','Kobe Bryant shot-selection - 2PT Experiment']

experiment_name = st.selectbox('Escolha uma opção:', opcoes)

st.write('Você escolheu:', experiment_name)

mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Recupera informações do experimento
runs = mlflow.search_runs(experiment_ids=[experiment_id])
runs['experiment_name'] = experiment_name

runs = runs[['experiment_name','run_id', 'start_time',
       'end_time','params.shot type', 'tags.mlflow.runName',
       'tags.algorithm', 'tags.model' ,
        'tags.random_state', 'metrics.num_features', 
       'params.prop_test_size',
        'metrics.f1_score',
       'metrics.log_loss']]

runs = runs[runs['tags.mlflow.runName'] != 'PreparacaoDados_2PT']
       
runs.columns =  ['experiment_name', 'id','start_time'
                                    ,  'end_time', 'shot_type', 
                                    'run_name','algorithm','model',
                                    'random_state','num_features',
                                    'prop_test_size',
                                    'f1_score',
                                    'log_loss']

runs = runs[runs['experiment_name'] == experiment_name]

run_ids = runs['id']

run_name = runs['run_name']

prop_test_size = runs['prop_test_size']
random_state = runs['random_state']

shot_type = runs['shot_type']
num_features = runs['num_features']


algorithm = runs['algorithm']
model = runs['model']

models = runs[['model','log_loss','f1_score']]

log_loss = runs['log_loss']
f1_score = runs['f1_score']

st.write('ID:', experiment_id)

# Seção de informações do modelo
st.header('Histórico de Runs')
# Seleciona o último run do experimento

last_run_id = run_ids.iloc[0]
last_run = mlflow.get_run(last_run_id)
# Exibe informações do modelo
st.write('Resumo:', runs)
# st.write('Versão:', model)
# st.write('Descrição:', algorithm)

# # Seção de métricas
# st.header('Métricas')
# # Exibe métricas de desempenho do modelo
# st.write('F1-Score:',f1_score )
# st.write('Log Loss:', log_loss)

list_models = models.model.unique()
for i in list_models:
    m = models[models['model'] == i]
    # Seção de gráficos
    st.write('Gráficos do modelo: ', i)
    # Gráfico de evolução do F1-Score
    fig, ax = plt.subplots()
    ax.plot(m['f1_score'])
    ax.set_xlabel('Iteração')
    ax.set_ylabel('F1-Score')
    st.pyplot(fig)

    # Gráfico de evolução do Log Loss
    fig, ax = plt.subplots()
    ax.plot(m['log_loss'])
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Log Loss')
    st.pyplot(fig)

st.header('Run mais recente')

last_run = runs[runs['id'] == last_run_id]
# Seção de parâmetros
st.header('Parâmetros')
# Exibe os parâmetros usados no último run
st.write(last_run['random_state'])
st.write(last_run['num_features'])
st.write(last_run['prop_test_size'])

# Seção de tags
st.header('Tags')
# Exibe as tags do último run
st.write(last_run['algorithm'])
st.write(last_run['model'])

# Seção de Resultados
st.header('Resultados')
# Exibe os Resultados do último run
st.write(last_run['f1_score'])
st.write(last_run['log_loss'])