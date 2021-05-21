from datetime import date
from networkx.generators.triads import TRIAD_EDGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def plot(data: pd.DataFrame):
    data = data.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    data = data.groupby(['test', 'agent']).mean()
    tests = data.columns.values.tolist()
    data = data.groupby('agent')
    for test in tests:
        agents = data[test].apply(list).to_dict()
        plt.figure(test)
        for agent in agents:
            plt.plot(agents[agent], label=agent)
        plt.ylabel(test)
        plt.xlabel('Number of tests')
        plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)
    plt.legend(loc='upper left')
    plt.show()


def show_plot():
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)
    plt.legend(loc='upper left', fancybox=True, shadow=True, prop={'size': 28})
    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10)

    plt.xlabel('Number of tests', fontsize=26)
    plt.show()


def _medical_treatment(path: str):
    data = pd.read_csv(path)
    data[['ID','agent_name']] = data['agent'].str.split(',', 1, expand=True)
    data['agent_name'] = data['agent_name'].replace({
            'Params: Random.+': 'Random',
            'Params: CausalQLearning.+': 'Causal Q-Learning',
            'Params: QLearning.+': 'Q-Learning',
            },
            regex=True)
    data = data.drop(['agent', 'step'], axis=1).groupby(['test', 'sample', 'agent_name', 'ID']).sum()
    data = data.groupby(['test', 'agent_name', 'ID']).mean()
    
    # SET PLOT FOR MEDICAL TREATMENT ENVIRONMENT
    plt.ylabel('Cumulative Reward', fontsize=26)
    plt.ylim([40.0, 92.5])
    plt.yticks(np.arange(40.0, 92.5+5, 5.0))
    
    plt.xlim([0, 9])
    plt.xticks(np.arange(0, 9+1, 1.0))
    return data

def plot_medical_treatment(path: str):
    data = _medical_treatment(path)
    sns.lineplot(data=data, x="test", y='reward', hue="agent_name", units="ID", estimator=None)
    show_plot()

def plot_medical_treatment_mean(path: str):
    data = _medical_treatment(path)
    sns.lineplot(data=data, x="test", y='reward', hue="agent_name", ci="sd")
    show_plot()


def _taxi(path: str, drop_column: str):
    data = pd.read_csv(path)
    data[['ID','agent_name']] = data['agent'].str.split(',', 1, expand=True)
    data['agent_name'] = data['agent_name'].replace({
        'Params: Random.+': 'Random',
        'Params: CausalQLearning.+': 'Causal Q-Learning',
        'Params: QLearning.+': 'Q-Learning',
        },
        regex=True)
    data = data.drop([drop_column, 'agent', 'step'], axis=1).groupby(['test', 'sample', 'agent_name', 'ID']).sum()
    data = data.groupby(['test', 'agent_name', 'ID']).mean()
    # SET PLOT FOR TAXI ENVIRONMENT  
    plt.xlim([0, 19])
    plt.xticks(np.arange(0, 19, 1.0))
    return data

def plot_taxi(path: str):
    ########### PLOT WINS ###########
    data = _taxi(path, drop_column='reward')
    sns.lineplot(data=data, x="test", y='wins', hue="agent_name", units="ID", estimator=None)
    plt.ylabel('Wins', fontsize=26)
    plt.ylim([0, 1])
    plt.yticks(np.arange(-0.1, 1.1, 0.1))
    show_plot()
    ########### PLOT REWARD ###########
    data = _taxi(path, drop_column='wins')
    sns.lineplot(data=data, x="test", y='reward', hue="agent_name", units="ID", estimator=None)
    plt.ylabel('Cumulative Reward', fontsize=26)
    plt.ylim([-2000, 100])
    plt.yticks(np.arange(-2000, 101, 200))
    show_plot()

def plot_taxi_mean(path: str):
    ########### PLOT WINS ###########
    data = _taxi(path, drop_column='reward')
    sns.lineplot(data=data, x="test", y='wins', hue="agent_name", ci="sd")
    plt.ylabel('Wins', fontsize=26)
    plt.ylim([0, 1])
    plt.yticks(np.arange(-0.1, 1.1, 0.1))
    show_plot()
    ########### PLOT REWARD ###########
    data = _taxi(path, drop_column='wins')
    sns.lineplot(data=data, x="test", y='reward', hue="agent_name", ci="sd")
    plt.ylabel('Cumulative Reward', fontsize=26)
    plt.ylim([-2000, 100])
    plt.yticks(np.arange(-2000, 101, 200))
    show_plot()
