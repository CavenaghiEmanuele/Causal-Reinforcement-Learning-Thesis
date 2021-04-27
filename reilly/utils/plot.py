from networkx.generators.triads import TRIAD_EDGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def plot_medical_treatment_results(data: pd.DataFrame, title: str):
    data = data.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    data = data.groupby(['test', 'agent']).mean()
    tests = data.columns.values.tolist()
    data = data.groupby('agent')
    for test in tests:
        agents = data[test].apply(list).to_dict()
        plt.figure(title)
        unique_legend_causal = True
        unique_legend = True
        for agent in agents:
            label = re.search('Params: ([A-Z.,a-z]*)', agent).group(1)
            if 'Random' in label:
                plt.plot(agents[agent], 'k', label=label, linewidth=4)
            elif 'CausalQLearning' in label:
                if unique_legend_causal:
                    plt.plot(agents[agent], 'b', label=label)
                else:
                    plt.plot(agents[agent], 'b')
                unique_legend_causal = False
            else:
                if unique_legend:
                    plt.plot(agents[agent], 'r', label=label)
                else:
                    plt.plot(agents[agent], 'r')
                unique_legend = False

    #plt.title(title, fontsize=30)

    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.ylabel('Cumulative Reward', fontsize=26)
    plt.ylim([42.5, 92.5])
    plt.yticks(np.arange(42.5, 92.5+5, 5.0))
    
    plt.xlabel('Number of tests', fontsize=26)
    plt.xlim([0, 9])
    plt.xticks(np.arange(0, 9+1, 1.0))
    
    plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)
    plt.legend(loc='upper left', fancybox=True, shadow=True, prop={'size': 28})
    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10)
       
    plt.show()


def plot_mean(data: pd.DataFrame, n:int):
    data = data.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    data = data.groupby(['test', 'agent']).mean()
    tests = data.columns.values.tolist()
    data = data.groupby('agent')
    for test in tests:
        agents = data[test].apply(list).to_dict()
        plt.figure(test)
        for agent in agents:
            if not 'Random' in agent:
                for i in range(len(agents[agent])):
                    agents[agent][i] /= n
            plt.plot(agents[agent], label=agent)
        plt.ylabel(test)
        plt.xlabel('Number of tests')
        plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)
    plt.legend(loc='upper left')
    plt.show()


def plot_taxi_results(data: pd.DataFrame, title: str):
    data = data.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    data = data.groupby(['test', 'agent']).mean()
    tests = data.columns.values.tolist()
    data = data.groupby('agent')
    for test in tests:
        agents = data[test].apply(list).to_dict()
        plt.figure(test)
        unique_legend_causal = True
        unique_legend = True
        for agent in agents:
            label = re.search('Params: ([A-Z.,a-z]*)', agent).group(1)
            if 'Random' in label:
                plt.plot(agents[agent], 'k', label=label, linewidth=4)
            elif 'CausalQLearning' in label:
                if unique_legend_causal:
                    plt.plot(agents[agent], 'b', label=label)
                else:
                    plt.plot(agents[agent], 'b')
                unique_legend_causal = False
            else:
                if unique_legend:
                    plt.plot(agents[agent], 'r', label=label)
                else:
                    plt.plot(agents[agent], 'r')
                unique_legend = False

        plt.tick_params(axis='both', which='major', labelsize=24)

        plt.xlabel('Number of tests', fontsize=26)
        plt.xlim([0, 20])
        plt.xticks(np.arange(0, 20+1, 1.0))

        plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)
        plt.legend(loc='upper left', fancybox=True, shadow=True, prop={'size': 28})
        plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10)

        if test == 'wins':
            plt.ylabel('Wins', fontsize=26)
            plt.ylim([0.0, 1.0])
            plt.yticks(np.arange(0-0.1, 1+0.1, 0.05))
        else:
            plt.ylabel('Cumulative Reward', fontsize=26)
            plt.ylim([-2000, 20])
            plt.yticks(np.arange(-2000, 20, 100.0))
        

       
    plt.show()
