import typing
import reilly as rl

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


def plot_results(data: pd.DataFrame):
    data = data.drop('step', axis=1).\
        groupby(['test', 'sample', 'agent']).sum()
    data = data.groupby(['test', 'agent']).mean()
    tests = data.columns.values.tolist()
    data = data.groupby('agent')
    for test in tests:
        agents = data[test].apply(list).to_dict()
        plt.figure(test)
        unique_legend = True
        for agent in agents:
            label = re.search('Params: (.*)', agent).group(1)
            if 'Random' in label:
                plt.plot(agents[agent], 'k', label=label)
            else:
                if unique_legend:
                    plt.plot(agents[agent], 'r', label=label)
                else:
                    plt.plot(agents[agent], 'r')
                unique_legend = False

        plt.ylabel(test)
        plt.xlabel('Number of tests')
        plt.grid(linestyle='--', linewidth=0.5, color='.25', zorder=-10)

    plt.legend(loc='upper left')
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
