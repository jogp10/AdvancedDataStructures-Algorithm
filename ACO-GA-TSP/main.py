# -*- coding: utf-8 -*-
import math
import sys
import argparse
import pandas
import csv

from plot import plot

parser = argparse.ArgumentParser(
    description='Parallelized and non-parallelized ACO algorithm.')
parser.add_argument('--parallel',
                    dest='is_parallel',
                    action='store_const',
                    const=True,
                    default=False,
                    help='Use parallelized algorithm\'s version')
parser.add_argument('--plot',
                    dest='plot',
                    action='store_const',
                    const=True,
                    default=False,
                    help='Plot the best path achieved')

args = parser.parse_args()

if args.is_parallel:
    print("Parallel!")
    from aco_parallel import ACO, Grafo
else:
    from aco import ACO, Grafo

def parse_cities(data_size):
    data_size += 1
    counter = 0
    matrix = [[0 for _ in range(data_size)] for _ in range(data_size)]

    with open('distances.csv', encoding="utf8") as file:
        read = csv.reader(file)
        next(read)
        src_id = 0
        dest_id = 0
        for row in read:
            for i in range(data_size + 1):
                if i == 0:
                    continue
                else:
                    matrix[src_id][dest_id] = float(row[i])
                dest_id += 1
            dest_id = 0
            src_id += 1
            counter += 1
            if counter == data_size:
                break

    return matrix


def main():
    matrix = parse_cities(50)
    aco = ACO(cont_formiga=100, geracoes=10, alfa=1.0, beta=10.0, ro=0.5, Q=10)
    grafo = Grafo(matrix, len(matrix))
    try:
        caminho, custo = aco.resolve(grafo)
        print('custo total: {}, caminho: {}'.format(custo, caminho))
    except TypeError:
        pass

if __name__ == '__main__':
    main()
