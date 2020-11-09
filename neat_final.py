import neat
import numpy as np
import sys, os, glob
import random
sys.path.insert(0, 'evoman') 
from environment import Environment
from neat_controller import player_controller
import time
import pickle
from math import fabs,sqrt
import multiprocessing
import csv

class generalist:
    def __init__(self,config='.',maxgen=50,popsize=50,fitness_cutoff=120,EA='neat',group=[2,4],subname='exp1'):

        self.maxgen = maxgen # cutoff number of generations
        self.fitness_cutoff = fitness_cutoff
        self.pop_size = popsize
        self.EA = EA
        self.group = group
        self.subname = subname
        self.config = config

        #LOAD AND  CHANGE CONFIGS
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,self.config)
        self.config.pop_size = self.pop_size
        self.config.fitness_threshold = self.fitness_cutoff

        #creating a folder for results
        self.experiment_name = 'EA' + self.EA + '/engroup%i%i%i%i'%(self.group[0],self.group[1],self.group[2],self.group[3])
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)
            
        # initializes simulation in multi evolution mode, for multiple static enemies.
        self.env = Environment(experiment_name=self.experiment_name,playermode="ai"
                               ,enemymode="static",player_controller=player_controller(),level=2,speed="fastest")

    def eval_genome(self,genome, config):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        totfitness=0
        minfitness=10000
        for enemy in self.group:
            self.env.update_parameter('enemies',[enemy])
            fitness,p,e,t = self.env.play(pcont=net)
            fitness = 0.7*(100-e) + 0.3*p - np.log(t)
            totfitness+=fitness

            if minfitness>fitness:
                minfitness=fitness

        return (totfitness/(2*len(self.group))+minfitness)

        
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)
    

    def my_save_func(self,delimiter=' ',filename='fitness_history.csv'):
        """ Write our own function that saves the 
        population's best, average and std of fitness per gen"""
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)
            best_fitness = [c.fitness for c in self.stats.most_fit_genomes]
            avg_fitness = self.stats.get_fitness_mean()
            std_fitness = self.stats.get_fitness_stdev()
            for best, avg, std in zip(best_fitness, avg_fitness,std_fitness):
                w.writerow([best, avg, std])

    def modify_config(self,modifications):
        self.config.max_stagnation = modifications['max_stagnation']
        self.config.activation_default = modifications['activation_default']
        self.config.initial_connection = modifications['initial_connection']
        self.config.num_hidden = modifications['num_hidden']
        print(modifications)

    def run(self):
        #creating population
        self.p = neat.Population(self.config)
        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(True))

        # Add reporter to collect stats
        self.stats = neat.StatisticsReporter()
        self.p.add_reporter(self.stats)
        
        #self.winner = self.p.run(self.eval_genomes,self.maxgen)

        #run NEAT
        self.p.run(self.eval_genomes,self.maxgen)

        # Write run statistics to CSV file
        self.my_save_func(delimiter=',', filename=self.experiment_name+'/'+ self.subname+ '_output.csv')

        # Display the winning genome
        self.winner = self.stats.best_genome()
        print('\nBest genome:\nfitness {!s}\n{!s}'.format(self.winner.fitness, self.winner))

        # Save winning genomes topology and weights to load for testing
        pickle.dump(self.winner, open(self.experiment_name+'/'+ self.subname+"_winner.p","wb"))

ea = 'neat_final' #outer folder
groups = [[1,2,5,8],[1,2,3,4]] 
group = groups[0]#ENTER WHICH group YOU ARE RUNNING


if __name__ == '__main__':
    local_dir = os.path.dirname(os.path.abspath("__file__"))
    config_path = os.path.join(local_dir, 'config_file.ini')

    #initialise parameter set
    parameters = {'max_stagnation':10,'activation_default':'random','initial_connection':'fs_neat_hidden',
                'num_hidden':15}

    #variable we're changing
    max_stags = [10,8,5]
    for m in max_stags:
        parameters['max_stagnation'] = m
        test = generalist(config=config_path,maxgen=5,popsize=20,EA=ea,group=group,subname='TRYY_%i'%m)
        test.modify_config(parameters)
        test.run()
    
    