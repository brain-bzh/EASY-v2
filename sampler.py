import torch
import numpy as np
import math
import random
import os
import json

class EpisodicGenerator():
    def __init__(self,
                 samples: list):
        self.samples = samples

        self.classes = list(set([s[1] for s in self.samples]))
        self.max_classes = len(self.classes)
        self.num_elements_per_class = [len([s for s in self.samples if s[1]==c]) for c in self.classes]


    def select_classes(self, ways: int = None):
        if ways is None:
            ways = random.randint(5, self.max_classes)
        else:
            assert ways <= self.max_classes, f'ways={ways} is greater than the number of classes in the dataset ({self.max_classes})'

        # get n_ways classes randomly
        picked_classes = random.sample(self.classes, ways)

        return picked_classes

    def get_query_size(self, picked_classes: list, n_queries:int = None):
        if n_queries is None:
            min_queries = 10
            query_size = min([int(0.5*self.num_elements_per_class[c]) for c in range(self.max_classes)])
            query_size = min(min_queries, query_size)
        else:
            query_size = n_queries
        return query_size

    def get_support_size(self, picked_classes, query_size, n_shots):
        # sample beta uniformly from (0,1]
        if n_shots == 0:
            beta = 0.
            while beta == 0.:
                beta = torch.rand(1).item()
            support_size = sum([math.ceil(beta*min(100, self.num_elements_per_class[c]-query_size)) for c in range(self.max_classes)])
            support_size = min(500, support_size)
        else:
            support_size = len(picked_classes)*n_shots
        return support_size

    def get_number_of_shots(self, picked_classes, support_size: int, query_size: int, n_shots: int = None):
        if n_shots is None:
            n_ways = len(picked_classes)
            alphas = torch.Tensor(np.random.rand(n_ways)*(np.log(2)-np.log(0.5))+np.log(0.5)) # sample uniformly between log(0.5) and log(2)
            proportions = torch.exp(alphas)*torch.cat([torch.Tensor([self.num_elements_per_class[c]]) for c in picked_classes])
            proportions /= proportions.sum() # make sum to 1
            n_shots_per_class = ((proportions*(support_size-n_ways)).int()+1).tolist()
            n_shots_per_class = [min(n_shots_per_class[i], self.num_elements_per_class[c]-query_size) for i,c in enumerate(picked_classes)]
        else:
            n_shots_per_class = [n_shots]*len(picked_classes)
        return n_shots_per_class

    def get_number_of_queries(self, picked_classes, query_size, unbalanced_queries):
        if unbalanced_queries:
            alpha = np.full(len(picked_classes), 2)
            prob_dist = np.random.dirichlet(alpha)
            while prob_dist.min()*query_size*len(picked_classes)<1: # if there is a class with less than one query resample
                prob_dist = np.random.dirichlet(alpha)
            n_queries_per_class = self.convert_prob_to_samples(prob=prob_dist, q_shot=query_size*len(picked_classes))
        else:
            n_queries_per_class = [query_size]*len(picked_classes)
        return n_queries_per_class

    def sample_indices(self, num_elements_per_chosen_classes, n_shots_per_class, n_queries_per_class):
        shots_idx = []
        queries_idx = []
        for k, q, elements_per_class in zip(n_shots_per_class, n_queries_per_class, num_elements_per_chosen_classes):
            choices = torch.randperm(elements_per_class)
            shots_idx.append(choices[:k].tolist())
            queries_idx.append(choices[k:k+q].tolist())
        return shots_idx, queries_idx

    def sample_episode(self, ways=0, n_shots=0, n_queries=0, unbalanced_queries=False, verbose=False):
        """
        Sample an episode
        """
        # get n_ways classes randomly
        picked_classes = self.select_classes(ways=ways)

        query_size = self.get_query_size(picked_classes, n_queries)
        support_size = self.get_support_size(picked_classes, query_size, n_shots)

        n_shots_per_class = self.get_number_of_shots(picked_classes, support_size, query_size, n_shots)
        n_queries_per_class = self.get_number_of_queries(picked_classes, query_size, unbalanced_queries)
        shots_idx, queries_idx = self.sample_indices([self.num_elements_per_class[c] for c in range(len(picked_classes))], n_shots_per_class, n_queries_per_class)

        if verbose:
            print(f'chosen class: {picked_classes}')
            print(f'n_ways={len(picked_classes)}, q={query_size}, S={support_size}, n_shots_per_class={n_shots_per_class}')
            print(f'queries per class:{n_queries_per_class}')
            print(f'shots_idx: {shots_idx}')
            print(f'queries_idx: {queries_idx}')

        return {'picked_classes':picked_classes, 'shots_idx':shots_idx, 'queries_idx':queries_idx}

    def get_features_from_indices(self, features, episode, validation=False):
        """
        Get features from a list of all features and from a dictonnary describing an episode
        """
        picked_classes, shots_idx, queries_idx = episode['picked_classes'], episode['shots_idx'], episode['queries_idx']
        if validation :
            validation_idx = episode['validations_idx']
            val = []
        shots, queries = [], []
        for i, c in enumerate(picked_classes):
            shots.append(features[c]['features'][shots_idx[i]])
            queries.append(features[c]['features'][queries_idx[i]])
            if validation :
                val.append(features[c]['features'][validation_idx[i]])
        if validation:
            return shots, queries, val
        else:
            return shots, queries

    def convert_prob_to_samples(self, prob, q_shot):
        """
        convert class probabilities to numbers of samples per class
        reused : https://github.com/oveilleux/Realistic_Transductive_Few_Shot
        Arguments:
            - prob: probabilities of each class
            - q_shot: total number of queries for all classes combined
        """
        prob = prob * q_shot
        if sum(np.round(prob)) > q_shot:
            while sum(np.round(prob)) != q_shot:
                idx = 0
                for j in range(len(prob)):
                    frac, whole = math.modf(prob[j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[idx] = np.floor(prob[idx])
            prob = np.round(prob)
        elif sum(np.round(prob)) < q_shot:
            while sum(np.round(prob)) != q_shot:
                idx = 0
                for j in range(len(prob)):
                    frac, whole = math.modf(prob[j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[idx] = np.ceil(prob[idx])
            prob = np.round(prob)
        else:
            prob = np.round(prob)
        return prob.astype(int)


class EpisodicSampler():
    """
        Sampler for episodic training
    """
    def __init__(self,
                 n_ways: int,
                 n_shots: int,
                 n_queries: int,
                 n_episodes: int,
                 generator: EpisodicGenerator = None):
        self.generator = generator
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        """
            Return indices used in one batch
            data is returned in a sequence of c1c1c1c1c2c2c2c2c3c3c3c3 with shots first then queries
        """
        for _ in range(self.n_episodes):
            episode = self.generator.sample_episode(ways=self.n_ways, n_shots=self.n_shots, n_queries=self.n_queries)
            shots = []
            queries = []
            for c, class_idx in enumerate(episode['picked_classes']):
                offset_idx = self.generator.classes.index(class_idx)
                offset = sum(self.generator.num_elements_per_class[:offset_idx])
                shots = shots + [offset+s for s in episode['shots_idx'][c]]
                queries = queries + [offset+s for s in episode['queries_idx'][c]]
            batch = torch.cat([torch.tensor(shots), torch.tensor(queries)])
            yield batch

    def split_shot_query(self, features):
        """
        Split features into shots and queries
        """
        shots, queries = [], []
        for c in range(self.n_ways):
            shots.append(features[self.n_shots*c:self.n_shots*(c+1)])
            queries.append(features[self.n_shots*self.n_ways+self.n_queries*c:self.n_shots*self.n_ways+self.n_queries*(c+1)])
        return shots, queries
