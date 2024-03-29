from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import random
import pandas as pd
from datetime import datetime
import numpy as np
import concurrent.futures
from numba import vectorize

'''
This file contains the code for the Forest Based Comparable Sales Method made by D.R. Kok for his thesis at the TU/e.
Notes on the hyperparameters:
- nr_trees need to be in stepsize 10
- subset_size needs to be between 0.1 and 1 with a stepsize of 0.1
- int_function has three possible entries: None, 'avg' and 'weighted_avg'
- transformation has three possible entries: None, 'semi-log' and 'log-log'
'''

class ProposedModel:
    def __init__(self,
                 nr_trees = 10,
                 min_samples_leaf = 100,
                 min_impurity_decrease = 0.0,
                 subset_size = 0.8,
                 bisquare_bandwidth = 1,
                 distance_factor = 1,
                 temporal_factor = 1,
                 K = 5,
                 int_function = 'weighted_avg',
                 transformation = None,
                 random_state = None):
        self.nr_trees = nr_trees
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.subset_size = subset_size
        self.bisquare_bandwidth = bisquare_bandwidth
        self.distance_factor = distance_factor
        self.temporal_factor = temporal_factor
        self.K = K
        self.int_function = int_function
        self.transformation = transformation
        self.random_state = random_state
        
        self.invalid_hyperparameter = False
        self.training_set = None
        self.test_set = None
        self.dep_var = None
        self.forest = []
        self.subsets = []
        self.properties_per_leaf_forest = []
        self.neighborhood = None
        self.adjustmentfactorsmodel = None

        self.hyperparameter_check()
    
    def hyperparameter_check(self):
        '''
        This function contains a few checks for the input parameters.
        '''
        
        if self.nr_trees % 10 != 0:
            self.invalid_hyperparameter = True
            print(f'{self.nr_trees} trees is not a valid hyperparameter. Please initialize a valid model.')
        if self.subset_size * 10 % 1 != 0 or self.subset_size > 1 or self.subset_size < 0:
            self.invalid_hyperparameter = True
            print(f'A subset size of {self.subset_size} is not a valid hyperparameter. Please initialize a valid model.')
        if self.int_function != 'avg' and self.int_function != 'weighted_avg':
            self.invalid_hyperparameter = True
            print(f'"{self.int_function}" is not a valid intregation function. Please initialize a valid model.')


    def fit(self, training_set, dep_var):
        '''
        This function is used for fitting the model. It takes as input the training set and the name of the dependend variable. 
        It then trains the forest as well as the adjustment factors. 
        '''
        
        if self.invalid_hyperparameter == True:
            print('This model contains invalid hyperparameters, please initialize a valid model.')
            return

        self.training_set = training_set.rename_axis("comparable_id")
        self.dep_var = dep_var

        self.train_forest()

        self.train_adjustment_factors()

        return self

    def predict(self, test_set):
        '''
        With the predict function the predictions are made for the samples in the test set. It takes the test set (excluding dependent variable) as input
        and outputs a list with predictions in order. 
        '''
        
        self.test_set = test_set.rename_axis("target_id")
        self.test_set["RDOS"] = 0

        self.forest_based_neighborhood()

        self.similarity_score()

        self.distance_score()

        self.temporal_score()

        self.comparability_score()

        self.nearest_neighbors()

        self.adjustment_factors()

        self.integrator()

        return self.predictions
    

    def MAPE(self, y):
        '''
        This function can be called upon to return the test set with the percentage errors of each sample. 
        It takes as input the list of transaction prices of the test set.
        '''
        self.test_set = (
            self.test_set.join(self.predictions, on = "target_id")
            .join(y, on = "target_id")
        )
        self.test_set["perc_error"] = (self.test_set[self.dep_var] - self.test_set["prediction"]) \
                                        / self.test_set[self.dep_var]
        
        return self.test_set


    def train_forest(self):
        '''
        The train_forest function uses the training set to build the forest. The amount of trees and subset size is determined by the user. 
        First the subsets of training samples are made for each tree by randomly assigning the correct amount of trees to each sample in the training set. 
        Then each tree is made using this subset. This tree is build using the build_tree function. Of each tree, all properties and their respective leaf nodes are stored.
        '''
        trees = list(range(self.nr_trees))
        self.subsets = []
        random.seed(self.random_state)
        
        for i in trees:
            self.subsets.append([])

        for i in self.training_set.index:
            assigned_trees = random.sample(trees, 
                                         int(self.nr_trees * self.subset_size))
            for k in assigned_trees:
                self.subsets[k].append(i)    
               
        self.forest = []
        self.properties_per_leaf_forest = []

        with concurrent.futures.ThreadPoolExecutor() as executor:

            tree_ids = list(range(len(self.subsets)))
            forest = executor.map(self.build_tree, tree_ids)

        for i in forest:
            self.forest.append(i['tree'])
            self.properties_per_leaf_forest.append(i['leaf'])


    def build_tree(self, tree_id):
        '''
        This function build each tree of the forest using the DecisionTreeRegressor from Scikit-Learn. 
        The leaf in which each training sample ends up in is stored in order to create the neighborhood for each target sample.
        '''
        
        subset = self.training_set.loc[self.subsets[tree_id]]
        y = subset[self.dep_var]
        X = subset.drop([self.dep_var], axis = 1)

        regressor = DecisionTreeRegressor(
                    min_samples_leaf = self.min_samples_leaf,
                    min_impurity_decrease = self.min_impurity_decrease,
                    random_state = self.random_state)

        tree = regressor.fit(X, y)

        leaf_of_property = tree.apply(X)
        leaf_of_property = {"comparable_id": X.index, "leaf": leaf_of_property}
        leaf_of_property = pd.DataFrame(leaf_of_property)

        return {'tree' : tree, 'leaf' : leaf_of_property}
    

    def train_adjustment_factors(self):       
        '''
        The coefficients for the adjustment factors are determined through multiple linear regression. 
        If a transformation is needed, this is performed before fitting the linear regression.
        '''
        X_training = self.training_set.drop([self.dep_var, "longitude", "latitude"], axis= 1)
        y_training = self.training_set[self.dep_var]
        
        if self.transformation == 'semi-log':
            y_training = np.log(y_training.astype(float))

        if self.transformation == 'log-log':
            y_training = np.log(y_training.astype(float))
            X_training['FloorArea'] = np.log(X_training[['FloorArea']].astype(float))
            X_training['LotSize'] = np.log(X_training[['LotSize']].astype(float))
        
        self.X_training = X_training 

        regressor = LinearRegression(n_jobs= -1)
        self.adjustmentfactorsmodel = regressor.fit(X_training, y_training)
    

    def forest_based_neighborhood(self):
        '''
        This is the first function for predicting the values of the target properties. 
        It finds in which leaf of each tree the target property lands and finds all training properties in the same leaves.
        '''
        
        X = self.test_set
        self.neighborhood = np.empty((0,2), int)
        for i in range(len(self.forest)):
            new_neighborhood = (
                pd.DataFrame({"target_id": X.index, "leaf": self.forest[i].apply(X)})
                .merge(self.properties_per_leaf_forest[i], on = "leaf")
                .drop("leaf", axis = 1)
            )
            self.neighborhood = np.append(self.neighborhood, new_neighborhood, axis = 0)

        self.neighborhood = (
            pd.DataFrame(self.neighborhood)
            .rename(columns={0:"target_id", 1:"comparable_id"})
        )

        
    def similarity_score(self):
        '''
        This function calculates the similarity score for each of the comparables in the potential neighborhood. 
        It counts how often a comparable ends up in the same leaf as the target. 
        Then it calculates the similarity as a fraction of actual same end leaves and possible same end leaves.
        '''

        self.neighborhood = (
            pd.DataFrame(self.neighborhood.groupby(["target_id", "comparable_id"]).size())
            .rename(columns={0: "similarity_score"})
        )
        maximum_similarity = int(self.nr_trees * self.subset_size)
        self.neighborhood["similarity_score"] = self.neighborhood["similarity_score"] / maximum_similarity * 100
        

    def distance_score(self):
        '''
        The distance score is calculated using the spatial distance between the comparable and the target property.
        It calculates this distance with the geographical coordinates. 
        A separate vectorized function is used for this calculation to speed up the computation time. 
        '''
        
        self.neighborhood = (
            self.neighborhood.join(self.test_set[["longitude", "latitude"]], on = "target_id")
            .rename(columns= {"longitude": "longitude_target", "latitude": "latitude_target"})
            .join(self.training_set[["longitude", "latitude", "RDOS"]], on = "comparable_id")
        )

        longitude_target = self.neighborhood["longitude_target"].values
        latitude_target = self.neighborhood["latitude_target"].values
        longitude = self.neighborhood["longitude"].values
        latitude = self.neighborhood["latitude"].values

        self.neighborhood['distance_score'] = self.calculate_distance(
                                                self.bisquare_bandwidth, 
                                                longitude_target, 
                                                latitude_target, 
                                                longitude, latitude)

        self.neighborhood = self.neighborhood.drop(["longitude_target", 
                                                    "latitude_target", 
                                                    "longitude", "latitude"]
                                                    , axis = 1)
        

    @vectorize
    def calculate_distance(bandwidth, longitude_target, latitude_target, longitude, latitude):
        '''
        This is the vectorized distance function. It is a simple but effective formula for calculating the distance between to geographical points.
        It is an approximation as it disregards the curvature of the earth and uses a fixed value for the longitude factor. 
        However, this is accurate enough for calculations in the Netherlands.
        '''
        distance = np.sqrt(
                        ((latitude_target - latitude) * 111.32) ** 2
                        + ((longitude_target - longitude) * 68.29) ** 2
                        )
        
        if distance < bandwidth:
            distance = (1 - (distance / bandwidth) ** 2) ** 2 * 100
        else:
            distance = float(0)

        return distance
                                    

    def temporal_score(self):
        '''
        The temporal score is a linear degrading function which calculates a score between 0 and 100 
        by using the temporal distance in days between the transaction date and the day after the end of the training set window.
        '''
        
        max_days = self.training_set["RDOS"].max()
        self.neighborhood["temporal_score"] = 100 - ((self.neighborhood["RDOS"] - 1) / (max_days - 1) * 100)
        self.neighborhood = self.neighborhood.drop("RDOS", axis = 1)


    def comparability_score(self):
        '''
        The comparability score is the combination of the three scores above by way of weighted average. 
        The weights are factors determined by the user in the hyperparameters 'distance_score' and 'temporal_score' the weight for the similarity score is set to 1.
        '''
        
        self.neighborhood["comparability_score"] = (
            (
            self.neighborhood["similarity_score"]
            + self.distance_factor * self.neighborhood["distance_score"]
            + self.temporal_factor * self.neighborhood["temporal_score"]
            ) 
            / (1 + self.distance_factor + self.temporal_factor)
        )

    
    def nearest_neighbors(self):
        '''
        The final neighorbhood is determined by sorting the potential neighborhoods and selecting the K comparables with the highest comparability score.
        '''
       
        self.neighborhood = (
            self.neighborhood.sort_values("comparability_score", ascending=False)
            .sort_index(level = "target_id", sort_remaining = False)
            .groupby("target_id").head(self.K)
        )     
        
    
    def adjustment_factors(self):
        '''
        The multiple linear regression built in the function 'train_adjustment_factors' is used to predict values for all neighbors and targets. 
        The difference between the neighbor and target is added to the transaction price of the neighbor to get the comparable prediction. 
        Transformations are performed if necessary.
        '''

        X_targets = self.test_set.drop(["longitude", "latitude"], axis = 1)

        if self.transformation == 'log-log':
            X_targets['FloorArea'] = np.log(X_targets[['FloorArea']].astype(float))
            X_targets['LotSize'] = np.log(X_targets[['LotSize']].astype(float)) 

        IDs_neighbors = list(set(self.neighborhood.index.get_level_values(1)))
        neighbors = self.training_set.loc[IDs_neighbors]

        self.prediction_targets = self.adjustmentfactorsmodel.predict(X_targets)
        self.prediction_targets = {"target_id": X_targets.index, "regression_prediction_target": self.prediction_targets}
        prediction_targets = pd.DataFrame(self.prediction_targets).set_index("target_id")

        prediction_neighbors = self.adjustmentfactorsmodel.predict(self.X_training)
        prediction_neighbors = {"comparable_id": self.X_training.index, "regression_prediction_comparable": prediction_neighbors}
        prediction_neighbors = pd.DataFrame(prediction_neighbors).set_index("comparable_id")

        if self.transformation == 'semi-log' or self.transformation == 'log-log':
            prediction_targets = np.exp(prediction_targets)
            prediction_neighbors = np.exp(prediction_neighbors)
        
        target_ids = self.neighborhood.index.get_level_values(0)
        neighbor_ids = self.neighborhood.index.get_level_values(1)

        self.neighborhood["regression_prediction_comparable"] = prediction_neighbors.loc[neighbor_ids].values.copy()
        self.neighborhood["regression_prediction_target"] = prediction_targets.loc[target_ids].values.copy()
        self.neighborhood["transaction_price_comparable"] = neighbors[[self.dep_var]].loc[neighbor_ids].values.copy()

        self.neighborhood["delta"] = (
            self.neighborhood["regression_prediction_target"].copy() 
            - self.neighborhood["regression_prediction_comparable"].copy()
        )

        self.neighborhood["comparable_prediction"] = (
            self.neighborhood["transaction_price_comparable"].copy() 
            + self.neighborhood["delta"].copy()
        )


    def integrator(self):
        '''
        This is the integration function. Both the average and weighted average are options, chosen as a hyperparameter by the user. 
        The weight in the weighted average is the comparability score of the concerned neighbor.
        '''
        
        if self.int_function == 'weighted_avg':
            total_weight = self.neighborhood.groupby("target_id")["comparability_score"].sum()
            prediction_times_weight = self.neighborhood["comparability_score"] \
                                    * self.neighborhood["comparable_prediction"]
            cum_predictions = prediction_times_weight.groupby("target_id").sum()
            self.predictions = cum_predictions / total_weight
            self.predictions = pd.DataFrame(self.predictions)
            self.predictions = self.predictions.rename(columns= {0: "prediction"})

        elif self.int_function == 'avg':
            self.predictions = self.neighborhood["comparable_prediction"].groupby("target_id").mean()
            self.predictions = pd.DataFrame(self.predictions)
            self.predictions = self.predictions.rename(columns= {"comparable_prediction": "prediction"})
