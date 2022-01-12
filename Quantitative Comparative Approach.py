import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

'''
This file contains the code for the Quantitative Comparative Approach by made I-Cheng Yeh and Tzu-Kuang Hsu 
proposed in their paper 'Building real estate valuation models with comparative approach through case-based reasoning' (2018).
It was remade by D.R. Kok for his thesis at the TU/e for the purpose of studying its predictive capabilities in relation to the Forest Based Comparable Sales Method.
Notes on the hyperparameters:
- transformation has three possible entries: None, 'semi-log' and 'log-log'.
'''

class QCA:
    def __init__(
                self,
                K = 5,
                effect_radius = 3,
                transformation = None,
                **importance
                ):

        self.K = K
        self.effect_radius = effect_radius
        self.transformation = transformation
        self.importance = importance
        
        self.training_set = None
        self.test_set = None
        self.dep_var = None

    
    def fit(self, training_set, dep_var):
        '''
        This function is used for fitting the model. It takes as input the training set and the name of the dependend variable. 
        It then trains the adjustment coefficients. 
        '''
        
        self.training_set = training_set.rename_axis("comparable_id")
        self.dep_var = dep_var

        self.train_adjustmentcoefficients()

        return self
    
    def predict(self, test_set):
        '''
        With the predict function the predictions are made for the samples in the test set. It takes the test set (excluding dependent variable) as input
        and outputs a list with predictions in order. 
        '''
        
        self.test_set = test_set.rename_axis("target_id")
        self.test_set["RDOS"] = 0

        self.calculate_distance()

        self.get_neighborhood()

        self.calculate_weight()

        self.calculate_adj_price()

        self.integrator()
        
        return self.predictions


    def MAPE(self, y):
        '''
        This function can be called upon to return the test set with the percentage errors of each sample. 
        It takes as input the list of transaction prices of the test set.
        '''
        
        self.test_set = (
            self.test_set
            .join(self.predictions, on = "target_id")
            .join(y, on = "target_id")
        )
        self.test_set["perc_error"] = (self.test_set[self.dep_var] - self.test_set["prediction"]) \
                                        / self.test_set[self.dep_var]
        
        return self.test_set


    def train_adjustmentcoefficients(self):
        '''
        This function uses the training set to calculate the adjustment coefficients. 
        Firstly, the variables are ordered by their importance. 
        If a transformation of the variables is necessary it is performed before the coefficients are calculated.
        Then the coefficients are stepwise calculated as described by yeh and Hsu (2018)
        
        '''
        
        self.features_ordered = pd.DataFrame(columns= ['Feature', 'Importance', 'regression_model'])
        for k, v in self.importance.items():
                self.features_ordered = self.features_ordered.append({'Feature': k, 
                                                            'Importance': v, 
                                                            'regression_model': None}, 
                                                            ignore_index=True)

        lon_lat = self.features_ordered.loc[self.features_ordered['Feature'] == 'lon_lat']
        self.features_ordered = (
            self.features_ordered
            .drop(lon_lat.index)
            .sort_values(by= 'Importance', ascending=False)
            .append(lon_lat)
            .reset_index(drop= True)
        )

        if self.transformation == 'semi-log':
            self.training_set[self.dep_var] = np.log(self.training_set[self.dep_var]).astype(float)

        if self.transformation == 'log-log':
            self.training_set[self.dep_var] = np.log(self.training_set[self.dep_var]).astype(float)
            self.training_set['FloorArea'] = np.log(self.training_set[['FloorArea']].astype(float))
            self.training_set['LotSize'] = np.log(self.training_set[['LotSize']].astype(float))
        
        self.features_ordered['Importance'] = self.features_ordered['Importance'] / self.features_ordered['Importance'].min()

        average_dep_var = self.training_set[self.dep_var].mean()

        self.training_set['agg_adj_coeff'] = self.training_set[self.dep_var] / average_dep_var

        self.predictions_adj_factor_comp = pd.DataFrame(columns=self.features_ordered['Feature'])

        for i in range(len(self.features_ordered['Feature'])):
            
            if self.features_ordered['Feature'][i] != 'lon_lat':
                X_feature = self.training_set[[self.features_ordered['Feature'][i]]].to_numpy()
            else:
                X_feature = self.training_set[['longitude', 'latitude']].copy()
                transformer = PolynomialFeatures(2, include_bias= False)
                X_feature = transformer.fit_transform(X_feature)

            y_coefficient = self.training_set.iloc[:,-1:].copy()
            regressor = LinearRegression(n_jobs= -1)
            model = regressor.fit(X_feature, y_coefficient)
            self.features_ordered.loc[i, 'regression_model'] = model

            prediction = model.predict(X_feature)
            self.predictions_adj_factor_comp[self.features_ordered['Feature'][i]] = prediction.reshape(1,len(self.training_set)).tolist()[0]

            if i < len(self.features_ordered['Feature']) - 1:
                self.training_set['adj_coeff_' + self.features_ordered['Feature'][i + 1]] = y_coefficient / prediction

        self.predictions_adj_factor_comp.index = self.training_set.index

    def calculate_distance(self):
        '''
        The multidimensional distance is calculated with a weighted mahanalobis distance. 
        Firstly the necessary transformations are performed. Then the neighborhood is created by combining the comparables with the targets.
        This combined table is used to calculated the multidimensional distance. 
        '''
        
        if self.transformation == 'log-log':
            self.test_set['FloorArea'] = np.log(self.test_set[['FloorArea']].astype(float))
            self.test_set['LotSize'] = np.log(self.test_set[['LotSize']].astype(float))
        
        features = list(self.features_ordered['Feature'])
        features.remove('lon_lat')
        features += ['longitude', 'latitude']

        features_comp_for_extraction = features + ['comparable_id']       
        features_comp =[i + '_comp' for i in features]
        features_comp += ['comparable_id']  
        features += ['target_id']

        targets = self.test_set.reset_index()[features]
        comparables = np.array(self.training_set.reset_index()[features_comp_for_extraction])

        self.neighborhood = np.empty((0,len(features) + len(features_comp)), int)

        for index, row in targets.iterrows():
            target = np.tile(np.array(row), (len(comparables), 1))
            concat = np.concatenate((target, comparables), axis = 1)
            self.neighborhood = np.concatenate((self.neighborhood, concat))

        self.neighborhood = pd.DataFrame(data= self.neighborhood, columns= features + features_comp)
        self.neighborhood[['target_id','comparable_id']] = self.neighborhood[['target_id','comparable_id']].astype(int)
        self.neighborhood = self.neighborhood.set_index(['target_id', 'comparable_id'])

        cum_distance = 0
        cum_weight = 0
        for feature in self.features_ordered['Feature']:
            weight = float(self.features_ordered.loc[self.features_ordered['Feature'] == feature, 'Importance'])
            if feature == 'lon_lat':
                std_dev_lon = np.std(self.training_set['longitude'])
                std_dev_lat = np.std(self.training_set['latitude'])
                delta_feature_lon = self.neighborhood['longitude'] - self.neighborhood['longitude_comp']
                delta_feature_lat = self.neighborhood['latitude'] - self.neighborhood['latitude_comp']
                feature_distance = (weight * ((delta_feature_lon/std_dev_lon) ** 2) 
                                    + weight * ((delta_feature_lat/std_dev_lat) ** 2))
                cum_weight += 2 * weight
            else:
                std_dev = np.std(self.training_set[feature])
                delta_feature = self.neighborhood[feature] - self.neighborhood[feature + '_comp']
                feature_distance = weight * ((delta_feature/std_dev) ** 2)
                cum_weight += weight
            
            cum_distance += feature_distance

        self.neighborhood['distance'] = np.sqrt(cum_distance / cum_weight)
        

    def get_neighborhood(self):
        '''
        The final neighborhood is determined by sorting the potential neighborhoods and selecting the K comparables with the highest lowest distance.
        '''
        
        self.neighborhood = (
            self.neighborhood
            .sort_values("distance", ascending=True)
            .sort_index(level = "target_id", sort_remaining = False)
            .groupby("target_id")
            .head(self.K)
        )


    def calculate_weight(self):
        '''
        The weight for the weighted average integration formula is calculated by transforming the distance value 
        using an exponential function with a user determined effect radius.
        '''
        
        self.neighborhood['weight'] = np.exp(-((self.neighborhood['distance']/self.effect_radius)**2))


    def calculate_adj_price(self):
        '''
        The comparable predictions are calculated using the adjustment coefficients calculated in the function 'train_adjustmentcoefficients'.
        
        '''
        
        predictions_adj_factor_target = pd.DataFrame(columns= self.predictions_adj_factor_comp.columns)
        for feature in predictions_adj_factor_target.columns:
            if feature == 'lon_lat':
                X_feature = self.test_set[['longitude', 'latitude']].copy()
                transformer = PolynomialFeatures(2, include_bias= False)
                X_feature = transformer.fit_transform(X_feature)
            else:
                X_feature = self.test_set[[feature]].to_numpy()

            model = self.features_ordered.loc[self.features_ordered['Feature'] == feature, 'regression_model'].values[0]
            predictions_adj_factor_feature = model.predict(X_feature)
            predictions_adj_factor_target[feature] = predictions_adj_factor_feature.reshape(1,len(self.test_set)).tolist()[0]
        
        predictions_adj_factor_target.index = self.test_set.index

        product_comparable = pd.DataFrame(data= self.predictions_adj_factor_comp.product(axis = 1), columns= ['agg_adj_factor_comparable']).rename_axis('comparable_id')
        product_target = pd.DataFrame(data= predictions_adj_factor_target.product(axis= 1), columns= ['agg_adj_factor_target']).rename_axis('target_id')

        self.neighborhood = (
            self.neighborhood
            .join(self.training_set[self.dep_var], on= 'comparable_id')
            .join(product_comparable, on= 'comparable_id')
            .join(product_target, on= 'target_id')
        )

        self.neighborhood['adj_price'] = self.neighborhood[self.dep_var] * self.neighborhood['agg_adj_factor_target'] / self.neighborhood['agg_adj_factor_comparable']
        
        if self.transformation == 'semi-log' or self.transformation == 'log-log':
            self.neighborhood['adj_price'] = np.exp(self.neighborhood['adj_price'])

    def integrator(self):
        total_weight = self.neighborhood.groupby("target_id")["weight"].sum()
        prediction_times_weight = self.neighborhood["weight"] \
                                * self.neighborhood['adj_price']
        cum_predictions = prediction_times_weight.groupby("target_id").sum()
        self.predictions = cum_predictions / total_weight
        self.predictions = pd.DataFrame(self.predictions)
        self.predictions = self.predictions.rename(columns= {0: "prediction"})
