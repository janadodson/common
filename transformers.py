from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from scipy.stats import norm
import re


__all__ = ['Scaler', 'OneHotEncoder', 'NAFiller', 'ModelFitAndEvaluation']


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features=[]):
        self.features = features

    def _is_numeric(self, dtype):
        return np.issubdtype(dtype, np.number)

    def fit(self, X, y=None):

        # If features list is empty, transform all numeric features
        if len(self.features) == 0:
            features = []
            for feature in X.columns:
                if self._is_numeric(X.dtypes[feature]):
                    features.append(feature)
        else:
            features = self.features

        self.means = X[features].mean()
        self.stds = X[features].std(ddof=0)

        return self

    def transform(self, X, y=None):
        X_temp = X.copy()
        for feature in self.means.index:
            X_temp[feature] = (X_temp[feature] - self.means[feature])
            if self.stds[feature] != 0:
                X_temp[feature] = X_temp[feature] / self.stds[feature]
        return X_temp


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=[], encode_na=False, min_samples=1, min_pct=0, drop_features=False):
        self.features = features
        self.encode_na = encode_na
        self.min_samples = min_samples
        self.min_pct = min_pct
        self.dummy_cols = {}
        self.drop_features = drop_features

    def _is_categorical(self, dtype):
        return np.issubdtype(dtype, np.object_)

    def fit(self, X, y=None):

        # If features list is empty, encode all categorical features
        if len(self.features) == 0:
            features = []
            for feature in X.columns:
                if self._is_categorical(X.dtypes[feature]):
                    features.append(feature)
        else:
            features = self.features

        for feature in features:
            val_counts = X[feature].value_counts(dropna=not self.encode_na)
            n_vals = len(val_counts)

            # Drop small categories
            val_counts = val_counts[val_counts >= self.min_samples]
            val_counts = val_counts[val_counts / len(X) >= self.min_pct]

            # If no categories are dropped, drop smallest category
            if len(val_counts) == n_vals:
                val_counts = val_counts.drop(val_counts.idxmax())

            self.dummy_cols[feature] = val_counts.index
                    
        return self

    def transform(self, X, y=None):
        X_dummy = X.copy()

        # Add dummy variables in training set not test set
        for feature in sorted(self.dummy_cols.keys()):
            for col in self.dummy_cols[feature]:
                col_name = re.sub('\W', '_', '{}_is_{}'.format(feature, col)).lower()
                X_dummy[col_name] = (X[feature] == col).astype(int)

            if self.drop_features:
                X_dummy = X_dummy.drop(feature, axis=1)
            
        return X_dummy


class NAFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fill_type='mean'):
        self.fill_type = fill_type

    def fit(self, X, y=None):
        if self.fill_type == 'mean':
            self.fill_values = X.mean()
        elif self.fill_type == 'median':
            self.fill_values = X.median()
        elif self.fill_type == 'mode':
            self.fill_values = X.apply(lambda col: col.value_counts().idxmax())
        elif self.fill_type == 'zero':
              self.fill_values = pd.Series({col: 0 for col in X.columns})
        return self

    def transform(self, X, y=None):
        X = X.fillna(self.fill_values)
        return X


def no_transformation(x):
    return x


class ModelFitAndEvaluation():
    def __init__(self, estimator, features, target, scoring, transform_func=None, inverse_transform_func=None, bootstrap_samples=100, cv_splits=5, probability=False, time_series_date_col=None, time_series_fold_size='28D'):
        self.estimator = deepcopy(estimator)
        self.features = features
        self.target = target
        self.scoring = scoring
        self.transform_func = transform_func
        if transform_func is None:
            self.transform_func = no_transformation
        self.inverse_transform_func = inverse_transform_func
        if inverse_transform_func is None:
            self.inverse_transform_func = no_transformation
        self.bootstrap_samples = bootstrap_samples
        self.cv_splits = cv_splits
        self.probability = probability
        self.time_series_date_col = time_series_date_col
        self.time_series_fold_size = time_series_fold_size

        self.model = Pipeline([
            ('filler', NAFiller(fill_type='median')), 
            ('scaler', Scaler()), 
            ('estimator', self.estimator)
        ])
        
        
    def __get_time_series_folds(self, df):
        folds = []
        split_date = df[self.time_series_date_col].max() - pd.Timedelta(self.time_series_fold_size)
        while split_date >= df[self.time_series_date_col].min() + pd.Timedelta(self.time_series_fold_size):
            df_train = df[df[self.time_series_date_col] <= split_date].copy()
            df_test = df[(df[self.time_series_date_col] > split_date) & (df[self.time_series_date_col] <= split_date + pd.Timedelta(self.time_series_fold_size))].copy()
            folds.append((df_train, df_test))
            split_date = split_date - pd.Timedelta(self.time_series_fold_size)
        return folds
    
    
    def __get_k_folds(self, df):
        folds = []
        for train, test in KFold(n_splits=self.cv_splits, shuffle=True).split(df):
            df_train, df_test = df.iloc[train], df.iloc[test]
            folds.append((df_train, df_test))
        return folds

    
    def __get_folds(self, df):
        if self.time_series_date_col is not None:
            return self.__get_time_series_folds(df)
        return self.__get_folds(df)
    

    # Calculate coefficients for un-standardized features
    def __set_coefficients(self):
        
        coefficients = pd.Series(self.model.named_steps['estimator'].coef_.flatten(), index=self.features)
        intercept = self.model.named_steps['estimator'].intercept_
        if intercept.ndim > 0:
            intercept = intercept[0]
        
        stds = self.model.named_steps['scaler'].stds
        means = self.model.named_steps['scaler'].means

        coefficients_trans = coefficients / stds
        coefficients_trans.loc['intercept'] = intercept - (coefficients_trans * means).sum()

        self.coefficients = coefficients_trans


    # Use bootstrapping to get coefficient p-values
    def __set_p_values(self):

        m = deepcopy(self)

        coefs_bootstrap = pd.DataFrame(index=list(m.features) + ['intercept'], columns=range(self.bootstrap_samples))
        for i in range(self.bootstrap_samples):
            m.fit(self.df_fit.sample(frac=1, replace=True))
            coefs_bootstrap[i] = m.coefficients

        se = coefs_bootstrap.std(1)
        z = -abs(self.coefficients / se)

        self.p_values = z.apply(norm.cdf)
        
        
    def __get_score(self, y_true, y_pred):
        y_pred_temp = y_pred.dropna()
        return self.scoring(y_true.loc[y_pred_temp.index], y_pred_temp)


    # Use cross-validation to calculate an overall score for the model
    def __set_score(self):
        m = deepcopy(self)

        y_pred = pd.Series(index=self.df_fit.index)
        folds = self.__get_folds(self.df_fit)
        for df_train, df_test in folds:
            m.fit(df_train)
            y_pred.loc[df_test.index] = m.predict(df_test)
        
        self.score = self.__get_score(self.df_fit[self.target], y_pred)
        self.cv_predictions = y_pred


    # Use permutation importance to calculate the impact of each feature on the overall score for the model
    def __set_feature_importances(self):
        m = deepcopy(self)

        y_pred = pd.DataFrame(index=self.df_fit.index, columns=self.features)
        folds = self.__get_folds(self.df_fit)
        for df_train, df_test in folds:
            m.fit(df_train)

            for feature in self.features:
                df_test_temp = df_test.copy()
                np.random.shuffle(df_test_temp[feature].values)
                y_pred.loc[df_test.index, feature] = m.predict(df_test_temp)
        
        feature_perm_scores = pd.Series({
            feature: self.__get_score(self.df_fit[self.target], y_pred[feature])
            for feature in self.features
        })

        self.feature_importances = self.score - feature_perm_scores


    def fit(self, df):
        self.df_fit = df.dropna(subset=[self.target])
        self.model.fit(self.df_fit[self.features],  self.transform_func(self.df_fit[self.target]))

        try:
            self.__set_coefficients()
        except:
            pass


    def predict(self, df):
        if self.probability:
            return self.inverse_transform_func(pd.Series(self.model.predict_proba(df[self.features])[:, 1], index=df.index))
        else:
            return self.inverse_transform_func(pd.Series(self.model.predict(df[self.features]), index=df.index))


    def evaluate(self, df):
        self.fit(df)
        self.__set_score()
        self.__set_feature_importances()

        try:
            self.__set_p_values()
        except:
            pass


    def recursive_feature_elimination(self, df, step_size_pct=0.5):
        m = deepcopy(self)
        m.bootstrap_samples = 0

        feature_list = []
        score_list = []
        keep_features = self.features

        while len(keep_features) > 0:
            m.features = keep_features

            m.fit(df)
            m.evaluate(df)
            
            print('Tested {} features'.format(len(keep_features)))

            feature_list.append(keep_features)
            score_list.append(m.score)

            feature_importances = m.feature_importances
            keep_features = feature_importances[feature_importances > feature_importances.quantile(step_size_pct)].index

        return feature_list, score_list
