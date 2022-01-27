from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import check_scoring
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ModelPipeline:
    def __init__(
        self,
        final_estimator,
        continuous_features,
        categorical_features,
        target,
        datetime_col,
        datetime_unit,
        n_splits,
        test_size,
        max_train_size,
        gap,
        scoring,
    ):
        self.final_estimator = final_estimator
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.features = continuous_features + categorical_features
        self.target = target
        self.datetime_col = datetime_col
        self.datetime_unit = datetime_unit
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.gap = gap
        self.scoring = scoring

    def __get_transformer(self):
        return ColumnTransformer(
            transformers=[
                ("continuous", KNNImputer(), self.continuous_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ],
            sparse_threshold=0,
        )

    def __get_pipeline(self):
        return Pipeline(
            [
                ("transformer", self.__get_transformer()),
                ("scaler", StandardScaler()),
                ("estimator", deepcopy(self.final_estimator)),
            ]
        )

    def __get_folds(self, df):
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            max_train_size=self.max_train_size,
            gap=self.gap,
        )

        df_datetimes = df[self.datetime_col].dt.floor(self.datetime_unit)
        unique_datetimes = pd.date_range(df_datetimes.min(), df_datetimes.max(), freq=self.datetime_unit)

        folds = []
        for train, test in splitter.split(unique_datetimes):
            train_index = df[df_datetimes.isin(unique_datetimes[train])].index
            test_index = df[df_datetimes.isin(unique_datetimes[test])].index
            folds.append((train_index, test_index))

        return folds

    def __get_final_training_data(self, df):
        df_datetimes = df[self.datetime_col].dt.floor(self.datetime_unit)
        unique_datetimes = pd.date_range(df_datetimes.min(), df_datetimes.max(), freq=self.datetime_unit)
        return df[df_datetimes.isin(unique_datetimes[-self.max_train_size:])]

    def __get_encoded_feature_names(self, pipeline):
        encoded_feature_names = []
        categorical_feature_values = pipeline.named_steps["transformer"].named_transformers_["categorical"].categories_
        for feature_name, feature_values in zip(self.categorical_features, categorical_feature_values):
            for feature_value in feature_values:
                encoded_feature_names.append(f"{feature_name}={feature_value}")
        return encoded_feature_names

    def get_trained_pipeline(self, df):
        df_final = self.__get_final_training_data(df)
        pipeline = self.__get_pipeline()
        pipeline.fit(df_final[self.features], df_final[self.target])
        return pipeline

    def get_score(self, df):
        cv_scores = cross_val_score(
            estimator=self.__get_pipeline(),
            X=df[self.features],
            y=df[self.target],
            cv=self.__get_folds(df),
            scoring=self.scoring,
            n_jobs=-1,
            verbose=10,
        )
        return cv_scores.mean()

    def get_coefficients(self, pipeline, undo_scaling=True):
        est = pipeline.named_steps["estimator"]
        scaler = pipeline.named_steps["scaler"]
        feature_names = self.continuous_features + self.__get_encoded_feature_names(pipeline)

        if "coef_" not in dir(est):
            raise Exception("Coefficients unavailable for this estimator.")

        coefs = pd.Series(est.coef_.flatten(), index=feature_names)
        intercept = est.intercept_
        if intercept.ndim > 0:
            intercept = intercept[0]

        if undo_scaling:
            coefs = coefs / scaler.scale_
            coefs.loc["intercept"] = intercept - (coefs * scaler.mean_).sum()
        else:
            coefs.loc["intercept"] = intercept

        return coefs

    def get_coefficient_p_values(self, pipeline, df, bootstrap_samples=100):
        coefs_final = self.get_coefficients(pipeline)
        df_final = self.__get_final_training_data(df)

        coefs_boot = pd.DataFrame()
        for i in range(bootstrap_samples):
            df_samp = df_final.sample(frac=1, replace=True)
            pipeline_boot = self.__get_pipeline()
            pipeline_boot.fit(df_samp[self.features], df_samp[self.target])
            coefs_boot[i] = self.get_coefficients(pipeline_boot)

        se = coefs_boot.std(1)
        z = -abs(coefs_final / se)

        return z.apply(norm.cdf)

    def get_feature_importance(self, pipeline, df):
        p = permutation_importance(
            estimator=pipeline,
            X=df[self.features],
            y=df[self.target],
            scoring=self.scoring,
            n_jobs=-1
        )

        return pd.Series(p['importances_mean'], index=self.features)

