# -*- coding: utf-8 -*-
import glob, os, json, math, shutil
#import sc_utilities as util
import matlab.engine
import pandas as pd
import numpy as np
import nibabel as nb

# sklearn
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet, ElasticNetCV  # L2-regularized regression
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

class Classification:
    
    def __init__(self, config):
        '''
        
        
        Attributes
            ----------
        
        '''
        
        #>>> I. Initiate variable -------------------------------------
        self.config = config # load config info
        


    def prediction_model(self,df=None,target_col="age",parcels_col="rois",feature_cols= False,model_name="Ridge",model_param=None,output_dir=False,output_tag=False,verbose=True,save=False):
        '''
        This function is used to train a model and predict the target variable
        Different model can be set: SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor
        
        Attributes
        ----------
        df: dataframe where the data are stored
            The dataframe should contain the feature columns and the target column
        target_col: str
            The name of the target column
        feature_cols: list
            The name of the feature columns
        task: str
            could be "prediction" for regression or "classification" for categorical data
        model: str
            The name of the model to use for the prediction
        model_param: dict
            The parameters of the model
        save: bool
            If True, the model will be saved in the model folder
        output_dir: str
            The path to the output directory
        output_tag: str
            The tag to add to the output files
                    
        Return
        ----------
        
        '''
        
        # --- Initiate variables
        if df is None or feature_cols is False:
            raise ValueError(">>>> df and feature_cols should be provided")
        
        if model_param is None:
            model_param = {"alpha": 1.2, "max_iter": 10000, "l1_ratio": 0}        
        else:
            model = ElasticNet(**model_param)
                
     
        # --- Get unique values ---
        participant_ids = df["IDs"].unique()
        parcels = sorted(df[parcels_col].unique())
        n_ids= len(participant_ids)
        n_parcels = len(parcels)
        n_features = len(feature_cols)
        parcel_index = {parcel: i for i, parcel in enumerate(parcels)} #

        # --- Create and fill tensor (ids × parcels × features) ---
        data_tensor = np.zeros((n_ids, n_parcels, n_features))
        age_vector = np.zeros(n_ids)

        for i, subj in enumerate(participant_ids):
            subj_df = df[df["IDs"] == subj]
            age_vector[i] = subj_df[target_col].iloc[0]
            for _, row in subj_df.iterrows():
                p_idx = parcel_index[row[parcels_col]]
                data_tensor[i, p_idx, :] = row[feature_cols].values
        
        X = data_tensor.reshape(n_ids, -1) #Reshape for regression: subject × (parcel × feature) 
        y = age_vector


        # --- Cross-validation ---
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
        y_true_all, y_pred_all, ids_all, mae_list, r2_list,mse_list, coef_list = [], [], [], [], [], [], []

        if verbose == True:
            print("Running ElasticNet on combined features...")
        
        for train_idx, test_idx in rkf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            ids_test = participant_ids[test_idx]

            # Handle missing values 
            imputer = SimpleImputer(strategy="mean")  # You can use 'median' or 'constant'
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            # Standardize 
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            ids_all.extend(ids_test) # store the true value

            mae_list.append(mean_absolute_error(y_test, y_pred))
            mse_list.append(mean_squared_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))
            coef_list.append(model.coef_)

        # --- Store CV metrics and predictions ---
        mean_mae = np.array(mae_list).mean()
        mean_mse = np.array(mse_list).mean()
        mean_r2 = np.array(r2_list).mean()
        mean_coef = np.abs(np.array(coef_list)).mean(axis=0)

        if verbose==True:
            print(f"Mean R²: {mean_r2:.3f} ± {np.array(r2_list).std():.3f}")
            print(f"Mean MAE: {mean_mae:.3f} ± {np.array(mae_list).std():.3f}")

        y_true_all = np.array(y_true_all) 
        y_pred_all = np.array(y_pred_all)

        # --- Compute the bias correction model (Cole et al.) ---
        bias_model = LinearRegression().fit(y_true_all.reshape(-1, 1), y_pred_all.reshape(-1, 1))
        intercept = bias_model.intercept_[0]
        slope = bias_model.coef_[0][0]
        y_pred_corrected = (y_pred_all - intercept) / slope
        PAD_corrected = y_pred_corrected - y_true_all

        # --- Store prediction in a dataframe ---
        df_pred = pd.DataFrame({"participant_id": ids_all,"y_true": y_true_all, "y_pred": y_pred_all})
        df_pred["y_pred_corr"] =y_pred_corrected
        df_pred["PAD_corr"] =y_pred_corrected - y_true_all
        df_pred["error"] = df_pred["y_true"] - df_pred["y_pred"]
        df_pred["error_abs"] = abs(df_pred["error"])
        df_pred["error_sq"] = df_pred["error"]**2
        df_agg = df_pred.groupby("participant_id").agg({
                "y_true": "mean",
                "y_pred": "mean",
                "y_pred_corr": "mean",
                "PAD_corr": "mean",
                "error": "mean",
                "error_abs": "mean",
                "error_sq": "mean"
            }).reset_index()

        
        # --- Store CV results in a dataframe ---#
        results = {}
        results["mae"] = mean_mae.mean()
        results["mse"] = mean_mse.mean()
        results["r2"] = mean_r2.mean()

         # --- Store the feature weights in a dataframe ---#
        mean_coef = np.abs(np.array(coef_list)).mean(axis=0)  # shape (n_parcels × n_features,)
        mean_coef_matrix = mean_coef.reshape((n_parcels, n_features))  # shape (parcels, features)
        mean_coef_per_feature = mean_coef_matrix.mean(axis=0)  # shape (features,)
        total = np.abs(mean_coef_per_feature).sum()
        percent_contrib = 100 * np.abs(mean_coef_per_feature) / total if total != 0 else np.zeros_like(mean_coef_per_feature)
        
        importances = pd.DataFrame({
                    "Feature": feature_cols,
                    "Coefficient": mean_coef_per_feature,
                    "AbsCoefficient": np.abs(mean_coef_per_feature),
                    "PercCoefficient": percent_contrib
                }).sort_values(by="AbsCoefficient", ascending=False)

        # --- Save prediction and CV results dataframes --- #
        if save==True:
            model_path = os.path.join(output_dir, "model_" + model + "_" + output_tag + ".csv")
            df_agg.to_csv(model_path, index=False)
            
            results_path = os.path.join(output_dir, "results_" + model + "_" + output_tag + ".csv")
            pd.DataFrame(results, index=[0]).to_csv(results_path, index=False)
        
        return results, df_agg, importances
