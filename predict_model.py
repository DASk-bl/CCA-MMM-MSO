import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
import math
warnings.filterwarnings("ignore")
from .helper_funcs import get_complete_db_data_as_df, get_query_res
from bq_queries import GET_COMPLETE_DATA_FOR_ID
from common.extensions import cache
# Global variable
progress_var = [0, 0]

class predictor_with_seasonality:
    def __init__(self, df, target_type):
        """
        intialization of the class
        df (dataframe) : aggregated dataframe
        date_range (array) : start and end date
        """

        self.df = df

        if "impression" in df.columns:
            self.use_impression = True
        else:
            self.use_impression = False
            self.target_type = target_type.lower()

        self.df_param = None

        global progress_var
        progress_var = [0, len(self.df.dimension.unique())]

    def s_curve_hill(
        self,
        X,
        a,
        b,
        c,
        coeff1,
        coeff2,
        coeff3,
        coeff4,
        coeff5,
        coeff6,
        coeffm1,
        coeffm2,
        coeffm3,
        coeffm4,
        coeffm5,
        coeffm6,
        coeffm7,
        coeffm8,
        coeffm9,
        coeffm10,
        coeffm11,
    ):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a))
            + X["weekday_1"] * coeff1
            + X["weekday_2"] * coeff2
            + X["weekday_3"] * coeff3
            + X["weekday_4"] * coeff4
            + X["weekday_5"] * coeff5
            + X["weekday_6"] * coeff6
            + X["month_2"] * coeffm1
            + X["month_3"] * coeffm2
            + X["month_4"] * coeffm3
            + X["month_5"] * coeffm4
            + X["month_6"] * coeffm5
            + X["month_7"] * coeffm6
            + X["month_8"] * coeffm7
            + X["month_9"] * coeffm8
            + X["month_10"] * coeffm9
            + X["month_11"] * coeffm10
            + X["month_12"] * coeffm11
        )

    def s_curve_hill_weekly(
        self,
        X,
        a,
        b,
        c,
        coeff1,
        coeff2,
        coeff3,
        coeff4,
        coeff5,
        coeff6,
    ):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a))
            + X["weekday_1"] * coeff1
            + X["weekday_2"] * coeff2
            + X["weekday_3"] * coeff3
            + X["weekday_4"] * coeff4
            + X["weekday_5"] * coeff5
            + X["weekday_6"] * coeff6
        )

    def s_curve_hill_decomp(
        self,
        X,
        a,
        b,
        c,
        coeff1,
        coeff2,
        coeff3,
        coeff4,
        coeff5,
        coeff6,
        coeffm1,
        coeffm2,
        coeffm3,
        coeffm4,
        coeffm5,
        coeffm6,
        coeffm7,
        coeffm8,
        coeffm9,
        coeffm10,
        coeffm11,
    ):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return (
            c * (X[metric] ** a / (X[metric] ** a + b**a)),
            X["weekday_1"] * coeff1
            + X["weekday_2"] * coeff2
            + X["weekday_3"] * coeff3
            + X["weekday_4"] * coeff4
            + X["weekday_5"] * coeff5
            + X["weekday_6"] * coeff6,
            
            X["month_2"] * coeffm1
            + X["month_3"] * coeffm2
            + X["month_4"] * coeffm3
            + X["month_5"] * coeffm4
            + X["month_6"] * coeffm5
            + X["month_7"] * coeffm6
            + X["month_8"] * coeffm7
            + X["month_9"] * coeffm8
            + X["month_10"] * coeffm9
            + X["month_11"] * coeffm10
            + X["month_12"] * coeffm11
        )

    def s_curve_hill_spend_comp(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        return c * (X[metric] ** a / (X[metric] ** a + b**a))

    def s_curve_hill_spend_imp(self, X, a, b, c):
        """This method performs the scurve function on param X and
        Returns the outcome as a varible called y"""
        return c * (X ** a / (X ** a + b ** a))

    def mape(self, actual, pred):
        """calculating mape
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: mape
        """
        return np.mean(abs(actual - pred) / actual)
    
    def RSE(self, actual, pred, nparam):
        """calculating Std. Error of Residual
        Args:
            actual (series): target
            pred (series): predicted target
            nparam: Number of parameters for calculating degrees of freedom
        Returns:
            float: RSE
        """
        
        RSS = np.sum(np.square(actual - pred))

        rse = math.sqrt(RSS / (len(actual) - nparam))
        return rse

    def SMAPE(self, actual, pred):
        """calculating Std. Error of Residual
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: SMAPE
        """

        smape = abs(actual-pred)/(actual + pred)
        smape = np.mean(smape[~smape.isna()])
        return smape

    def seas_check(self, df_sub):

        """to check weekly or monthly seasonality is applicable
        Returns:
            boolean: 0/1
        """
        weekly_seas = 0
        monthly_seas = 0

        df_count_week = df_sub["weekday"].value_counts()

        if df_count_week[df_count_week >= 2].shape[0] == 7:
            weekly_seas = 1
        else:
            weekly_seas = 0

        df_sub["year"] = df_sub["date"].dt.year

        l_count = []
        for i in df_sub.groupby("year")["month"].unique():
            l_count += list(i)

        df_count_month = pd.Series(l_count).value_counts()

        if df_count_month[df_count_month >= 2].shape[0] == 12:
            monthly_seas = 1
        else:
            monthly_seas = 0

        return weekly_seas, monthly_seas

    def fit_curve(self, drop_dimension):
        """function to fit the model and find their optimal parameters
        Args:
            drop_dimension (list): dimensions with very less data
        Returns:
            df_param(dataframe) : model parameter
            df_score(dataframe) : mape,r2,drop points
            prediction_df(dataframe) : predictions
            drop_dimension(list) : dimensions with very less data
        """

        if self.use_impression:
            metric = "impression"
        else:
            metric = "spend"

        params, score = {}, {}

        seas_drop = []

        prediction_df = pd.DataFrame()

        global progress_var

        weekly_seas_flag = [None] * int(len(self.df.dimension.unique()))

        for count, dim in enumerate(self.df.dimension.unique()):

            dim_df = self.df[self.df.dimension == dim].reset_index(drop=True)

            weekly_seas, monthly_seas = self.seas_check(dim_df)

            weekly_seas_flag[count] = weekly_seas

            if weekly_seas == 0:
                seas_drop.append(dim)
                continue

            temp_df = pd.DataFrame()

            bounds = (
                [
                    0.5,
                    dim_df[metric].quantile(0.3),
                    -np.inf,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    3,
                    dim_df[metric].quantile(1),
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
            )

            # Number of parameters for RSE: 3(s-curve) + 6(weekly) + 11(monthly)
            nparam = 20

            if monthly_seas == 0:
                bounds = (bounds[0][0:9], bounds[1][0:9])

                # Number of parameters for RSE: 3(s-curve) + 6(weekly)
                nparam = 9

            try:
                if monthly_seas == 1:
                    popt, pcov = curve_fit(
                        f=self.s_curve_hill,
                        xdata=dim_df,
                        ydata=dim_df["target"],
                        bounds=bounds,
                    )
                else:
                    popt, pcov = curve_fit(
                        f=self.s_curve_hill_weekly,
                        xdata=dim_df,
                        ydata=dim_df["target"],
                        bounds=bounds,
                    )

                    popt = list(popt) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                if sum(self.s_curve_hill_spend_comp(dim_df, *popt[0:3])) == 0:
                    drop_dimension.append(dim)
                
                spend_pred,weekday_pred,monthly_pred = self.s_curve_hill_decomp(dim_df, *popt)
                pred = self.s_curve_hill(dim_df, *popt)
                params[dim] = list(popt) + [dim_df[metric].median()] + [dim_df[metric].mean()]

                score[dim] = []
                # score[dim].append(r2_score(dim_df["target"], pred))
                # score[dim].append(self.RSE(dim_df["target"], pred, nparam))
                # score[dim].append(self.mape(dim_df["target"], pred))
                score[dim].append(self.SMAPE(dim_df['target'],pred))
                score[dim].append(pd.Series(pred).corr(dim_df["target"]))
                

                temp_df["date"] = dim_df.date
                temp_df['weekday_'] = temp_df['date'].dt.day_name()
                temp_df['month_'] = temp_df['date'].dt.month_name()


                temp_df["spend_prediction"] = spend_pred.round(decimals=2)
                temp_df['weekly_prediction'] = weekday_pred.round(decimals=2)
                temp_df['monthly_prediction'] = monthly_pred.round(decimals=2)
                temp_df["predictions"] = pred.round(decimals=2)

                temp_df["predictions"] = np.where(
                    temp_df["predictions"] > 0, temp_df["predictions"], 0
                )
                temp_df["dimension"] = dim

            except Exception as e:
                print(dim, " exception: ", e)
                drop_dimension.append(dim)

            prediction_df = pd.concat([temp_df, prediction_df],axis=0)

            progress_var[0] = count + 1

            print("progress_var ", progress_var)

        if weekly_seas_flag.count(0) == int(len(self.df.dimension.unique())):
            raise Exception("Seasonality not available in data")

        df_score = (
            pd.DataFrame(score)
            .T.reset_index()
            .rename(
                columns={"index": "dimension",  0: "SMAPE", 1: "correlation"}
            )
        )

        df_score[["SMAPE", "correlation"]] = df_score[
            ["SMAPE", "correlation"]
        ].round(decimals=2)

        df_param = (
            pd.DataFrame(params)
            .T.reset_index()
            .rename(
                columns={
                    "index": "dimension",
                    0: "param a",
                    1: "param b",
                    2: "param c",
                    3: "weekday 1",
                    4: "weekday 2",
                    5: "weekday 3",
                    6: "weekday 4",
                    7: "weekday 5",
                    8: "weekday 6",
                    9: "month 2",
                    10: "month 3",
                    11: "month 4",
                    12: "month 5",
                    13: "month 6",
                    14: "month 7",
                    15: "month 8",
                    16: "month 9",
                    17: "month 10",
                    18: "month 11",
                    19: "month 12",
                    20: "median spend",
                    21: "mean spend"
                }
            )
        )
        drop_dimension = drop_dimension + seas_drop

        return df_param, df_score, prediction_df, drop_dimension

    def data_filter(self):

        """filter data for selected date range"""

        self.df = self.df[~((self.df["spend"] == 0) & (self.df["target"] == 0))]

        # self.df = self.df[self.df.dimension.isin(self.dimension_value)]

        # self.df = self.df[
        #     (self.df.date >= self.date_range[0]) & (self.df.date <= self.date_range[1])
        # ]

    def param_adjust(self, df_param):

        """adjust parameter of seasonality
        Returns:
            df_param(dataframe) : model parameter
        """

        # df_param = df_param.set_index("dimension")

        # df_self_grp_sum = self.df.groupby("dimension").sum().reset_index()

        # df_self_grp_sum.columns = [i.replace("_", " ") for i in df_self_grp_sum.columns]

        df_param = df_param.set_index("dimension")

        df_temp = self.df
        df_temp = df_temp.drop('date', axis=1)
        df_self_grp_sum = df_temp.groupby("dimension").sum().reset_index()

        df_self_grp_sum.columns = [i.replace("_", " ") for i in df_self_grp_sum.columns]

        for dim in df_param.index:

            df_self_grp_sum_ = df_self_grp_sum[df_self_grp_sum["dimension"] == dim]

            for week in range(1, 7):
                weekday = "weekday " + str(week)

                if df_self_grp_sum_[weekday].values[0] == 0:
                    df_param.loc[dim, weekday] = 0

            for month_ in range(2, 13):
                month = "month " + str(month_)

                if df_self_grp_sum_[month].values[0] == 0:
                    df_param.loc[dim, month] = 0

        return df_param.reset_index()
    
    def roi_cpa_outlier_treatment(self, df):

        if self.use_impression:
            col = 'impression_predictions_rate'
            metric = 'impression'
        else:
            col = 'spend_predictions_rate'
            metric = 'spend'

        df_ = pd.DataFrame()
        
        for dim in df['dimension'].unique():
            df_temp = df[df["dimension"] == dim].reset_index(drop=True)
            df_temp_filter=df_temp[df_temp[col]!=-1]
            mean = np.mean(df_temp_filter[col])
            std = np.std(df_temp_filter[col])
            df_temp_filter[col] = np.where((abs((df_temp_filter[col] - mean) / std) > 3), -1, df_temp_filter[col])
            df_temp.drop(columns=[col], inplace=True)
            df_temp = df_temp.merge(df_temp_filter[['date', col]], on='date', how='left').sort_values(by='date').reset_index(drop=True)
            # df_ = df_.append(df_temp, ignore_index=True).reset_index(drop=True)
            df_ = pd.concat([df_, df_temp], ignore_index=True).reset_index(drop=True)
            df_=df_.fillna(-1)

        return df_

    def execute(self):

        """execute function
        Returns:
           df_param(dataframe) : parameter of the model
           df_score_final(dataframe) : quality metric of the curve
           scatter_plot_df(dataframe) : predicted value
           drop_dimension(array) : drop dimension
           d_cpm(dictionary) : cpm for each dimension
        """

        dimension_val = list(self.df.dimension.unique())

        self.data_filter()

        # df_spend_dis = self.df.groupby('dimension').agg({'spend':'sum'}).reset_index()

        # outlier_obj = outlier_treatment(self.df, self.use_impression)

        # self.df = outlier_obj.z_outlier()

        # df_spend_dis = self.df.groupby('dimension').spend.agg(['sum', 'median']).reset_index()
        # df_spend_dis.rename({'sum': 'spend', 'median': 'median spend'}, axis=1, inplace=True)

        if self.use_impression:
            df_spend_dis = self.df.groupby('dimension').agg(spend=('spend', 'sum'),
                                                            impression=('impression', 'sum'),
                                                            median_impression=('impression', 'median'),
                                                            mean_impression=('impression', 'mean'),
                                                            return_conv=('target', 'sum')).reset_index()
            df_spend_dis['cpm'] = df_spend_dis["spend"] * 1000 / df_spend_dis["impression"]
            df_spend_dis['median spend'] = (df_spend_dis["median_impression"] * df_spend_dis["cpm"]) / 1000
            df_spend_dis['mean spend'] = (df_spend_dis["mean_impression"] * df_spend_dis["cpm"]) / 1000
            df_spend_dis=df_spend_dis[['dimension', 'spend', 'median spend', 'mean spend', 'return_conv']]
            
            for i in list(set(dimension_val) - set(df_spend_dis['dimension'])):
        
                df_spend_dis.loc[-1] = [i,0,0,0,0]

                df_spend_dis.index = df_spend_dis.index + 1
        else:
            df_spend_dis = self.df.groupby('dimension').agg(spend=('spend', 'sum'),
                                                            median_spend=('spend', 'median'),
                                                            mean_spend=('spend', 'mean'),
                                                            return_conv=('target', 'sum')).reset_index()
            df_spend_dis.rename({'median_spend': 'median spend', 'mean_spend': 'mean spend'}, axis=1, inplace=True)
            
            for i in list(set(dimension_val) - set(df_spend_dis['dimension'])):
        
                df_spend_dis.loc[-1] = [i,0,0,0,0]

                df_spend_dis.index = df_spend_dis.index + 1

        # df_drop = outlier_obj.drop_points(self.df)
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d')
        self.df["weekday"] = self.df["date"].dt.weekday
        self.df["month"] = self.df["date"].dt.month

        self.df = pd.concat(
            [
                self.df,
                pd.get_dummies(self.df["weekday"], prefix="weekday"),
                pd.get_dummies(self.df["month"], prefix="month"),
            ],
            axis=1,
        )

        col_prsnt = [
            "month_1",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "weekday_0",
            "weekday_1",
            "weekday_2",
            "weekday_3",
            "weekday_4",
            "weekday_5",
            "weekday_6",
        ]

        for col in [p for p in col_prsnt if (p not in self.df.columns)]:
            self.df[col] = 0

        self.df.drop(columns=["month_1", "weekday_0"], inplace=True)

        drop_dimension = list(set(dimension_val) - set(self.df.dimension.unique()))

        df_param, df_score, prediction_df, drop_dimension = self.fit_curve(
            drop_dimension
        )

        # df_score_final = df_score.merge(df_drop, on=["dimension"], how="outer")
        df_score_final = df_score

        scatter_plot_df = self.df.merge(
            prediction_df, on=["date", "dimension"], how="inner"
        )
        scatter_plot_df[["spend", "target"]] = scatter_plot_df[
            ["spend", "target"]
        ].round(decimals=2)

        df_contri = (
            scatter_plot_df.groupby("dimension").agg({"spend": "sum"}).reset_index()
        )

        df_contri["spend_%"] = df_contri["spend"] / df_contri["spend"].sum()

        df_param = df_param.merge(
            df_contri[["dimension", "spend_%"]], on="dimension", how="left"
        )

        scatter_plot_df_chk = (
            scatter_plot_df.groupby("dimension")
            .agg({"predictions": "sum"})
            .reset_index()
        )

        if (
            len(
                list(
                    scatter_plot_df_chk[scatter_plot_df_chk["predictions"] == 0][
                        "dimension"
                    ]
                )
            )
            > 0
        ):

            for dim in scatter_plot_df_chk[scatter_plot_df_chk["predictions"] == 0][
                "dimension"
            ]:
                drop_dimension.append(dim)

            df_param = df_param[
                ~df_param["dimension"].isin(drop_dimension)
            ].reset_index(drop=True)

        scatter_plot_df = scatter_plot_df[
            ~scatter_plot_df["dimension"].isin(drop_dimension)
        ].reset_index(drop=True)

        if self.use_impression == False:
            if self.target_type == "revenue":
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["predictions"]/scatter_plot_df["spend"]).round(decimals=2)
            else:
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["spend"]/scatter_plot_df["predictions"]).round(decimals=2)
            scatter_plot_df["spend_predictions_rate"] = scatter_plot_df["spend_predictions_rate"].replace([np.inf, -np.inf, np.nan], 0)

        self.df_param = df_param

        # df_score_final.drop(columns=["MAPE"], inplace=True)

        if self.use_impression:
            scatter_plot_df["impression_predictions_rate"] = (scatter_plot_df["predictions"]/(scatter_plot_df["impression"]/1000)).round(decimals=2)
            scatter_plot_df["impression_predictions_rate"] = scatter_plot_df["impression_predictions_rate"].replace([np.inf, -np.inf, np.nan], -1)
            scatter_plot_df = self.roi_cpa_outlier_treatment(scatter_plot_df)
            
            scatter_plot_df = scatter_plot_df[['date', 'spend', 'impression', 'target', 'dimension', 'weekday_',
                            'month_', 'spend_prediction', 'weekly_prediction', 'monthly_prediction',
                            'predictions', 'impression_predictions_rate']]

            df_cpm = (
                self.df.groupby("dimension").agg(
                    {"spend": "sum", "impression": [np.sum, np.median, np.mean]}
                )
            ).reset_index()

            df_cpm.columns = ["dimension", "spend", "impression", "impression_median", "impression_mean"]

            df_cpm["cpm"] = df_cpm["spend"] * 1000 / df_cpm["impression"]

            d_cpm = df_cpm[["dimension", "cpm"]].set_index("dimension").to_dict()["cpm"]

            df_param = df_param.merge(
                df_cpm[["dimension", "cpm", "impression_median", "impression_mean"]],
                on=["dimension"],
                how="left",
            )

        else:
            if self.target_type == "revenue":
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["predictions"]/scatter_plot_df["spend"]).round(decimals=2)
            else:
                scatter_plot_df["spend_predictions_rate"] = (scatter_plot_df["spend"]/scatter_plot_df["predictions"]).round(decimals=2)
            scatter_plot_df["spend_predictions_rate"] = scatter_plot_df["spend_predictions_rate"].replace([np.inf, -np.inf, np.nan], -1)
            scatter_plot_df = self.roi_cpa_outlier_treatment(scatter_plot_df)

            scatter_plot_df = scatter_plot_df[['date', 'spend', 'target', 'dimension', 'weekday_',
                            'month_', 'spend_prediction', 'weekly_prediction', 'monthly_prediction',
                            'predictions', 'spend_predictions_rate']]

        df_param = self.param_adjust(df_param)
        self.df_param = df_param

        median_mean_dic     = {}
        for dim in self.df_param['dimension'].unique():
            if self.use_impression:
                median_var = 'impression_median'
                mean_var = 'impression_mean'
            else:
                median_var = 'median spend'
                mean_var = 'mean spend'
            param_ = self.df_param[self.df_param["dimension"] == dim]
            weekday_ = [0, param_["weekday 1"].values[0],
                        param_["weekday 2"].values[0],
                        param_["weekday 3"].values[0],
                        param_["weekday 4"].values[0],
                        param_["weekday 5"].values[0],
                        param_["weekday 6"].values[0]]
            month_ = [0, param_["month 2"].values[0],
                        param_["month 3"].values[0],
                        param_["month 4"].values[0],
                        param_["month 5"].values[0],
                        param_["month 6"].values[0],
                        param_["month 7"].values[0],
                        param_["month 8"].values[0],
                        param_["month 9"].values[0],
                        param_["month 10"].values[0],
                        param_["month 11"].values[0],
                        param_["month 12"].values[0]]
            metric_median = self.df_param[self.df_param['dimension']==dim][median_var].values[0]
            metric_mean = self.df_param[self.df_param['dimension']==dim][mean_var].values[0]
            median_mean_dic[dim] = {'median': [round(metric_median, 2),
                                                round(self.s_curve_hill_spend_imp(
                                                    metric_median, 
                                                    param_["param a"].values[0], 
                                                    param_["param b"].values[0], 
                                                    param_["param c"].values[0])
                                                    + np.median(weekday_)
                                                    + np.median(month_), 2)],
                                    'mean': [round(metric_mean, 2), 
                                                round(self.s_curve_hill_spend_imp(
                                                    metric_mean, 
                                                    param_["param a"].values[0], 
                                                    param_["param b"].values[0], 
                                                    param_["param c"].values[0])
                                                    + np.mean(weekday_) 
                                                    + np.mean(month_), 2)]}
            
        if self.use_impression:
            return (
                df_param,
                df_score_final,
                scatter_plot_df,
                drop_dimension,
                d_cpm,
                df_spend_dis,
                median_mean_dic
            )
        else:
            return df_param, df_score_final, scatter_plot_df, drop_dimension, df_spend_dis, median_mean_dic

def s_curve_hill_with_seasonality(
    X,
    a,
    b,
    c,
    coeff1,
    coeff2,
    coeff3,
    coeff4,
    coeff5,
    coeff6,
    coeffm1,
    coeffm2,
    coeffm3,
    coeffm4,
    coeffm5,
    coeffm6,
    coeffm7,
    coeffm8,
    coeffm9,
    coeffm10,
    coeffm11,
    weekday_a,
    month_a,
):
    """This method performs the scurve function on param X and
    Returns the outcome as a varible called y"""
    return (
        c * (X ** a / (X ** a + b ** a))
        + coeff1 * weekday_a[0]
        + coeff2 * weekday_a[1]
        + coeff3 * weekday_a[2]
        + coeff4 * weekday_a[3]
        + coeff5 * weekday_a[4]
        + coeff6 * weekday_a[5]
        + coeffm1 * month_a[0]
        + coeffm2 * month_a[1]
        + coeffm3 * month_a[2]
        + coeffm4 * month_a[3]
        + coeffm5 * month_a[4]
        + coeffm6 * month_a[5]
        + coeffm7 * month_a[6]
        + coeffm8 * month_a[7]
        + coeffm9 * month_a[8]
        + coeffm10 * month_a[9]
        + coeffm11 * month_a[10]
    )


def predict_dimesion_with_seasonality(
    df_param, dimension, date_range, budget, cpm=None
):

    """function to predict target
     Args:
        df_param (dataframe) : model param dataframe
        dimension (string): dimensions for which predict run
        date_range (array) : start and end date
        budget(int) : budget
        cpm (float) : when use impression is true
    Returns:
        float: target
    """

    days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days + 1

    budget_per_day = budget / days

    param = df_param[df_param["dimension"] == dimension]

    init_weekday = [0, 0, 0, 0, 0, 0]
    init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    total_rev = []

    for day_ in pd.date_range(date_range[0], date_range[1], inclusive="both"):

        if "cpm" in df_param.columns:
            x_var = (budget_per_day * 1000) / cpm
        else:
            x_var = budget_per_day

        if day_.weekday() != 0:
            init_weekday[day_.weekday() - 1] = 1

        if day_.month != 1:
            init_month[day_.month - 2] = 1

        rev = s_curve_hill_with_seasonality(
            x_var,
            param["param a"].values[0],
            param["param b"].values[0],
            param["param c"].values[0],
            param["weekday 1"].values[0],
            param["weekday 2"].values[0],
            param["weekday 3"].values[0],
            param["weekday 4"].values[0],
            param["weekday 5"].values[0],
            param["weekday 6"].values[0],
            param["month 2"].values[0],
            param["month 3"].values[0],
            param["month 4"].values[0],
            param["month 5"].values[0],
            param["month 6"].values[0],
            param["month 7"].values[0],
            param["month 8"].values[0],
            param["month 9"].values[0],
            param["month 10"].values[0],
            param["month 11"].values[0],
            param["month 12"].values[0],
            init_weekday,
            init_month,
        )
        if rev < 0:
            rev = 0
        total_rev.append(rev)

        init_weekday = [0, 0, 0, 0, 0, 0]
        init_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return int(sum(total_rev))


def progress_bar_var():
    global progress_var
    return progress_var

@cache.memoize(timeout=600)
def get_utils_from_predictor_instance(location_id):
    df = get_query_res(GET_COMPLETE_DATA_FOR_ID.format(location_id=location_id)).to_dataframe()
    dfnew = df[['monthdate', 'Channel', 'Spend', 'TotalLeads']]
    dfnew.rename(columns = {'monthdate': 'date', 'Channel': 'dimension', 'Spend':'spend', 'TotalLeads':'target'},inplace = True)

    is_monthly_selected = True
    target_type = 'Acquisitions'

    predict_obj = predictor_with_seasonality(dfnew, target_type)
    df_param,df_score_final,scatter_plot_df,drop_dimension,df_spend_dis,median_mean_dic = predict_obj.execute()

    return scatter_plot_df, df_param, df_score_final, df_spend_dis, dfnew

def s_curve_hill(X, a, b, c):
    """This method performs the scurve function on param X and
    Returns the outcome as a variable called y"""
    return c * (X**a / (X**a + b**a))

def generate_response_curves_data(df_param):
    df_param_opt = df_param.T
    df_param_opt.columns = df_param_opt.iloc[0, :]
    d_param = df_param_opt.iloc[1:, :].to_dict()

    # Create dictionaries to store the results for each dimension
    response_curves_data = {
        'display': [],
        'search': [],
        'social': []
    }

    dim_ = {}
    for dim in df_param['dimension'].unique():
        dim_[dim] = {}
        dim_[dim]['spend'] = []
        dim_[dim]['rev'] = []
        spend_ = []
        rev_ = []
        i = 0
        while i <= 80000:
            temp = s_curve_hill(i, d_param[dim]["param a"], d_param[dim]["param b"], d_param[dim]["param c"])
            rev_.append(temp)
            spend_.append(i)
            i += 100

        # Store the spend and rev lists for the current dimension in the dictionary
        dim_[dim]['spend'] = spend_
        dim_[dim]['rev'] = rev_

        # Create a list to store the data for the current dimension
        dimension_data = [{'predictedLeads': pred, 'spend': sp} for pred, sp in zip(rev_, spend_)]

        # Append the data to the corresponding dimension in the main dictionary
        if dim == 'DisplaySpend':
            response_curves_data['display'] = dimension_data
        elif dim == 'SearchSpend':
            response_curves_data['search'] = dimension_data
        elif dim == 'SocialSpend':
            response_curves_data['social'] = dimension_data

       
    # Create the final results dictionary
    results_dict = {'responseCurvesData': response_curves_data}

    return results_dict

def generate_area_chart_data(scatter_plot_df):
    # Splitting the Display Data
    df_scatter_plot_disp = scatter_plot_df[scatter_plot_df['dimension'] == 'DisplaySpend']
    df_scatter_plot_disp.rename(columns={'spend': 'Display_Spend', 'target': 'Display_Target', 'predictions': 'Display_Predictions'}, inplace=True)
    df_scatter_plot_disp = df_scatter_plot_disp.drop("dimension", axis='columns')
    df_scatter_plot_disp.fillna(0, inplace=True)

    # Splitting the Social Data
    df_scatter_plot_soc = scatter_plot_df[scatter_plot_df['dimension'] == 'SocialSpend']
    df_scatter_plot_soc.rename(columns={'spend': 'Social_Spend', 'target': 'Social_Target', 'predictions': 'Social_Predictions'}, inplace=True)
    df_scatter_plot_soc = df_scatter_plot_soc.drop("dimension", axis='columns')
    df_scatter_plot_soc.fillna(0, inplace=True)
    
    # Splitting the Search Data
    df_scatter_plot_ser = scatter_plot_df[scatter_plot_df['dimension'] == 'SearchSpend']
    df_scatter_plot_ser.rename(columns={'spend': 'Search_Spend', 'target': 'Search_Target', 'predictions': 'Search_Predictions'}, inplace=True)
    df_scatter_plot_ser = df_scatter_plot_ser.drop("dimension", axis='columns')
    df_scatter_plot_ser.fillna(0, inplace=True)
    
    # Joining the 3 Separate data frames
    df_join = pd.merge(df_scatter_plot_ser, df_scatter_plot_disp, on='date', how='left')
    df_join = pd.merge(df_join, df_scatter_plot_soc, on='date', how='left')

    # Filling the null values
    df_join.fillna(0, inplace=True)

    # Dropping the unwanted columns
    df_join.drop(['Display_Spend', 'Display_Target','weekday__x', 'month__x','spend_predictions_rate_x',
                  'weekly_prediction_x', 'monthly_prediction_x','spend_prediction_x','Search_Spend', 'Search_Target',
                  'weekday__y', 'month__y', 'spend_prediction_y','weekly_prediction_y', 'monthly_prediction_y',
                 'spend_predictions_rate_y', 'Social_Spend','Social_Target','weekday_',
                  'month_', 'spend_prediction', 'weekly_prediction', 'monthly_prediction','spend_predictions_rate'], axis=1, inplace=True)

    df_join['Total_Predictions'] = df_join['Display_Predictions'] + df_join['Search_Predictions'] + df_join['Social_Predictions']

    df_join['Disp%'] = df_join['Display_Predictions'] / df_join['Total_Predictions']  
    df_join['Ser%'] = df_join['Disp%'] + (df_join['Search_Predictions'] / df_join['Total_Predictions'])
    df_join['Soc%'] = df_join['Ser%'] + (df_join['Social_Predictions'] / df_join['Total_Predictions'])

    df_join_new = df_join[['date', 'Disp%', 'Ser%', 'Soc%']]

    Area_Chart = {
        'channelSplitAreaChartData': {
            'monthdate': df_join_new['date'].dt.strftime('%Y-%m').tolist(),
            'displayPercentage': (df_join_new['Disp%']).tolist(),
            'socialPercentage': (df_join_new['Soc%']).tolist(),
            'searchPercentage': (df_join_new['Ser%']).tolist(),
        }
    }

    return Area_Chart

def format_data_to_predict_table(df_score_final):

    predict_table_data = {
        'dimension': df_score_final['dimension'].tolist(),
        'SMAPE': (df_score_final['SMAPE'] * 100).astype(int).tolist(),  # Convert SMAPE to percentage
        'correlation': df_score_final['correlation'].tolist()
    }
    return predict_table_data

def generate_combined_data(scatter_plot_df, df_param, df_score_final):
    # Generate Area Chart data
    area_chart_data = generate_area_chart_data(scatter_plot_df)
    
    # Generate Response Curves data
    response_curves_data = generate_response_curves_data(df_param)

    # Predict Table data 
    predict_table_data = format_data_to_predict_table(df_score_final)
    
    # Combine the data into a single dictionary
    combined_data = {
        'channelSplitAreaChartData': area_chart_data['channelSplitAreaChartData'],
        'responseCurvesData': response_curves_data['responseCurvesData'],
        'predictTableData':predict_table_data,
    }
    
    return combined_data