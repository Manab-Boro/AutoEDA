import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
from PIL import Image



def __get_segment_arrays(colname_list, rowsize):
    colname_list= colname_list
    nn= 0
    outer_list= colname_list[rowsize*nn : rowsize*(nn+1)]

    # no_of_elements= 0
    temp_3= []
    while True:
        # no_of_elements+= 1
        n= 0
        inner_list= colname_list[rowsize*n : rowsize*(n+1)]
        temp_2= []
        while True:
            temp= []
            for c in outer_list:
                for r in inner_list:
                    temp.append((c,r))
            temp_2.append(temp)
            if inner_list[-1]== colname_list[-1]:
                break
            n+=1
            inner_list= colname_list[rowsize*n : rowsize*(n+1)]
        
        temp_3.append(temp_2)
        if outer_list[-1]== colname_list[-1]:
            break
        nn+=1
        outer_list= colname_list[rowsize*nn : rowsize*(nn+1)]

    return temp_3   #, no_of_elements


def __get_unique_values(tuples_list):
    unique_first_elements = OrderedDict()
    unique_last_elements = OrderedDict()

    for t in tuples_list:
        unique_first_elements[t[0]]= None
        unique_last_elements[t[1]]= None
    return list(unique_first_elements), list(unique_last_elements)


def get_segment_arrays(colname_list, rowsize):
    temp_2= []
    m= 0
    outer_list= colname_list[rowsize*m : rowsize*(m+1)]
    while True:
        temp= []
        n= 0
        inner_list= colname_list[rowsize*n : rowsize*(n+1)]
        while True:
            temp.append([outer_list, inner_list])
            if inner_list[-1]== colname_list[-1]:
                break
            n+= 1
            inner_list= colname_list[rowsize*n : rowsize*(n+1)]
        temp_2.append(temp)
        if outer_list[-1]== colname_list[-1]:
            break
        m+= 1
        outer_list= colname_list[rowsize*m : rowsize*(m+1)]
    return temp_2



class PlotGraphs():
    def __init__(self, df):
        self.df= df
        self.floating_point_limit= 0
        self.floating_point_limit_for_percentage= 0
        self.num_rows= 2
        self.barplot_color= "purple"
        self.boxplot_color= "red"
        self.countplot_color= "red"
        self.img_format= "png"
        self.count_plot_use_bean= False
        self.count_plot_num_bean= 15

    def plot_barplot(self, colname, num_rows, img_file_path):
        pdseries_df= self.df[colname]
        top_freq= pdseries_df.sort_index(ascending= False).value_counts(dropna= False).nlargest(num_rows).reset_index()
        top_freq["Count_Percentage"]= (top_freq["count"]/pdseries_df.shape[0])*100

        others_count= pdseries_df.shape[0]-top_freq["count"].sum()
        if others_count> 0:
            top_freq[pdseries_df.name]= top_freq[pdseries_df.name].astype(str)
            top_freq.loc[len(top_freq)]= ["others...", others_count, (100-top_freq["Count_Percentage"].sum()).round(self.floating_point_limit_for_percentage)]
        
        bar_fig= plt.figure(figsize=(15,5))
        sns.barplot(data= top_freq, x= colname, y="count", color= self.barplot_color)
        # plt.xticks(rotation= 45)
        plt.savefig(os.path.join(img_file_path, colname+"." +self.img_format), bbox_inches='tight')
        plt.close(bar_fig)
    
    def plot_boxplot(self, colname, img_file_path):
        box_fig= plt.figure(figsize=(12,5))
        sns.boxplot(x= self.df[colname], color= self.boxplot_color)
        plt.savefig(os.path.join(img_file_path, colname+"." +self.img_format), bbox_inches='tight')
        plt.close(box_fig)

    def plot_countplot(self, colname, img_file_path):
        if self.count_plot_use_bean:
            bean_df= pd.DataFrame()
            bean_df[colname]= self.df[colname].copy(deep=True)
            count_plot_no_of_beans= 15

            bean_df["label"]= pd.cut(self.df[colname], bins= count_plot_no_of_beans, labels= [f'label_{i}' for i in range(1, count_plot_no_of_beans+1)])
            countplot_df= bean_df.groupby("label", observed=False ).agg(sum= (colname, "sum"), count= (colname, "count"), square_sum= (colname, lambda x: sum(x*x))).reset_index()
            countplot_df.columns = [''.join(col).strip() for col in countplot_df.columns.values]
            countplot_df["square_sum_avg"]= countplot_df["square_sum"]/countplot_df["count"]

            count_fig= plt.figure(figsize=(17,14))
            sns.barplot(data=countplot_df, x= "square_sum_avg", y= "count", color= self.countplot_color)
            plt.xticks(rotation= 45)
            plt.savefig(os.path.join(img_file_path, colname+"." +self.img_format), bbox_inches='tight')
            plt.close(count_fig)
        else:
            count_fig= plt.figure(figsize=(17,14))
            sns.countplot(x= self.df[colname],color= self.countplot_color)
            plt.xticks(rotation= 45)
            plt.savefig(os.path.join(img_file_path, colname+"." +self.img_format), bbox_inches='tight')
            plt.close(count_fig)

    def __plot_corr_heatmap(self, columns, img_file_path):
        rowsize= 5
        temp_df= self.df[columns]

        corr_data= temp_df.apply(lambda x: x.factorize()[0]).corr()
        vmax= corr_data.max(axis=1).max(axis=0)
        vmin= corr_data.min(axis=1).max(axis=0)
        
        
        list_of_col_mapping= __get_segment_arrays(columns, rowsize)
        
        i= 1
        for cols in list_of_col_mapping:
            for rows in cols:
                vertical_cols, hori_cols= __get_unique_values(rows)
                temp= corr_data.loc[vertical_cols][hori_cols]
                fig= plt.figure(figsize=(len(hori_cols), len(vertical_cols)))
                sns.heatmap(temp, cmap="YlGnBu", annot=True,  linewidth= 5, vmax= vmax, vmin= vmin, cbar= False) 
                plt.xticks(rotation=45,fontsize=18)
                plt.yticks(rotation=45,fontsize=18)
                plt.savefig(os.path.join(img_file_path, f"{i}.{self.img_format}"), bbox_inches='tight')
                plt.close(fig)
                i+= 1

    def plot_corr_heatmap(self, columns, img_file_path, rowsize, cmap= "YlGnBu"):
        temp_df= self.df[columns]

        corr_data= temp_df.apply(lambda x: x.factorize()[0]).corr()
        vmax= corr_data.max(axis=1).max(axis=0)
        vmin= corr_data.min(axis=1).min(axis=0)
        
        list_of_col_mapping= get_segment_arrays(columns, rowsize)
        fig= plt.figure(figsize=(20,25))
        sns.heatmap(pd.DataFrame(), cmap="YlGnBu", vmax= vmax, vmin= vmin, cbar= True)
        plt.savefig(os.path.join(img_file_path, f"0.{self.img_format}"), bbox_inches='tight')
        plt.close()
        im = Image.open(os.path.join(img_file_path, f"0.{self.img_format}"))
        width, height = im.size
        im1 = im.crop((1300, 0, width, height))
        im1.save(os.path.join(img_file_path, f"0.{self.img_format}"))
        im1.close()

        i= 1
        annote_cbar= False
        for cols in list_of_col_mapping:
            if len(cols)== 1:
                annote_cbar= True
            for rows in cols:
                vertical_cols, hori_cols= rows
                temp= corr_data.loc[vertical_cols][hori_cols]
                fig= plt.figure(figsize=(len(hori_cols), len(vertical_cols)))
                sns.heatmap(temp, cmap=cmap, annot=True,  linewidth= 5, vmax= vmax, vmin= vmin, cbar= annote_cbar) 
                plt.xticks(rotation=45,fontsize=18)
                plt.yticks(rotation=45,fontsize=18)
                plt.savefig(os.path.join(img_file_path, f"{i}.{self.img_format}"), bbox_inches='tight')
                plt.close(fig)
                i+= 1


class FeatureStatsExtraction():
    def __init__(self, df):
        self.df= df
        self.floating_point_limit= 0
        self.floating_point_limit_for_percentage= 0

    def get_desc_stats(self, colname, full_path):
        pdseries_df= self.df[colname]
        desc_stats= pd.DataFrame({
            "Values": [pdseries_df.count()],
            "Unique": [pdseries_df.nunique()],
            "Missing": [pdseries_df.isnull().sum()],
            })
        desc_stats["Distinct"]= pdseries_df.nunique()

        desc_stats= desc_stats.round(self.floating_point_limit)
        
        desc_stats["Missing_Percentage"]= ((desc_stats["Missing"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Distinct_Percentage"]= ((desc_stats["Distinct"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Values_Percentage"]= (100- desc_stats["Missing_Percentage"]).round(self.floating_point_limit_for_percentage)
        
        self.dump_df_to_pickle(desc_stats, colname, full_path)

    def get_top_freq(self, colname, num_rows, full_path):
        pdseries_df= self.df[colname]
        top_freq= pdseries_df.sort_index(ascending= False).value_counts(dropna= False).nlargest(num_rows).reset_index()
        top_freq["Count_Percentage"]= ((top_freq["count"]/pdseries_df.shape[0])*100).round(self.floating_point_limit_for_percentage)

        others_count= pdseries_df.shape[0]-top_freq["count"].sum()
        if others_count> 0:
            top_freq[pdseries_df.name]= top_freq[pdseries_df.name].astype(str)
            top_freq= top_freq._append(pd.DataFrame({
                    colname: "others",
                    "count": others_count,
                    "Count_Percentage": (100-top_freq["Count_Percentage"].sum()).round(self.floating_point_limit_for_percentage)
                }, index= [0]), ignore_index = True)

        self.dump_df_to_pickle(top_freq, colname, full_path)

    def get_top_smallest(self,colname, num_rows, full_path):
        pdseries_df= self.df[colname]
        smallest_val= pdseries_df.value_counts( dropna= False).reset_index().sort_values(by= [pdseries_df.name]).nsmallest(num_rows, columns= [pdseries_df.name])
        smallest_val["Count_Percentage"]= ((smallest_val["count"]/pdseries_df.shape[0])*100).round(self.floating_point_limit_for_percentage)

        others_count= pdseries_df.shape[0]-smallest_val["count"].sum()
        if others_count> 0:
            smallest_val[pdseries_df.name]= smallest_val[pdseries_df.name].astype(str)
            smallest_val= smallest_val._append(pd.DataFrame({
                    colname: "others",
                    "count": others_count,
                    "Count_Percentage": (100-smallest_val["Count_Percentage"].sum()).round(self.floating_point_limit_for_percentage)
                }, index= [0]), ignore_index = True)

        self.dump_df_to_pickle(smallest_val, colname, full_path)

    def get_top_largest(self,colname, num_rows, full_path):
        pdseries_df= self.df[colname]
        lagest_val= pdseries_df.value_counts( dropna= False).reset_index().sort_values(by= [pdseries_df.name]).nlargest(num_rows, columns= [pdseries_df.name])
        lagest_val["Count_Percentage"]= ((lagest_val["count"]/pdseries_df.shape[0])*100).round(self.floating_point_limit_for_percentage)
        
        others_count= pdseries_df.shape[0]-lagest_val["count"].sum()
        if others_count> 0:
            lagest_val[pdseries_df.name]= lagest_val[pdseries_df.name].astype(str)
            lagest_val= lagest_val._append(pd.DataFrame({
                    colname: "others",
                    "count": others_count,
                    "Count_Percentage": (100-lagest_val["Count_Percentage"].sum()).round(self.floating_point_limit_for_percentage)
                }, index= [0]), ignore_index = True)
        
        self.dump_df_to_pickle(lagest_val, colname, full_path)

    def dump_df_to_pickle(self,data, filename, full_path):
        pfile = open(os.path.join(full_path, filename), 'ab')
        pickle.dump(data, pfile)                    
        pfile.close()


class NumericFeatureStatsExtraction(FeatureStatsExtraction):
    def __init__(self, df):
        self.df= df
        self.floating_point_limit= 3
        self.floating_point_limit_for_percentage= 3

    def get_desc_stats(self, colname, full_path):
        pdseries_df= self.df[colname]
        desc_stats= pdseries_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame().T

        desc_stats["sum"]= np.sum(pdseries_df, axis=0)
        desc_stats["skew"]= skew(pdseries_df, axis=0, bias=False, nan_policy='omit')
        desc_stats["kurtosis"]= kurtosis(pdseries_df, axis=0, bias=False, nan_policy='omit')
        desc_stats["var"]= np.var(pdseries_df, axis=0)
        desc_stats["Range"]= desc_stats["max"]- desc_stats["min"]
        desc_stats["IQR"]= desc_stats["75%"]- desc_stats["25%"]
        desc_stats["Upper_Bound"]= desc_stats["75%"]+ (1.5* desc_stats["IQR"])
        desc_stats["Lower_Bound"]= desc_stats["25%"]- (1.5* desc_stats["IQR"])
        desc_stats["Missing"]= pdseries_df.isnull().sum()
        desc_stats["Distinct"]= pdseries_df.nunique()
        desc_stats["count"]= desc_stats["count"].astype(np.int64)
        desc_stats["Zeros"]= pdseries_df[pdseries_df==0].count()
        
        desc_stats= desc_stats.round(self.floating_point_limit)
    
        gt_upper= 0
        lt_lower= 0

        gt_upper= pdseries_df.gt(desc_stats["Upper_Bound"].iloc[0]).sum()
        lt_lower= pdseries_df.lt(desc_stats["Lower_Bound"].iloc[0]).sum()

        desc_stats["Gt_Upper"]= gt_upper
        desc_stats["Lt_Lower"]= lt_lower

        desc_stats["Gt_Upper_Percentage"]= ((desc_stats["Gt_Upper"]/desc_stats["count"])* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Lt_Lower_Percentage"]= ((desc_stats["Lt_Lower"]/desc_stats["count"])* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Zeros_Percentage"]= ((desc_stats["Zeros"]/(desc_stats["count"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Missing_Percentage"]= ((desc_stats["Missing"]/(desc_stats["count"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Values_Percentage"]= (100- desc_stats["Missing_Percentage"]).round(self.floating_point_limit_for_percentage)
        desc_stats["Distinct_Percentage"]= ((desc_stats["Distinct"]/(desc_stats["count"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        
        desc_stats.rename(columns=
                        {
                            "count": "Values", 
                            "mean": "Avg",
                            "std": "Std",
                            "min": "Min",
                            "max": "Max",
                            "sum": "Sum",
                            "skew": "Skew",
                            "kurtosis": "Kurtosis",
                            "var": "Var"
                        }, inplace= True
                    )
        super().dump_df_to_pickle(desc_stats, colname, full_path)
    
    def get_perr_corr(self, colname, full_path):
        pear_corr= self.df.select_dtypes(include= np.number).corr('pearson')[colname]
        pear_corr_val= pd.DataFrame({
            "Colname": pear_corr.index,
            "Value": pear_corr
        })

        super().dump_df_to_pickle(pear_corr_val, colname, full_path)

    
class DateTimeFeatureStatsExtraction(FeatureStatsExtraction):
    def __init__(self, df):
        self.df= df
        self.floating_point_limit= 3
        self.floating_point_limit_for_percentage= 3

    def get_desc_stats(self, colname, full_path):
        pdseries_df= self.df[colname]
        desc_stats= pd.DataFrame({
            "Values": [pdseries_df.count()],
            "Unique": [pdseries_df.nunique()],
            "Start": [pdseries_df.min()],
            "End": [pdseries_df.max()],
            "Missing": [pdseries_df.isnull().sum()],
            })
        
        desc_stats["Distinct"]= pdseries_df.nunique()

        desc_stats= desc_stats.round(self.floating_point_limit)
         
        desc_stats["Missing_Percentage"]= ((desc_stats["Missing"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Distinct_Percentage"]= ((desc_stats["Distinct"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100).round(self.floating_point_limit_for_percentage)
        desc_stats["Values_Percentage"]= (100- desc_stats["Missing_Percentage"]).round(self.floating_point_limit_for_percentage)
        
        if False:
            std_df= pd.DataFrame({"Time_1":self.df[colname].sort_values()[1:].to_list(), "Time_2": self.df[colname].sort_values()[:-1].to_list()})
            std_df["Time_diff"]= (std_df["Time_1"]- std_df["Time_2"])
            std_df["Time_diff_in_sec"]= (std_df["Time_1"]- std_df["Time_2"]).dt.total_seconds()
            desc_stats["Std"]= str(std_df["Time_diff_in_sec"].std().round(floating_point_limit))+ " sec"
        else:
            desc_stats["Std"]= None

        super().dump_df_to_pickle(desc_stats, colname, full_path)




if __name__ == '__main__':
    pass

