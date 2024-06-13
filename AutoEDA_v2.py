import pandas as pd
import numpy as np
import os
import warnings
from multiprocessing import Pool
from collections import Counter
import datetime
from multiprocessing.pool import ThreadPool
import shutil
from AutoEDA_include.FeatureStatsExtraction import FeatureStatsExtraction, NumericFeatureStatsExtraction, DateTimeFeatureStatsExtraction, PlotGraphs
from AutoEDA_include.HTMLWraper import HTMLWraper
import json


class AutoEDA():
    def __init__(self, df):
        self.df= df
        self.configs()
        self.create_paths()
        self.df_feature_types= self.__change_datatypes()

    def configs(self):
        self.corr_heatmap_pics_per_row= 20
        self.corr_heatmap_datatype= [np.number, 'datetime64', 'category', 'bool']

        self.floating_point_limit= 3
        self.floating_point_limit_for_percentage= 2
        self.cat_threshold= 10
        self.cat_threshold_percentage= 0.25
        self.no_of_rows_right_small_tables= 12
        self.no_of_rows_left_top_freq_table= 7
        self.no_of_rows_right_tables= 48

        self.img_format= "png"
        self.boxplot_color= "red"  #for numeric
        self.countplot_color= "blue"    #for numeric
        self.cat_barplot_color= "cyan"
        self.text_barplot_color= "green"
        self.dt_barplot_color= "orange"
        self.others_barplot_color= "purple"
        self.corr_heatmap_cmap= "YlGnBu"

        self.count_plot_use_bean= True
        self.count_plot_num_bean= 15

    def write_config_for_html(self, htmlpath, standalone_html):
        conf_file= open(self.config_file, "w")
        contents= {
            "img_format": self.img_format,

            "dump_path": self.temp_path,
            "graph_path": self.graph_path,
            "corr_heatmap_path": self.corr_heatmap_graph_path,
            "desc_stats_path": self.desc_stats_path,
            "perr_corr_path": self.perr_corr_path,
            "top_largest_path": self.top_largest_path,
            "top_smallest_path": self.top_smallest_path,
            "top_freq_path": self.top_freq_path,
            "left_top_freq_path": self.left_top_freq_path,
            "boxplot_path": self.boxplot_path,
            "countplot_path": self.countplot_path,
            "barplot_path": self.barplot_path,

            "corr_heatmap_datatype": [str(i) for i in self.corr_heatmap_datatype],
            "df_feature_types": self.df_feature_types,

            "html_file": htmlpath,
            "standalone": standalone_html,
            "page_overview": 'This is an auto-generated HTML page demo.<br>The scope of this project is to analyze exploratory data and do some basic preprocessing autometically. Find the GitHub Repo <a href="https://github.com/manab36/AutoEDA">here.</a>'
        }
        json.dump(contents, conf_file, indent = 6) 
        conf_file.close()

    def create_paths(self):
        self.temp_path= os.path.join(os.getcwd(), "AutoEDA_temp")
        self.graph_path= os.path.join(os.getcwd(), "AutoEDA_graph")

        
        self.corr_heatmap_graph_path= os.path.join(self.graph_path, "corr_heatmap")

        self.config_file= os.path.join(self.temp_path, "html_config.json")
        
        self.desc_stats_path= os.path.join(self.temp_path, "desc_stats")
        self.perr_corr_path= os.path.join(self.temp_path, "perr_corr")
        self.top_largest_path= os.path.join(self.temp_path, "top_largest")
        self.top_smallest_path= os.path.join(self.temp_path, "top_smallest")
        self.top_freq_path= os.path.join(self.temp_path, "top_freq")
        self.left_top_freq_path= os.path.join(self.temp_path, "left_top_freq")

        self.boxplot_path= os.path.join(self.graph_path, "boxplot")
        self.countplot_path= os.path.join(self.graph_path, "countplot")
        self.barplot_path= os.path.join(self.graph_path, "barplot")

        if os.path.exists(self.temp_path):
            shutil.rmtree(self.temp_path)
        os.mkdir(self.temp_path)
        if os.path.exists(self.graph_path):
            shutil.rmtree(self.graph_path)
        os.mkdir(self.graph_path)

        if not os.path.exists(self.corr_heatmap_graph_path):
            os.mkdir(self.corr_heatmap_graph_path)
        if not os.path.exists(self.desc_stats_path):
            os.mkdir(self.desc_stats_path)
        if not os.path.exists(self.perr_corr_path):
            os.mkdir(self.perr_corr_path)
        if not os.path.exists(self.top_largest_path):
            os.mkdir(self.top_largest_path)
        if not os.path.exists(self.top_smallest_path):
            os.mkdir(self.top_smallest_path)
        if not os.path.exists(self.top_freq_path):
            os.mkdir(self.top_freq_path)
        if not os.path.exists(self.left_top_freq_path):
            os.mkdir(self.left_top_freq_path)

        if not os.path.exists(self.boxplot_path):
            os.mkdir(self.boxplot_path)
        if not os.path.exists(self.countplot_path):
            os.mkdir(self.countplot_path)
        if not os.path.exists(self.barplot_path):
            os.mkdir(self.barplot_path)
        
    def __change_datatypes(self):
        """checking and changing datatypes for better EDA"""
        uni_cat= []
        bi_cat= []

        """object to Uni-Categorical"""
        col_list= self.df.select_dtypes(include=np.object_).columns.to_list()
        for col in col_list:
            unique_no= self.df[col].nunique()
            unique_percentage= (unique_no/self.df[col].count())*100
            if unique_no==1:
                self.df[col]= self.df[col].astype('category')
                uni_cat.append(col)
        
        """Numeric to Uni-Categorical"""
        col_list= self.df.select_dtypes(include=np.number).columns.to_list()
        for col in col_list:
            unique_no= self.df[col].nunique()
            unique_percentage= (unique_no/self.df[col].count())*100
            if unique_no==1:
                self.df[col]= self.df[col].astype('category')
                uni_cat.append(col)

        """object to Bi-Categorical"""
        col_list= self.df.select_dtypes(include=np.object_).columns.to_list()
        for col in col_list:
            unique_no= self.df[col].nunique()
            unique_percentage= (unique_no/self.df[col].count())*100
            if unique_no== 2:
                self.df[col]= self.df[col].astype('category')
                bi_cat.append(col)
        
        """Numeric to Bi-Categorical"""
        col_list= self.df.select_dtypes(include=np.number).columns.to_list()
        for col in col_list:
            unique_no= self.df[col].nunique()
            unique_percentage= (unique_no/self.df[col].count())*100
            if unique_no== 2:
                self.df[col]= self.df[col].astype('category')
                bi_cat.append(col)

        """object to Categorical"""
        col_list= self.df.select_dtypes(include=np.object_).columns.to_list()
        for col in col_list:
            unique_no= self.df[col].nunique()
            unique_percentage= (unique_no/self.df[col].count())*100
            if unique_no< self.cat_threshold or unique_percentage< self.cat_threshold_percentage:
                self.df[col]= self.df[col].astype('category')
            

        """object to Datetime"""
        col_list= self.df.select_dtypes(include=np.object_).columns.to_list()
        for col in col_list:
            col_list= self.df.select_dtypes(include=np.object_).columns.to_list()
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=UserWarning)
                try:
                    self.df[col]= pd.to_datetime(self.df[col])
                except Exception as e:
                    pass
        
                
        temp= dict()
        for col in self.df.columns.to_list():
            if col in uni_cat:
                temp[col]=  "Category-Uni-Value"
            elif col in bi_cat:
                temp[col]=  "Category-Bi-Value"
            elif col in list(set(self.df.select_dtypes(include= 'category').columns.to_list())- set(uni_cat+bi_cat)):
                temp[col]=  "Category"
            elif col in self.df.select_dtypes(include= np.number).columns.to_list():
                temp[col]=  "Numeric"
            elif col in self.df.select_dtypes(include= 'datetime64').columns.to_list():
                temp[col]=  "Datetime"
            elif col in self.df.select_dtypes(include= 'bool').columns.to_list():
                temp[col]=  "Boolean"
            else:
                temp[col]=  "TEXT"

        return temp
    
    def create_html(self, standalone_html= False, htmlpath= "index.html"):
        self.extract_dataframe_info()
        self.num_feature_stats()
        self.dt_feature_stats()
        self.cat_feature_stats()
        self.other_feature_stats()

        self.write_config_for_html(htmlpath, standalone_html)
        html_writer_obj= HTMLWraper(self.config_file)
        html_writer_obj.write_html()

        shutil.rmtree(self.temp_path)
        if standalone_html:
            shutil.rmtree(self.graph_path)

    def extract_dataframe_info(self):
        obj= FeatureStatsExtraction(self.df)
        df_details= pd.DataFrame({
            "Rows": self.df.shape[0],
            "Features": self.df.shape[1],
            "Memory": f"{self.df.memory_usage(index=False, deep=False).sum()/1024} kb",
            "Duplicates": f"{len(self.df)-len(self.df.drop_duplicates())} ({(len(self.df)-len(self.df.drop_duplicates())/self.df.shape[0])* 100}%)"
        }, index=[0])

        feature_details= pd.DataFrame(dict(Counter(self.df_feature_types.values())), index=[0])

        obj.dump_df_to_pickle(df_details, "df_details", self.temp_path)
        obj.dump_df_to_pickle(feature_details, "feature_details", self.temp_path)

        
        self.corr_heatmap_columns= self.df.select_dtypes(include= self.corr_heatmap_datatype).columns.to_list()
        plt_obj= PlotGraphs(self.df)
        plt_obj.plot_corr_heatmap(self.corr_heatmap_columns, self.corr_heatmap_graph_path, self.corr_heatmap_pics_per_row, self.corr_heatmap_cmap)

    def num_feature_stats(self, num_col_list= []):
        if len(num_col_list)== 0:
            num_col_list= [key for key in self.df_feature_types.keys() if self.df_feature_types[key]== "Numeric"]
        num_feature_stats_obj= NumericFeatureStatsExtraction(self.df[num_col_list])
        graph_obj= PlotGraphs(self.df[num_col_list])

        """setup graph colurs and types"""
        num_feature_stats_obj.floating_point_limit= self.floating_point_limit
        num_feature_stats_obj.floating_point_limit_for_percentage= self.floating_point_limit_for_percentage

        graph_obj.count_plot_use_bean= self.count_plot_use_bean
        graph_obj.count_plot_num_bean= self.count_plot_num_bean
        graph_obj.img_format= self.img_format
        graph_obj.boxplot_color= self.boxplot_color
        graph_obj.countplot_color= self.countplot_color

        """creating input for methods"""
        num_col_list_box_plot= [(col, self.boxplot_path) for col in num_col_list]
        num_col_list_count_plot= [(col, self.countplot_path) for col in num_col_list]

        num_desc_col_list= [(col, self.desc_stats_path) for col in num_col_list]
        num_freq_col_list= [(col, self.no_of_rows_right_small_tables, self.top_freq_path) for col in num_col_list]
        num_smallest_col_list= [(col, self.no_of_rows_right_small_tables, self.top_smallest_path) for col in num_col_list]
        num_largest_col_list= [(col, self.no_of_rows_right_small_tables, self.top_largest_path) for col in num_col_list]
        num_perr_corr_col_list= [(col, self.perr_corr_path) for col in num_col_list]
        

        """ploting and storing graphs"""
        with Pool(os.cpu_count()) as process_pool:
            process_pool.starmap(graph_obj.plot_boxplot, num_col_list_box_plot)
            process_pool.starmap(graph_obj.plot_countplot, num_col_list_count_plot)

        """generating and storing stats tables"""
        with ThreadPool(os.cpu_count()) as thread_pool:
            thread_pool.starmap(num_feature_stats_obj.get_desc_stats, num_desc_col_list)
            thread_pool.starmap(num_feature_stats_obj.get_top_freq, num_freq_col_list)
            thread_pool.starmap(num_feature_stats_obj.get_top_smallest, num_smallest_col_list)
            thread_pool.starmap(num_feature_stats_obj.get_top_largest, num_largest_col_list)
            thread_pool.starmap(num_feature_stats_obj.get_perr_corr, num_perr_corr_col_list)
    
    def dt_feature_stats(self, dt_col_list= []):
        stats_rows= 48
        if len(dt_col_list)== 0:
            dt_col_list= [key for key in self.df_feature_types.keys() if self.df_feature_types[key]== "Datetime"]
        dt_feature_stats_obj= DateTimeFeatureStatsExtraction(self.df[dt_col_list])
        graph_obj= PlotGraphs(self.df[dt_col_list])

        """setup graph colurs and types"""
        dt_feature_stats_obj.floating_point_limit= self.floating_point_limit
        dt_feature_stats_obj.floating_point_limit_for_percentage= self.floating_point_limit_for_percentage

        graph_obj.img_format= self.img_format
        graph_obj.barplot_color= self.dt_barplot_color
        
        num_desc_col_list= [(col, self.desc_stats_path) for col in dt_col_list]
        num_freq_col_list= [(col, stats_rows, self.top_freq_path) for col in dt_col_list]
        num_left_freq_col_list= [(col, 7, self.left_top_freq_path) for col in dt_col_list]
        num_left_freq_barplot_col_list= [(col, 7, self.barplot_path) for col in dt_col_list]
        num_smallest_col_list= [(col, stats_rows, self.top_smallest_path) for col in dt_col_list]
        num_largest_col_list= [(col, stats_rows, self.top_largest_path) for col in dt_col_list]
        

        """ploting and storing graphs"""
        with Pool(os.cpu_count()) as process_pool:
            process_pool.starmap(graph_obj.plot_barplot, num_left_freq_barplot_col_list)

        """generating and storing stats tables"""
        with ThreadPool(os.cpu_count()) as thread_pool:
            thread_pool.starmap(dt_feature_stats_obj.get_desc_stats, num_desc_col_list)
            thread_pool.starmap(dt_feature_stats_obj.get_top_freq, num_freq_col_list)
            thread_pool.starmap(dt_feature_stats_obj.get_top_freq, num_left_freq_col_list)
            thread_pool.starmap(dt_feature_stats_obj.get_top_smallest, num_smallest_col_list)
            thread_pool.starmap(dt_feature_stats_obj.get_top_largest, num_largest_col_list)

    def cat_feature_stats(self, cat_col_list= []):
        if len(cat_col_list)== 0:
            cat_col_list= [key for key in self.df_feature_types.keys() if self.df_feature_types[key]in  ["Category", "TEXT"]]
        cat_feature_stats_obj= FeatureStatsExtraction(self.df[cat_col_list])
        graph_obj= PlotGraphs(self.df[cat_col_list])

        """setup graph colurs and types"""
        cat_feature_stats_obj.floating_point_limit= self.floating_point_limit
        cat_feature_stats_obj.floating_point_limit_for_percentage= self.floating_point_limit_for_percentage

        graph_obj.img_format= self.img_format
        graph_obj.barplot_color= self.cat_barplot_color

        cat_desc_col_list= [(col, self.desc_stats_path) for col in cat_col_list]
        cat_freq_col_list= [(col, self.no_of_rows_right_tables, self.top_freq_path) for col in cat_col_list]
        cat_left_freq_col_list= [(col, self.no_of_rows_left_top_freq_table, self.left_top_freq_path) for col in cat_col_list]
        cat_left_freq_barplot_col_list= [(col, self.no_of_rows_left_top_freq_table, self.barplot_path) for col in cat_col_list]
        

        """ploting and storing graphs"""
        with Pool(os.cpu_count()) as process_pool:
            process_pool.starmap(graph_obj.plot_barplot, cat_left_freq_barplot_col_list)

        """generating and storing stats tables"""
        with ThreadPool(os.cpu_count()) as thread_pool:
            thread_pool.starmap(cat_feature_stats_obj.get_desc_stats, cat_desc_col_list)
            thread_pool.starmap(cat_feature_stats_obj.get_top_freq, cat_freq_col_list)
            thread_pool.starmap(cat_feature_stats_obj.get_top_freq, cat_left_freq_col_list)

    def other_feature_stats(self, other_col_list= []):
        if len(other_col_list)== 0:
            other_col_list= [key for key in self.df_feature_types.keys() if self.df_feature_types[key] not in  ["Category", "Datetime", "Numeric"]]
        ot_feature_stats_obj= FeatureStatsExtraction(self.df[other_col_list])
        graph_obj= PlotGraphs(self.df[other_col_list])

        """setup graph colurs and types"""
        ot_feature_stats_obj.floating_point_limit= self.floating_point_limit
        ot_feature_stats_obj.floating_point_limit_for_percentage= self.floating_point_limit_for_percentage
        
        graph_obj.img_format= self.img_format
        graph_obj.barplot_color= self.others_barplot_color

        other_desc_col_list= [(col, self.desc_stats_path) for col in other_col_list]
        other_left_freq_col_list= [(col, self.no_of_rows_left_top_freq_table, self.left_top_freq_path) for col in other_col_list]
        other_left_freq_barplot_col_list= [(col, self.no_of_rows_left_top_freq_table, self.barplot_path) for col in other_col_list]
        

        """ploting and storing graphs"""
        with Pool(os.cpu_count()) as process_pool:
            process_pool.starmap(graph_obj.plot_barplot, other_left_freq_barplot_col_list)

        """generating and storing stats tables"""
        with ThreadPool(os.cpu_count()) as thread_pool:
            thread_pool.starmap(ot_feature_stats_obj.get_desc_stats, other_desc_col_list)
            thread_pool.starmap(ot_feature_stats_obj.get_top_freq, other_left_freq_col_list)




if __name__ == '__main__':
    time_before= datetime.datetime.now()
    #your code here
    dataset_file= R"dataset_in\Divvy_Trips_2019_Q1.xlsx\Divvy_Trips_2019_Q1.xlsx"
    df= pd.read_excel(dataset_file)
    #your code ends here
    time_after= datetime.datetime.now()
    total_time_taken= time_after- time_before
    print("time taken to load dataframe: ",total_time_taken)



    time_before= datetime.datetime.now()
    #your code here
    auto_eda= AutoEDA(df)
    auto_eda.create_html(True)

    #your code ends here
    time_after= datetime.datetime.now()
    total_time_taken= time_after- time_before
    print("time taken: ",total_time_taken)

    

