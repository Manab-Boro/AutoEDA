import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
from io import BytesIO
from collections import Counter





class AutoEDAHelper():
    def __init__(self, df):
        self.df= df
        self.features_type= self.__change_datatypes()
        self.vairables_declaration()

    @property
    def get_dataframe(self):
        return self.df
    
    @property
    def get_memory_usage(self):
        """memory consumption of the dataframe in bytes"""
        return self.df.memory_usage(index=False, deep=False).sum()

    @property
    def get_duplicates_row_no(self):
        return len(self.df)-len(self.df.drop_duplicates())
    
    @property
    def get_duplicates_df(self):
        temp= self.df.groupby(self.df.columns.tolist(),as_index=False).size().sort_values(by=["size"], ascending= [False])
        return temp[temp["size"] >1]

    @property
    def get_features_type(self):
        return self.features_type

    def __change_datatypes(self, cat_threshold= 10, cat_threshold_percentage= 0.25):
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
        
        """object to Bi-Categorical"""
        col_list= self.df.select_dtypes(include=np.object_).columns.to_list()
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
            if unique_no< cat_threshold or unique_percentage< cat_threshold_percentage:
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

    def vairables_declaration(self):
        #For df overview
        self.div_class_df_overiew= "df-div-overiew"
        self.img_class_logo= "logo-img"
        self.table_df= "df-table"

        self.div_class_df_details= "df-div-details"
        self.para_class_df_overiew_descprtion= "df-div-details-desc"
        self.div_id_df_details= "df-details-id-"    #nwill chnaged by the method
        self.para_class_df_header= "df-para-header"
        self.div_class_df_details_inner= "df-div-details-inner"
        self.img_class_df= "df-img-plot"
        self.para_class_df= "df-paragraph"

        #For features:
        self.div_class_ft_overiew= "ft-div-overview"
        self.div_ft_overiew_id= "ft-div-overview-id"
        self.div_class_ft_details_id= "ft-div-details-id-"  #nwill chnaged by the method
        self.para_class_ft_overview_heading= "ft-overview-heading"
        self.div_class_ft_overiew_img= "ft-div-overview-img"
        self.img_class_boxplt= "ft-img-boxplot"
        self.img_class_barplt= "ft-img-barplot"
        self.div_class_ft_overiew_tab= "ft-div-overview-tables"
        self.table_id_overview_0= "ft-table-overview-0"
        self.table_id_overview_1= "ft-table-overview-1" 
        self.table_id_overview_2= "ft-table-overview-2" 

        self.div_class_ft_details= "ft-div-details"
        self.div_class_ft_img_and_table_wraper= "ft-div-details-inner"
        self.div_class_ft_details_img= "ft-div-details-inner-img"
        self.img_class_countplt= "ft-img-countplt"
        self.div_class_ft_details_corr= "ft-div-details-inner-corr"
        self.para_class_ft_small_header_corr= "ft-overview-corr-heading"
        self.para_class_ft_details_tables_header= "ft-details-inner-tables-header"
        self.para_class_ft_details_tables_header_single= "ft-details-inner-tables-header-single"
        self.div_class_ft_details_tables= "ft-div-details-inner-tables"
        self.table_ft_details_corr_id= "ft-table-details-corr"
        self.table_ft_details_id= "ft-table-details"
        self.table_ft_details_single_id= "ft-table-details-single"
        
        self.datatypes_for_corr_plot= []
        self.img_type= ''
        self.pairplot_sample_size= ''
        self.floating_point_limit= ''
        
    def df_overview_as_html(self, datatypes_for_corr_plot, pairplot_sample_size= 50, floating_point_limit=3, img_type= 'png'):
        self.datatypes_for_corr_plot= datatypes_for_corr_plot
        self.img_type= img_type
        self.pairplot_sample_size= pairplot_sample_size
        self.floating_point_limit= floating_point_limit

        memory_use= self.get_memory_usage
        Rows, Features= self.df.shape
        duplicate_rows= self.get_duplicates_row_no
        features_type= self.get_features_type
        features_type= dict(Counter(features_type.values()))

        top_table_1= pd.DataFrame({
            "Rows": [Rows],
            "Features": [Features],
            "Memory": [f"{round(memory_use/1024, floating_point_limit)} kb"],
            "Duplicates": [f"{duplicate_rows} ({round((duplicate_rows/Rows)*100, floating_point_limit)}%)"]
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_df,header= False)

        temp_dict= dict()
        for k in features_type.keys():
            temp_dict[k]= [features_type[k]]

        top_table_2= pd.DataFrame(temp_dict).T.to_html(index= True, justify= "justify-all", table_id= self.table_df,header= False)
        
        corr_img_encoded= ""
        pairplot_img_encoded= ""
        corr_img_error= ""
        pairplot_img_error= ""
        try:
            fig= plt.figure(figsize=(len(datatypes_for_corr_plot)*5.5, len(datatypes_for_corr_plot)*5))
            sns.heatmap(self.df.select_dtypes(include= datatypes_for_corr_plot).apply(lambda x: x.factorize()[0]).corr(), cmap="YlGnBu", annot=True,  linewidth= 5) 
            plt.xticks(rotation=45,fontsize=18)
            plt.yticks(rotation=45,fontsize=18)
            imgfile_2 = BytesIO()
            fig.savefig(imgfile_2, format= img_type, bbox_inches='tight')
            corr_img_encoded = base64.b64encode(imgfile_2.getvalue()).decode('utf-8')
            plt.close(fig)
        except Exception as e:
            corr_img_error= e

        if False:
            try:
                pairplot_img = BytesIO()
                pairplot_fig= sns.pairplot(data= self.df.sample(n=pairplot_sample_size, random_state= 4, ignore_index= True), height= 7)
                pairplot_fig.figure.savefig(pairplot_img, format= img_type, bbox_inches='tight')
                pairplot_img_encoded= base64.b64encode(pairplot_img.getvalue()).decode('utf-8')
                plt.close(pairplot_fig.fig)
            except Exception as e:
                pairplot_img_error= e
        
        df_overiew_html= f'''
            <div class= {self.div_class_df_overiew} id="{self.div_ft_overiew_id}">
                <img class= "{self.img_class_logo}" src="pngwing.com(1).png" alt="logo.png">
                {top_table_1}{top_table_2}
                <p class= "{self.para_class_df_overiew_descprtion}">This is an auto-generated HTML page demo.<br>
                The scope of this project is to analyze exploratory data and do some basic preprocessing autometically.
                Find the GitHub Repo <a href="https://github.com/manab36/AutoEDA">here.</a></p>
            </div>
        '''
        # df_details_html= f'''
        #     <div class= "{self.div_class_df_details}" id= "{self.div_id_df_details}heatmap">
        #         <p class= "{self.para_class_df_header}">Dataframe Overiew</p>
        #         <hr style= "margin-right: 20px; margin-left: 20px;">
        #         <div class= "{self.div_class_df_details_inner}">
        #             <img class= "{self.img_class_df}" src="data:image/{img_type};base64, {corr_img_encoded}" alt="{corr_img_error}" />
        #             <hr style= "margin-right: 25px; margin-left: 25px;">
        #             <p class= "{self.para_class_df}">Correlation Heatmap: *Data Types incuded: {datatypes_for_corr_plot}</p>
        #         </div>
        #     </div>
        #     <div class= "{self.div_class_df_details}" id= "{self.div_id_df_details}pairplot">
        #         <p class= "{self.para_class_df_header}">Dataframe Overiew</p>
        #         <hr style= "margin-right: 20px; margin-left: 20px;">
        #         <div class= "{self.div_class_df_details_inner}">
        #             <img class= "{self.img_class_df}" src="data:image/{img_type};base64, {corr_img_encoded}" alt="{pairplot_img_error}" />
        #             <hr style= "margin-right: 25px; margin-left: 25px;">
        #             <p class= "{self.para_class_df}">*Ploting of Pair-plot was done on {pairplot_sample_size} random samples. (Increasing the size of sample will significantly increase process time.)</p>
        #         </div>
        #     </div>'''
        df_details_html= f'''
            <div class= "{self.div_class_df_details}" id= "{self.div_id_df_details}heatmap">
                <p class= "{self.para_class_df_header}">Dataframe Overiew</p>
                <hr style= "margin-right: 20px; margin-left: 20px;">
                <div class= "{self.div_class_df_details_inner}">
                    <img class= "{self.img_class_df}" src="data:image/{img_type};base64, {corr_img_encoded}" alt="{corr_img_error}" />
                    <hr style= "margin-right: 25px; margin-left: 25px;">
                    <p class= "{self.para_class_df}">Correlation Heatmap: *Data Types incuded: {datatypes_for_corr_plot}</p>
                </div>
            </div>
            '''

        return df_overiew_html, df_details_html

    def feacture_as_html(self, colname):
        feature_overiew_html= ''
        feature_details_html= ''

        if self.features_type[colname]== "Numeric":
            feature_overiew_html, feature_details_html= self.num_feacture_as_html(colname)     
                
        elif self.features_type[colname]== "Datetime":
            feature_overiew_html, feature_details_html= self.datetime_feacture_as_html(colname)
            
        elif self.features_type[colname]== "Category":
            feature_overiew_html, feature_details_html= self.cat_feacture_as_html(colname)

        elif self.features_type[colname] in  ["Boolean", "Category-Bi-Value"]:
            feature_overiew_html= self.bi_cat_and_bool_feacture_as_html(colname)

        elif self.features_type[colname]== "Category-Uni-Value":
            feature_overiew_html= self.uni_cat_feacture_as_html(colname)
        else:
            feature_overiew_html, feature_details_html= self.text_feacture_as_html(colname)

            
        
        return feature_overiew_html, feature_details_html

    def num_feacture_as_html(self, colname):
        desc_stats= self.num_feacture_desc_stats(colname)
        table_1= pd.DataFrame({
                "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
                "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
                "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
                "Zeros": [f'{desc_stats["Zeros"].iloc[0]} ({desc_stats["Zeros_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0,header= False)
        table_2= desc_stats[["Min", "5%", "25%", "50%", "75%", "95%", "Max"]].T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1,header= False)
        table_3= pd.DataFrame({
                "Range": [f'{desc_stats["Range"].iloc[0].round(self.floating_point_limit)}'],
                "Upper Bound": [f'{desc_stats["Upper_Bound"].iloc[0].round(self.floating_point_limit)}'],
                "IQR": [f'{desc_stats["IQR"].iloc[0].round(self.floating_point_limit)}'],
                "Lower Bound": [f'{desc_stats["Lower_Bound"].iloc[0].round(self.floating_point_limit)}'],
                "Upper Bound <": [f'{desc_stats["Gt_Upper"].iloc[0].round(self.floating_point_limit)} ({desc_stats["Gt_Upper_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
                "Lower Bound >": [f'{desc_stats["Lt_Lower"].iloc[0].round(self.floating_point_limit)} ({desc_stats["Lt_Lower_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1,header= False)
        table_4= desc_stats[["Std", "Var", "Skew", "Kurtosis"]].round(self.floating_point_limit).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1,header= False)


        pear_corr= self.df.select_dtypes(include= np.number).corr('pearson')[colname]
        ken_corr= self.df.select_dtypes(include= np.number).corr(method='kendall')[colname]

        right_table_1= pd.DataFrame({
            "Colname": pear_corr.index,
            "Value": pear_corr
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_corr_id,header= False)
        right_table_2= pd.DataFrame({
            "Colname": ken_corr.index,
            "Value": ken_corr
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_corr_id,header= False)

        top_freq= self.df[[colname]].sort_index(ascending= False).value_counts().sort_values(ascending= False).nlargest(15).reset_index()
        top_freq["Count_Percentage"]= (top_freq["count"]/self.df[colname].count())*100
        top_freq[colname]= top_freq[colname].astype(str)
        top_freq.loc[len(top_freq)]= ["others...", len(self.df[colname].notnull())-top_freq["count"].sum(), 100-top_freq["Count_Percentage"].sum()]


        smallest_val= self.df[[colname]].value_counts().reset_index().sort_values(by= [colname]).nsmallest(15, columns= [colname])
        smallest_val["Count_Percentage"]= (smallest_val["count"]/self.df[colname].count())*100
        smallest_val[colname]= smallest_val[colname].astype(str)
        smallest_val.loc[len(smallest_val)]= ["others...", len(self.df[colname].notnull())-smallest_val["count"].sum(), 100-smallest_val["Count_Percentage"].sum()]

        lagest_val= self.df[[colname]].value_counts().reset_index().sort_values(by= [colname]).nlargest(15, columns= [colname])
        lagest_val["Count_Percentage"]= (lagest_val["count"]/self.df[colname].count())*100
        lagest_val[colname]= lagest_val[colname].astype(str)
        lagest_val.loc[len(lagest_val)]= ["others...", len(self.df[colname].notnull())-lagest_val["count"].sum(), 100-lagest_val["Count_Percentage"].sum()]

        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id,header= False)
        right_table_4= pd.DataFrame({
            "Data": smallest_val[colname],
            "Count": smallest_val["count"].astype(str) + " ("+ smallest_val["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id,header= False)
        right_table_5= pd.DataFrame({
            "Data": lagest_val[colname],
            "Count": lagest_val["count"].astype(str) + " ("+ lagest_val["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id,header= False)
        
        """box plot"""
        box_fig= plt.figure(figsize=(12,5))
        sns.boxplot(data=self.df, x= colname,color="red")
        box_fig_tmpfile = BytesIO()
        box_fig.savefig(box_fig_tmpfile, format=self.img_type, bbox_inches='tight')
        box_encoded = base64.b64encode(box_fig_tmpfile.getvalue()).decode('utf-8')
        plt.close(box_fig)
        """count plot"""
        count_fig= plt.figure(figsize=(17,14))
        sns.countplot(data=self.df, x= colname,color="red")
        count_fig_tmpfile = BytesIO()
        count_fig.savefig(count_fig_tmpfile, format=self.img_type, bbox_inches='tight')
        count_encoded = base64.b64encode(count_fig_tmpfile.getvalue()).decode('utf-8')
        plt.close(count_fig)
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{self.df.columns.get_loc(colname)+1} {colname} ({self.features_type[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_boxplt}" src="data:image/{self.img_type};base64,{box_encoded}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_2}{table_3}{table_4}
                </div>
            </div>'''
        
        feature_details_html= f'''
            <div class= "{self.div_class_ft_details}" id= "{self.div_class_ft_details_id}{colname}">
                <p class ="{self.para_class_df_header}">{colname}</p>
                <hr style="margin-left: 25px; margin-right: 25px;">
                <div class="{self.div_class_ft_img_and_table_wraper}">
                    <div class= "{self.div_class_ft_details_img}">
                        <img class= "{self.img_class_countplt}" src="data:image/{self.img_type};base64,{count_encoded}" alt="Graph">
                    </div>
                    <div class= "{self.div_class_ft_details_corr}">
                        <p class="{self.para_class_ft_small_header_corr}">Correlation: Pearson</p>{right_table_1}<p class="{self.para_class_ft_small_header_corr}">Correlation: Kendall</p>{right_table_2}
                    </div>
                </div>
                <div class= "{self.div_class_ft_details_tables}">
                    <hr style="margin-left: 25px; margin-right: 25px;">
                    <p class= "{self.para_class_ft_details_tables_header}">Top Frequent</p>
                    <p class= "{self.para_class_ft_details_tables_header}">Top Smallest Values</p>
                    <p class= "{self.para_class_ft_details_tables_header}">Top Largest Values</p>
                    {right_table_3}{right_table_4}{right_table_5}
                </div>
            </div>'''

        return feature_overiew_html, feature_details_html

    def datetime_feacture_as_html(self, colname):
        no_of_rows_right= 50
        no_of_rows_left= 7

        desc_stats= self.datetime_feacture_desc_stats(colname)
        temp= pd.DataFrame(self.df[colname].value_counts().nlargest(no_of_rows_left)).reset_index()
        temp[colname]= temp[colname].astype(str)
        temp.loc[len(temp)]= ["others...", self.df[colname].count()- temp["count"].sum()]
        temp["Count_Percentage"]= (temp["count"]/self.df[colname].count())*100
        temp= temp.round(decimals=self.floating_point_limit)

        bar_fig= plt.figure(figsize=(15,5))
        sns.barplot(data=temp, x=colname, y="count", color= "blue")
        bar_fig_tmpfile = BytesIO()
        bar_fig.savefig(bar_fig_tmpfile, format=self.img_type, bbox_inches='tight')
        bar_encoded = base64.b64encode(bar_fig_tmpfile.getvalue()).decode('utf-8')
        plt.close(bar_fig)

        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0].round(self.floating_point_limit)} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_2= desc_stats[["Start", "End", "Std"]].T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1, header= False)
        table_3= pd.DataFrame({
            "Date": temp[colname],
            "Values": temp["count"].astype(str)+ " ("+ temp["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)

        #for right tab
        top_freq= self.df[[colname]].sort_index(ascending= False).value_counts().sort_values(ascending= False).nlargest(no_of_rows_right).reset_index()
        top_freq["Count_Percentage"]= (top_freq["count"]/self.df[colname].count())*100
        top_freq.loc[len(top_freq)]= ["others...", len(self.df[colname].notnull())-top_freq["count"].sum(), 100-top_freq["Count_Percentage"].sum()]

        smallest_val= self.df[[colname]].value_counts().reset_index().sort_values(by= [colname]).nsmallest(no_of_rows_right, columns= [colname])
        smallest_val["Count_Percentage"]= (smallest_val["count"]/self.df[colname].count())*100
        smallest_val.loc[len(smallest_val)]= ["others...", len(self.df[colname].notnull())-smallest_val["count"].sum(), 100-smallest_val["Count_Percentage"].sum()]

        lagest_val= self.df[[colname]].value_counts().reset_index().sort_values(by= [colname]).nlargest(no_of_rows_right, columns= [colname])
        lagest_val["Count_Percentage"]= (lagest_val["count"]/self.df[colname].count())*100
        lagest_val.loc[len(lagest_val)]= ["others...", len(self.df[colname].notnull())-lagest_val["count"].sum(), 100-lagest_val["Count_Percentage"].sum()]

        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id, header= False)
        right_table_4= pd.DataFrame({
            "Data": smallest_val[colname],
            "Count": smallest_val["count"].astype(str) + " ("+ smallest_val["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id, header= False)
        right_table_5= pd.DataFrame({
            "Data": lagest_val[colname],
            "Count": lagest_val["count"].astype(str) + " ("+ lagest_val["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id, header= False)
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{self.df.columns.get_loc(colname)+1} {colname} ({self.features_type[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_barplt}" src="data:image/{self.img_type};base64,{bar_encoded}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_2}{table_3}
                </div>
            </div>'''
        feature_details_html= f'''
            <div class= "{self.div_class_ft_details}" id= "{self.div_class_ft_details_id}{colname}">
                <p class ="{self.para_class_df_header}">{colname}</p>
                <hr style="margin-left: 25px; margin-right: 25px;">
                <div class= "{self.div_class_ft_details_tables}">
                    <p class= "{self.para_class_ft_details_tables_header}">Top Frequent</p>
                    <p class= "{self.para_class_ft_details_tables_header}">Earliest Dates/Times</p>
                    <p class= "{self.para_class_ft_details_tables_header}">Latest Dates/Times</p>
                    {right_table_3}{right_table_4}{right_table_5}
                </div>
            </div>'''
        return feature_overiew_html, feature_details_html

    def cat_feacture_as_html(self, colname):
        no_of_rows_right= 50
        no_of_rows_left= 7

        desc_stats= self.cat_feacture_desc_stats(colname)
        temp= pd.DataFrame(self.df[colname].value_counts().nlargest(no_of_rows_left)).reset_index()
        temp[colname]= temp[colname].astype(str)
        temp.loc[len(temp)]= ["others...", self.df[colname].count()- temp["count"].sum()]
        temp["Count_Percentage"]= (temp["count"]/self.df[colname].count())*100
        temp= temp.round(decimals=self.floating_point_limit)

        bar_fig= plt.figure(figsize=(15,5))
        sns.barplot(data=temp, x=colname, y="count", color= "purple")
        bar_fig_tmpfile = BytesIO()
        bar_fig.savefig(bar_fig_tmpfile, format=self.img_type, bbox_inches='tight')
        bar_encoded = base64.b64encode(bar_fig_tmpfile.getvalue()).decode('utf-8')
        plt.close(bar_fig)

        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0].round(self.floating_point_limit)} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_3= pd.DataFrame({
            "Date": temp[colname],
            "Values": temp["count"].astype(str)+ " ("+ temp["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)

        #for right tab
        top_freq= self.df[[colname]].sort_index(ascending= False).value_counts().sort_values(ascending= False).nlargest(no_of_rows_right).reset_index()
        top_freq["Count_Percentage"]= (top_freq["count"]/self.df[colname].count())*100
        top_freq.loc[len(top_freq)]= ["others...", len(self.df[colname].notnull())-top_freq["count"].sum(), 100-top_freq["Count_Percentage"].sum()]

        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_single_id, header= False)
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{self.df.columns.get_loc(colname)+1} {colname} ({self.features_type[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_barplt}" src="data:image/{self.img_type};base64,{bar_encoded}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_3}
                </div>
            </div>'''
        feature_details_html= f'''
            <div class= "{self.div_class_ft_details}" id= "{self.div_class_ft_details_id}{colname}">
                <p class ="{self.para_class_df_header}">{colname}</p>
                <hr style="margin-left: 25px; margin-right: 25px;">
                <div class= "{self.div_class_ft_details_tables}">
                    <p class= "{self.para_class_ft_details_tables_header_single}">Top Frequent</p>
                    {right_table_3}
                </div>
            </div>'''
        return feature_overiew_html, feature_details_html

    def text_feacture_as_html(self, colname):
        no_of_rows_right= 50
        no_of_rows_left= 7

        desc_stats= self.cat_feacture_desc_stats(colname)
        temp= pd.DataFrame(self.df[colname].value_counts().nlargest(no_of_rows_left)).reset_index()
        temp[colname]= temp[colname].astype(str)
        temp.loc[len(temp)]= ["others...", self.df[colname].count()- temp["count"].sum()]
        temp["Count_Percentage"]= (temp["count"]/self.df[colname].count())*100
        temp= temp.round(decimals=self.floating_point_limit)

        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0].round(self.floating_point_limit)} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_3= pd.DataFrame({
            "Date": temp[colname],
            "Values": temp["count"].astype(str)+ " ("+ temp["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)

        #for right tab
        top_freq= self.df[[colname]].sort_index(ascending= False).value_counts().sort_values(ascending= False).nlargest(no_of_rows_right).reset_index()
        top_freq["Count_Percentage"]= (top_freq["count"]/self.df[colname].count())*100
        top_freq.loc[len(top_freq)]= ["others...", len(self.df[colname].notnull())-top_freq["count"].sum(), 100-top_freq["Count_Percentage"].sum()]

        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].round(self.floating_point_limit).astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_single_id, header= False)
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{self.df.columns.get_loc(colname)+1} {colname} ({self.features_type[colname]})</p>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_3}
                </div>
            </div>'''
        feature_details_html= f'''
            <div class= "{self.div_class_ft_details}" id= "{self.div_class_ft_details_id}{colname}">
                <p class ="{self.para_class_df_header}">{colname}</p>
                <hr style="margin-left: 25px; margin-right: 25px;">
                <div class= "{self.div_class_ft_details_tables}">
                    <p class= "{self.para_class_ft_details_tables_header_single}">Top Frequent</p>
                    {right_table_3}
                </div>
            </div>'''
        return feature_overiew_html, feature_details_html

    def bi_cat_and_bool_feacture_as_html(self, colname):
        desc_stats= self.cat_feacture_desc_stats(colname)
        temp= pd.DataFrame(self.df[colname].value_counts().nlargest(2)).reset_index()
        temp["Count_Percentage"]= (temp["count"]/self.df[colname].count())*100
        temp= temp.round(decimals=self.floating_point_limit)

        desc_stats= self.cat_feacture_desc_stats(colname)
        temp= pd.DataFrame(self.df[colname].value_counts().nlargest(2)).reset_index()
        temp[colname]= temp[colname].astype(str)
        temp.loc[len(temp)]= ["others...", self.df[colname].count()- temp["count"].sum()]
        temp["Count_Percentage"]= (temp["count"]/self.df[colname].count())*100
        temp= temp.round(decimals=self.floating_point_limit)

        bar_fig= plt.figure(figsize=(15,5))
        sns.barplot(data=temp, x=colname, y="count", color= "cyan")
        bar_fig_tmpfile = BytesIO()
        bar_fig.savefig(bar_fig_tmpfile, format=self.img_type, bbox_inches='tight')
        bar_encoded = base64.b64encode(bar_fig_tmpfile.getvalue()).decode('utf-8')
        plt.close(bar_fig)

        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0].round(self.floating_point_limit)} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_3= pd.DataFrame({
            "Date": temp[colname],
            "Values": temp["count"].astype(str)+ " ("+ temp["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" id="{self.div_ft_overiew_id}">
                <p class ="{self.para_class_ft_overview_heading}">{self.df.columns.get_loc(colname)+1} {colname} ({self.features_type[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_barplt}" src="data:image/{self.img_type};base64,{bar_encoded}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_3}
                </div>
            </div>'''
        return feature_overiew_html

    def uni_cat_feacture_as_html(self, colname):
        desc_stats= self.cat_feacture_desc_stats(colname)
        temp= pd.DataFrame(self.df[colname].value_counts().nlargest(2)).reset_index()
        temp["Count_Percentage"]= (temp["count"]/self.df[colname].count())*100
        temp= temp.round(decimals=self.floating_point_limit)

        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0].round(self.floating_point_limit)} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0].round(self.floating_point_limit)} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_3= pd.DataFrame({
            "Date": temp[colname],
            "Values": temp["count"].astype(str)+ " ("+ temp["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" id="{self.div_ft_overiew_id}">
                <p class ="{self.para_class_ft_overview_heading}">{self.df.columns.get_loc(colname)+1} {colname} ({self.features_type[colname]})</p>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_3}
                </div>
            </div>'''
        return feature_overiew_html

    def num_feacture_desc_stats(self, colname):
        temp= self.df[colname]
        desc_stats= temp.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame().T

        desc_stats["sum"]= np.sum(temp, axis=0)
        desc_stats["skew"]= skew(temp, axis=0, bias=False, nan_policy='omit')
        desc_stats["kurtosis"]= kurtosis(temp, axis=0, bias=False, nan_policy='omit')
        desc_stats["var"]= np.var(temp, axis=0)
        desc_stats["Range"]= desc_stats["max"]- desc_stats["min"]
        desc_stats["IQR"]= desc_stats["75%"]- desc_stats["25%"]
        desc_stats["Upper_Bound"]= desc_stats["75%"]+ (1.5* desc_stats["IQR"])
        desc_stats["Lower_Bound"]= desc_stats["25%"]- (1.5* desc_stats["IQR"])
        desc_stats["Missing"]= temp.isnull().sum()
        desc_stats["Missing_Percentage"]= (desc_stats["Missing"]/(desc_stats["count"]+ desc_stats["Missing"]))* 100
        desc_stats["Distinct"]= temp.nunique()
        desc_stats["Distinct_Percentage"]= (desc_stats["Distinct"]/(desc_stats["count"]+ desc_stats["Missing"]))* 100
        desc_stats["count"]= desc_stats["count"].astype(np.int64)
        desc_stats["Zeros"]= temp[temp==0].count()
        desc_stats["Zeros_Percentage"]= (desc_stats["Zeros"]/(desc_stats["count"]+ desc_stats["Missing"]))* 100
        desc_stats["Values_Percentage"]= 100- desc_stats["Missing_Percentage"]

        gt_upper= 0
        lt_lower= 0

        gt_upper= temp.gt(desc_stats["Upper_Bound"].iloc[0]).sum()
        lt_lower= temp.lt(desc_stats["Lower_Bound"].iloc[0]).sum()

        desc_stats["Gt_Upper"]= gt_upper
        desc_stats["Gt_Upper_Percentage"]= (desc_stats["Gt_Upper"]/desc_stats["count"])* 100
        desc_stats["Lt_Lower"]= lt_lower
        desc_stats["Lt_Lower_Percentage"]= (desc_stats["Lt_Lower"]/desc_stats["count"])* 100

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

        return desc_stats

    def datetime_feacture_desc_stats(self, colname):
        temp=  self.df[colname]
        desc_stats= pd.DataFrame({
            "Values": [temp.count()],
            "Unique": [temp.nunique()],
            "Start": [temp.min()],
            "End": [temp.max()],
            "Missing": [temp.isnull().sum()],
            })
        desc_stats["Missing_Percentage"]= (desc_stats["Missing"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100
        desc_stats["Distinct"]= temp.nunique()
        desc_stats["Distinct_Percentage"]= (desc_stats["Distinct"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100
        desc_stats["Values_Percentage"]= 100- desc_stats["Missing_Percentage"]
        
        if False:
            std_df= pd.DataFrame({"Time_1":self.df[colname].sort_values()[1:].to_list(), "Time_2": self.df[colname].sort_values()[:-1].to_list()})
            std_df["Time_diff"]= (std_df["Time_1"]- std_df["Time_2"])
            std_df["Time_diff_in_sec"]= (std_df["Time_1"]- std_df["Time_2"]).dt.total_seconds()
            desc_stats["Std"]= str(std_df["Time_diff_in_sec"].std().round(floating_point_limit))+ " sec"
        else:
            desc_stats["Std"]= None

        return desc_stats 

    def cat_feacture_desc_stats(self, colname):
        temp=  self.df[colname]
        desc_stats= pd.DataFrame({
            "Values": [temp.count()],
            "Unique": [temp.nunique()],
            "Missing": [temp.isnull().sum()],
            })
        desc_stats["Missing_Percentage"]= (desc_stats["Missing"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100
        desc_stats["Distinct"]= temp.nunique()
        desc_stats["Distinct_Percentage"]= (desc_stats["Distinct"]/(desc_stats["Values"]+ desc_stats["Missing"]))* 100
        desc_stats["Values_Percentage"]= 100- desc_stats["Missing_Percentage"]
        # print(temp.std())

        return desc_stats   #.round(decimals=2)

    



if __name__ == '__main__':
    print("hrllo")

