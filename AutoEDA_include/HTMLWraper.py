import os
import base64
import pandas as pd
import pickle
import json
import numpy as np

class HTMLWraper():
    def __init__(self, config_filepath):
        self.read_config(config_filepath)
    
    def read_config(self, config_filepath):
        file_obj= open(config_filepath, "r")
        config_file_contet= json.load(file_obj)
        file_obj.close()
        
        self.html_out_file=  config_file_contet["html_file"]
        self.img_format= config_file_contet["img_format"]

        self.temp_path= config_file_contet["dump_path"]
        self.graph_path= config_file_contet["graph_path"]
        
        self.desc_stats_path= config_file_contet["desc_stats_path"]
        self.perr_corr_path= config_file_contet["perr_corr_path"]
        self.top_largest_path= config_file_contet["top_largest_path"]
        self.top_smallest_path= config_file_contet["top_smallest_path"]
        self.top_freq_path= config_file_contet["top_freq_path"]
        self.left_top_freq_path= config_file_contet["left_top_freq_path"]

        self.corr_heatmap_path= os.path.relpath(config_file_contet["corr_heatmap_path"], os.getcwd())
        self.boxplot_path= os.path.relpath(config_file_contet["boxplot_path"], os.getcwd())
        self.countplot_path= os.path.relpath(config_file_contet["countplot_path"], os.getcwd())
        self.barplot_path= os.path.relpath(config_file_contet["barplot_path"], os.getcwd())

        self.df_feature_types= config_file_contet["df_feature_types"]
        self.standalone_html= config_file_contet["standalone"]
        self.page_overview= config_file_contet["page_overview"]
        self.corr_heatmap_datatype= config_file_contet["corr_heatmap_datatype"]

    def write_html(self):
        self.html_tags()
        self.write_header()
        self.write_body()

    def html_tags(self):
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

    def read_from_pickle(self, filepath):
        file_obj= open(filepath, 'rb')
        file_content= pickle.load(file_obj)
        file_obj.close()
        return file_content

    def write_header(self):
        '''This function wirtes dataframe overview and css file in the html file'''
        css_file= "AutoEDA_include/AutoEDA.css"
        if self.standalone_html:
            css_file_obj= open(css_file, 'r')
            css_content= css_file_obj.read()
            css_file_obj.close()
            html_content_start= f'<!DOCTYPE html><html><head><title>AutoEDA</title><style>{css_content}</style></head><body>'
        else:
            html_content_start= f'<!DOCTYPE html><html><head><link rel="stylesheet" href="{css_file}"><title>AutoEDA</title></head><body>'
        html_file = open(self.html_out_file, "w")
        html_file.write(html_content_start)
        html_file.close()

    def write_body(self):
        """for dataframe details"""
        df_details= self.read_from_pickle(os.path.join(self.temp_path, 'df_details')).T.to_html(index= True, justify= "justify-all", table_id= self.table_df,header= False)
        feature_details= self.read_from_pickle(os.path.join(self.temp_path, 'feature_details')).T.to_html(index= True, justify= "justify-all", table_id= self.table_df,header= False)

        logo_img_src= "AutoEDA_include/pngwing.com(1).png"
        if self.standalone_html:
            img_file= open(logo_img_src, "rb")
            img_encd= base64.b64encode(img_file.read())
            logo_img_src= f"data:image/png;base64, {str(img_encd)[2:-1]}"

        df_overiew_html= f'''
            <div class= {self.div_class_df_overiew} id="{self.div_ft_overiew_id}">
                <img class= "{self.img_class_logo}" src="{logo_img_src}" alt="logo.png">
                {df_details}{feature_details}
                <p class= "{self.para_class_df_overiew_descprtion}">{self.page_overview}</p>
            </div>
        '''

        list_of_corr_images= os.listdir(self.corr_heatmap_path)
        if len(list_of_corr_images) == 2:
            corr_heatmap= os.path.join(self.corr_heatmap_path, f"1.{self.img_format}")
            if self.standalone_html:
                img_file= open(corr_heatmap, "rb")
                img_encd= base64.b64encode(img_file.read())
                corr_heatmap= f"data:image/png;base64, {str(img_encd)[2:-1]}"
            corr_heatmap= f'<img class= "{self.img_class_df}" src="{corr_heatmap}" alt="" />'
        else:
            img_per_row= int(np.ceil(np.sqrt(len(list_of_corr_images)-1)))
            temp_html= """<div style="float: left; width: 900px; margin-left: 3px;">"""

            count= 1
            width= 900/img_per_row
            for i in range(0, img_per_row):
                temp_html+= """<div style="float: left;">"""
                for j in range(0, img_per_row):
                    img_path= os.path.join(self.corr_heatmap_path, f"{count}.{self.img_format}")
                    if not os.path.exists(img_path):
                        break
                    if self.standalone_html:
                        img_file= open(img_path, "rb")
                        img_encd= base64.b64encode(img_file.read())
                        img_path= f"data:image/png;base64, {str(img_encd)[2:-1]}"
                    temp_html+= f"""<img style="float: left;  width: {width}px; height: {width}px;" src="{img_path}" alt="Italian Trulli">"""
                    count+= 1
                temp_html+= "</div>"
            temp_html+= "</div>"
            img_path= os.path.join(self.corr_heatmap_path, f"0.{self.img_format}")
            if self.standalone_html:
                img_file= open(img_path, "rb")
                img_encd= base64.b64encode(img_file.read())
                img_path= f"data:image/png;base64, {str(img_encd)[2:-1]}"
            temp_html+= f"""<div  style="float: left; margin-left: 10px; margin-top: 10px;"><img style="float: left; width: 30px; height: 800px;" src="{img_path}" alt="Italian Trulli"></div>"""
            corr_heatmap= temp_html
        
        df_details_html= f'''
            <div class= "{self.div_class_df_details}" id= "{self.div_id_df_details}heatmap">
                <p class= "{self.para_class_df_header}">Dataframe Overiew</p>
                <hr style= "margin-right: 20px; margin-left: 20px;">
                <div class= "{self.div_class_df_details_inner}">
                    {corr_heatmap}
                    <hr style= "margin-right: 25px; margin-left: 25px;">
                    <p class= "{self.para_class_df}">Correlation Heatmap: *Data Types incuded: {self.corr_heatmap_datatype}</p>
                </div>
            </div>
            '''
        
        html_file = open(self.html_out_file, "a")
        html_file.write(df_overiew_html+ df_details_html)
        html_file.close()
    
        """For each Features"""
        i=1 
        for colname in self.df_feature_types.keys():
            feature_overiew_html= ''
            feature_details_html= ''
            if self.df_feature_types[colname]== "Numeric":
                feature_overiew_html, feature_details_html= self.num_feature(colname, i)     
                    
            elif self.df_feature_types[colname]== "Datetime":
                feature_overiew_html, feature_details_html= self.dt_feature(colname, i)     
                
            elif self.df_feature_types[colname] in ["Category", "TEXT"]:
                feature_overiew_html, feature_details_html= self.cat_feature(colname, i)     

            else:
                feature_overiew_html= self.others_feature(colname, i)     
            
            i+= 1
            html_file = open(self.html_out_file, "a")
            html_file.write(feature_overiew_html+ feature_details_html)
            html_file.close()
        
        '''wirtes the end of the html file and javascripts file in the html file'''
        html_content_end= '''<script>
        function displayrightdiv(id, divclass) {
            var element = document.getElementById(id);
            if (element.style.display !== 'block') {
                var elements = document.getElementsByClassName(divclass);
                for (var i = 0; i < elements.length; i++) {
                  elements[i].style.display = 'none';
                }
                element.style.display = 'block';
            element.style.display = 'block';
            } else {
                element.style.display = 'none';
            }
        }
        </script></body></html>'''
        html_file = open(self.html_out_file, "a")
        html_file.write(html_content_end)
        html_file.close()
        
    def num_feature(self, colname, col_no):
        """Left side"""
        desc_stats= self.read_from_pickle(os.path.join(self.desc_stats_path, colname))
        table_1= pd.DataFrame({
                "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0]} %)'],
                "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0]} %)'],
                "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0]} %)'],
                "Zeros": [f'{desc_stats["Zeros"].iloc[0]} ({desc_stats["Zeros_Percentage"].iloc[0]} %)'],
            }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0,header= False)
        table_2= desc_stats[["Min", "5%", "25%", "50%", "75%", "95%", "Max"]].T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1,header= False)
        table_3= pd.DataFrame({
                "Range": [f'{desc_stats["Range"].iloc[0]}'],
                "Upper Bound": [f'{desc_stats["Upper_Bound"].iloc[0]}'],
                "IQR": [f'{desc_stats["IQR"].iloc[0]}'],
                "Lower Bound": [f'{desc_stats["Lower_Bound"].iloc[0]}'],
                "Upper Bound <": [f'{desc_stats["Gt_Upper"].iloc[0]} ({desc_stats["Gt_Upper_Percentage"].iloc[0]} %)'],
                "Lower Bound >": [f'{desc_stats["Lt_Lower"].iloc[0]} ({desc_stats["Lt_Lower_Percentage"].iloc[0]} %)'],
            }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1,header= False)
        table_4= desc_stats[["Std", "Var", "Skew", "Kurtosis"]].T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1,header= False)

        boxplot= os.path.join(self.boxplot_path, f"{colname}.{self.img_format}")
        if self.standalone_html:
            img_file= open(boxplot, "rb")
            img_encd= base64.b64encode(img_file.read())
            boxplot= f"data:image/png;base64, {str(img_encd)[2:-1]}"

        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{col_no} {colname} ({self.df_feature_types[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_boxplt}" src="{boxplot}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_2}{table_3}{table_4}
                </div>
            </div>'''
        
        """Right side"""
        pearson_corr= self.read_from_pickle(os.path.join(self.perr_corr_path, colname))
        top_freq= self.read_from_pickle(os.path.join(self.top_freq_path, colname))
        smallest_val= self.read_from_pickle(os.path.join(self.top_smallest_path, colname))
        lagest_val= self.read_from_pickle(os.path.join(self.top_largest_path, colname))
        right_table_1= pearson_corr.to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_corr_id,header= False)
        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id,header= False)
        right_table_4= pd.DataFrame({
            "Data": smallest_val[colname],
            "Count": smallest_val["count"].astype(str) + " ("+ smallest_val["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id,header= False)
        right_table_5= pd.DataFrame({
            "Data": lagest_val[colname],
            "Count": lagest_val["count"].astype(str) + " ("+ lagest_val["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id,header= False)

        countplot= os.path.join(self.countplot_path, f"{colname}.{self.img_format}")
        if self.standalone_html:
            img_file= open(countplot, "rb")
            img_encd= base64.b64encode(img_file.read())
            countplot= f"data:image/png;base64, {str(img_encd)[2:-1]}"

        feature_details_html= f'''
            <div class= "{self.div_class_ft_details}" id= "{self.div_class_ft_details_id}{colname}">
                <p class ="{self.para_class_df_header}">{colname}</p>
                <hr style="margin-left: 25px; margin-right: 25px;">
                <div class="{self.div_class_ft_img_and_table_wraper}">
                    <div class= "{self.div_class_ft_details_img}">
                        <img class= "{self.img_class_countplt}" src="{countplot}" alt="Graph">
                        <p class="{self.para_class_ft_small_header_corr}" style= "margin-left: 25px;">{55555555}</p>
                    </div>
                    <div class= "{self.div_class_ft_details_corr}">
                        <p class="{self.para_class_ft_small_header_corr}">Correlation: Pearson</p>{right_table_1}
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

    def dt_feature(self, colname, col_no):
        """Left side"""
        desc_stats= self.read_from_pickle(os.path.join(self.desc_stats_path, colname))
        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0]} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0]} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0]} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_2= desc_stats[["Start", "End", "Std"]].T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_1, header= False)
        table_3= self.read_from_pickle(os.path.join(self.left_top_freq_path, colname)).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)

        barplot= os.path.join(self.barplot_path, f"{colname}.{self.img_format}")
        if self.standalone_html:
            img_file= open(barplot, "rb")
            img_encd= base64.b64encode(img_file.read())
            barplot= f"data:image/png;base64, {str(img_encd)[2:-1]}"

        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{col_no} {colname} ({self.df_feature_types[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_barplt}" src="{barplot}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_2}{table_3}
                </div>
            </div>'''


        """Right side"""
        top_freq= self.read_from_pickle(os.path.join(self.top_freq_path, colname))
        smallest_val= self.read_from_pickle(os.path.join(self.top_smallest_path, colname))
        lagest_val= self.read_from_pickle(os.path.join(self.top_largest_path, colname))
        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id, header= False)
        right_table_4= pd.DataFrame({
            "Data": smallest_val[colname],
            "Count": smallest_val["count"].astype(str) + " ("+ smallest_val["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id, header= False)
        right_table_5= pd.DataFrame({
            "Data": lagest_val[colname],
            "Count": lagest_val["count"].astype(str) + " ("+ lagest_val["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_id, header= False)

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

    def cat_feature(self, colname, col_no):
        """Left side"""
        desc_stats= self.read_from_pickle(os.path.join(self.desc_stats_path, colname))
        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0]} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0]} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0]} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_3= self.read_from_pickle(os.path.join(self.left_top_freq_path, colname)).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)
        
        barplot= os.path.join(self.barplot_path, f"{colname}.{self.img_format}")
        if self.standalone_html:
            img_file= open(barplot, "rb")
            img_encd= base64.b64encode(img_file.read())
            barplot= f"data:image/png;base64, {str(img_encd)[2:-1]}"
        
        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" onclick= "displayrightdiv('{self.div_class_ft_details_id}{colname}', '{self.div_class_ft_details}');">
                <p class ="{self.para_class_ft_overview_heading}">{col_no} {colname} ({self.df_feature_types[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_barplt}" src="{barplot}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_3}
                </div>
            </div>'''


        """Right side"""
        top_freq= self.read_from_pickle(os.path.join(self.top_freq_path, colname))
        right_table_3= pd.DataFrame({
            "Data": top_freq[colname],
            "Count": top_freq["count"].astype(str) + " ("+ top_freq["Count_Percentage"].astype(str)+ "%)",
        }).to_html(index= False, justify= "justify-all", table_id= self.table_ft_details_single_id, header= False)
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

    def others_feature(self, colname, col_no):
        feature_overiew_html= ""
        """Left side"""
        desc_stats= self.read_from_pickle(os.path.join(self.desc_stats_path, colname))
        table_1= pd.DataFrame({
            "Values": [f'{desc_stats["Values"].iloc[0]} ({desc_stats["Values_Percentage"].iloc[0]} %)'],
            "Missing": [f'{desc_stats["Missing"].iloc[0]} ({desc_stats["Missing_Percentage"].iloc[0]} %)'],
            "Distinct": [f'{desc_stats["Distinct"].iloc[0]} ({desc_stats["Distinct_Percentage"].iloc[0]} %)']
        }).T.to_html(index= True, justify= "justify-all", table_id= self.table_id_overview_0, header= False)
        table_3= self.read_from_pickle(os.path.join(self.left_top_freq_path, colname)).to_html(index= False, justify= "justify-all", table_id= self.table_id_overview_2, header= False)
        
        barplot= os.path.join(self.barplot_path, f"{colname}.{self.img_format}")
        if self.standalone_html:
            img_file= open(barplot, "rb")
            img_encd= base64.b64encode(img_file.read())
            barplot= f"data:image/png;base64, {str(img_encd)[2:-1]}"

        feature_overiew_html= f'''
            <div class= "{self.div_class_ft_overiew}" id="{self.div_ft_overiew_id}">
                <p class ="{self.para_class_ft_overview_heading}">{col_no} {colname} ({self.df_feature_types[colname]})</p>
                <div class= "{self.div_class_ft_overiew_img}">
                    <img class= "{self.img_class_barplt}" src="{barplot}" alt="Graph">
                </div>
                <div class= "{self.div_class_ft_overiew_tab}">
                    {table_1}{table_3}
                </div>
            </div>'''
        
        return feature_overiew_html



if __name__ == '__main__':
    pass

