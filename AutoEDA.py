import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from AutoEDAHelper import AutoEDAHelper



class AutoEDA():
    def __init__(self, df_path, htmlfilename= "index.html", **karwas):
        '''Any keyarugemts needed to read the datafile send it inside 'datafile' keyargument'''
        self.df= pd.DataFrame()
        self.no_of_steps= 4
        self.__read_from_file(df_path)
        
        self.__constrains_for_df_extract__(htmlfilename)
        self.create_html()
        

    def __read_from_file(self, df_path):
        print(f"step 1/{self.no_of_steps}: Loading the Dataframe in memory")
        if type(df_path)== pd.core.frame.DataFrame:
            self.df= df_path
        elif not os.path.exists(df_path):
            raise ValueError('Not a valid path or pandas dataframe')
        else:
            try:
                df_path= os.path.normpath(df_path)
                if df_path.lower().endswith('.csv'):
                    self.df= pd.read_csv(df_path)
                elif df_path.lower().endswith('.xml'):
                    self.df= pd.read_xml(df_path)
                elif df_path.lower().endswith('.table'):
                    self.df= pd.read_table(df_path)
                elif df_path.lower().endswith('.json'):
                    self.df= pd.read_json(df_path)
                elif df_path.lower().endswith(tuple(['xls' , 'xlsx' , 'xlsm' , 'xlsb' , 'odf' , 'ods', 'odt'])):
                    self.df= pd.read_excel(df_path)
                elif df_path.lower().endswith('.html'):
                    self.df= pd.read_html(df_path)
                else:
                    raise ValueError('Not a valid file format, Supported file types are: csv, xml, table, json, html, excel')
            except Exception as e:
                print(e)

    def __constrains_for_df_extract__(self, htmlfilename):
        self.html_out_file= htmlfilename
        self.datatypes_for_corr_plot= [np.number, 'datetime64', 'category', 'bool']
        self.img_type= 'png'
        self.pairplot_sample_size= 50
        self.floating_point_limit= 3

    def write_html_header(self, df_overiew_html, df_details_html):
        '''This function wirtes dataframe overview and css file in the html file'''
        html_content_start= '<!DOCTYPE html><html><head><link rel="stylesheet" href="AutoEDA.css"><title>AutoEDA</title></head><body>'
        html_file = open(self.html_out_file, "w")
        html_file.write(html_content_start+ df_overiew_html+ df_details_html)
        html_file.close()
        
    def write_html_tail(self):
        '''This function wirtes the end of the html file and javascripts file in the html file'''
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

    def create_html(self):
        #HTML df overiew:
        print(f"step 2/{self.no_of_steps}: Processing dataframe values")
        auto_eda_helper= AutoEDAHelper(self.df)
        df_overiew_html, df_details_html= auto_eda_helper.df_overview_as_html(self.datatypes_for_corr_plot, self.pairplot_sample_size, self.floating_point_limit, self.img_type)
        self.write_html_header(df_overiew_html, df_details_html)

        #HTML write all feature details:
        print(f"step 3/{self.no_of_steps}: Extracting information for each columns")
        col_and_type_mapping= auto_eda_helper.get_features_type
        for colname in (pbar:= tqdm(col_and_type_mapping)):
            pbar.set_postfix_str(colname)
            df_overiew_html, df_details_html= auto_eda_helper.feacture_as_html(colname)
            html_file = open(self.html_out_file, "a")
            html_file.write("\n"+df_overiew_html+"\n")
            html_file.write("\n"+df_details_html+"\n")
            html_file.close()

        #HTML write tail: -js, etc
        print(f"step 4/{self.no_of_steps}: Wrapping up html")
        self.write_html_tail()


if __name__ == '__main__':
    # dataset_file= R"D:\Scaler\projects\automated_eda\dataset_in\Online Retail II UCI\online_retail_II.csv"
    # dataset_file= R"D:\Scaler\projects\automated_eda\dataset_in\Credit EDA Case Study\application_data.csv"
    # dataset_file= R"D:\Scaler\projects\automated_eda\dataset_in\basketball_players.csv"
    dataset_file= R"D:\Scaler\projects\automated_eda\dataset_in\Divvy_Trips_2019_Q1.xlsx\Divvy_Trips_2019_Q1.xlsx"
    auto_eda= AutoEDA(dataset_file)

