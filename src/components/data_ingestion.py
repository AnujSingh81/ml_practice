import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfing:
    train_data_path :str=os.path.join("artifacts","train.csv")
    test_data_path : str=os.path.join("artifacts","test.csv")
    raw_data_path :str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfing()
        
    def initiate_data_ingestion(self):
        logging.info("Enter the ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the Data set as Data Frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_path,index= False,header=True)

            logging.info("Train Test Split initiate ")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion has been completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()