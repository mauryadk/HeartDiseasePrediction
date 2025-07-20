import pandas as pd 
from sklearn.model_selection import train_test_split
import logging
import yaml 
import os

# logging 
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("log/data_ingestion.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(root_dir: str)-> dict:
    """creating summary for load_params

    Returns:
        [type]: [description]
    """
    try:
        with open(os.path.join(root_dir, "config/params.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    except FileNotFoundError as e:
        logging.info(e)
        raise e
    except yaml.YAMLError as e:
        logging.info(e)
        raise e
    except Exception as e:
        logging.info(e)
        raise e


def load_data(file_path: str)-> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"data loaded successfully from  path : %s", file_path)
        return df
    except FileNotFoundError as e:
        logging.info(e)
        raise e
    except pd.errors.ParserError as e:
        logging.info(e)
        raise e
    except Exception as e:
        logging.info(e)
        raise e
    

def preprocess_data(column_name: str, df: pd.DataFrame)-> pd.DataFrame:
    """ preprocess_data

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df[f'{column_name}'].str.strip() !=""]
        
        logger.info("data preprocessing completed")
        return df
    except KeyError as e:
        logging.error(e)
        raise e
    except Exception as e:
        logging.info(e)
        raise e
    

def save_data(file_path: str, df: pd.DataFrame):
    """summary for save_data

    Args:
        file_path (str): [description]
        df (pd.DataFrame): [description]
    """
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"data saved successfully at %s", file_path)
    except Exception as e:
        logging.info(e)
        raise e
    

def main():
    try:
        root_dir = os.path.abspath( os.path.join( os.path.dirname(__file__), "../"))

        params= os.path.join(root_dir, "config/params.yaml")
        params = yaml.load(open(params), Loader=yaml.FullLoader)
        
        # load data)
        df = load_data(os.path.join(root_dir, "data/heart.xls"))

        # spilit preprocessed data 
        train_data, test_data  = train_test_split(df, test_size=params['data_injestion']['test_size'], random_state=params['data_injestion']['random_state'])
        
        # save data at raw data path
        save_data(os.path.join(root_dir, "data/raw/train.csv"), train_data)
        save_data(os.path.join(root_dir, "data/raw/test.csv"), test_data)
        
        logger.info(f"data saving completed at path %s", "data/raw")

    except Exception as e:
        logging.info(e)
        raise e
   
if __name__ == "__main__":
    main()
