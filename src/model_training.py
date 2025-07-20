from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import yaml
import logging
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


# logging
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("log/model_training.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(root_dir: str) -> dict:
    """creating summary for load_params

    Returns:
        [type]: [description]
    """
    try:
        with open(os.path.join(root_dir, "config/params.yaml"), "r") as file:
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


def load_data(file_path: str) -> pd.DataFrame:
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """preprocess_data

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        logger.info("data preprocessing completed")
        return df
    except KeyError as e:
        logging.error(e)
        raise e
    except Exception as e:
        logging.info(e)
        raise e


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, model_save_path: str
) -> None:
    """Trains a Pipeline (StandardScaler + LogisticRegression) and saves it."""
    try:
        logger.info("--- Starting model training ---")
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=100000))]
        )
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline training completed.")

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, "wb") as file:
            pickle.dump(pipeline, file)
        logger.info(f"Pipeline saved successfully to {model_save_path}")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise e


def test_model(
    X_test: pd.DataFrame, y_test: pd.DataFrame, model_path: str, metrics_path: str
) -> None:
    """Evaluates the trained model on the test set and saves metrics/plots.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test labels.
        model_path (str): Path to the saved model file.
        metrics_path (str): Path to save the confusion matrix plot.
    """
    try:
        logger.info("--- Starting model evaluation on test data ---")
        with open(model_path, "rb") as file:
            pipeline = pickle.load(file)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

        report_cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Test Confusion Matrix:\n{report_cm}")

        class_report = classification_report(y_test, y_pred)
        logger.info(f"Test Classification Report:\n{class_report}")

        # Create and save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(report_cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        plt.savefig(metrics_path)
        logger.info(f"Confusion matrix plot saved to {metrics_path}")
        plt.close()  # Close the plot to free up memory

    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during model testing: {e}")
        raise e


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    params = load_params(root_dir)

    # Define paths
    train_data_path = os.path.join(root_dir, "data/raw/train.csv")
    test_data_path = os.path.join(root_dir, "data/raw/test.csv")
    model_save_path = os.path.join(root_dir, "model/pipeline.pkl")
    metrics_path = os.path.join(root_dir, "reports/confusion_matrix.png")


    # Load and preprocess data
    logger.info("Loading training and testing data.")
    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)

    logger.info("Preprocessing data.")
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Separate features and target
    target_column = "target"
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    # Train the pipeline
    train_model(X_train, y_train, model_save_path)

    # Evaluate the pipeline
    test_model(X_test, y_test, model_save_path, metrics_path)


if __name__ == "__main__":
    main()
