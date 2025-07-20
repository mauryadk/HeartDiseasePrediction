import fastapi
import pickle
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
import os
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends


# Pydantic model for input data validation to ensure structure and data types.
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


app = fastapi.FastAPI()

# --- Pipeline Loading ---
pipeline = None
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
pipeline_path = os.path.join(root_dir, "model/pipeline.pkl")
try:
    with open(pipeline_path, "rb") as file:
        pipeline = pickle.load(file)
    print("Pipeline loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Pipeline file not found at {pipeline_path}")
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading the pipeline: {e}")
# --- End Pipeline Loading ---


@app.get("/")
async def root():
    return {"message": "Welcome to the Heart Disease Prediction API"}


# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Dummy user/token for demonstration
DUMMY_TOKEN = "supersecrettoken"


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, validate username/password and return a real token
    if form_data.username == "user" and form_data.password == "password":
        return {"access_token": DUMMY_TOKEN, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=400, detail="Incorrect username or password")


def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != DUMMY_TOKEN:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )
    return {"user": "user"}


# The endpoint is changed to /predict. It now accepts a JSON body via POST.
@app.post("/predict")
async def predict_disease(
    data: HeartDiseaseInput, user: dict = Depends(get_current_user)
):
    """
    Predicts the presence of heart disease based on input features.

    - **data**: A JSON object with the 13 required features for prediction.

    Returns a JSON object with the prediction: "Disease" or "No Disease".
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not available. The service is not ready.",
        )

    try:
        # Convert the Pydantic model to a numpy array in the correct order.
        # The order is guaranteed by the Pydantic model's field definition.
        input_data = np.array([list(data.dict().values())])

        # Make prediction
        prediction = pipeline.predict(input_data)

        # Interpret the prediction
        result = "Disease" if prediction[0] == 1 else "No Disease"

        return {"prediction": result}
    except Exception as e:
        # Catch any errors during the prediction process
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
