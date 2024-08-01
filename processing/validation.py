# processing/validation.py
from pydantic import BaseModel

class IrisDataSchema(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str
