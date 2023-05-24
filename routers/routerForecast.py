from fastapi import APIRouter, Depends, HTTPException, status, Response
# import database, models, schemas, oauth2
import schemas
from sqlalchemy.orm import Session
from typing import List
# from repository import blog
from repositories import repoForecast
#Pake Tags buat dokumentasi nya mudah dibaca & prefix supaya URL nya ga berulang ulang ditulis
router = APIRouter(
    tags=['Forecast'],
    prefix="/forecast"
) 

@router.post("/",)
def getForecast(request: schemas.forecast):
    return repoForecast.main(request)