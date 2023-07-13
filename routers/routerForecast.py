from fastapi import APIRouter, Depends, HTTPException, status, Response
import database, model
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
def getForecast(request: schemas.forecast, db: Session = Depends(database.get_db)):
    return repoForecast.main(request,db)

@router.get("/")
def getData(db:Session = Depends(database.get_db)):
    return repoForecast.getAllData(db)