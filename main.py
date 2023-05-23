from fastapi import FastAPI
# import models
# from database import engine
from routers import forecast



app = FastAPI()

# models.Base.metadata.create_all(engine) #Bikin database + tabel nya

#ROUTERS -> panggil URL nya sesuai route
app.include_router(forecast.router)





