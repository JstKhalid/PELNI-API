from fastapi import FastAPI
import model
from database import engine
from routers import routerForecast



app = FastAPI()

model.Base.metadata.create_all(engine) #Bikin database + tabel nya

#ROUTERS -> panggil URL nya sesuai route
app.include_router(routerForecast.router)





