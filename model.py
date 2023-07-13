#models ini sama kaya schema cuman buat bikin model database
from sqlalchemy import Column, Integer, String
from database import Base

class forecast(Base):
    __tablename__ = "hist_pax_revenue" #nama tabel
    
    id = Column(Integer,primary_key = True) 
    departure_date = Column(String)
    kode_org = Column(Integer)
    kode_des = Column(Integer)
    type_rev = Column(Integer)
    revenue_cargo = Column(Integer)
    revenue_pax = Column(Integer)
