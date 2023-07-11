from pydantic import BaseModel
from typing import List, Optional

class forecast(BaseModel):
    period_start:"str"
    period_end:"str"