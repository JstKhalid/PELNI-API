from pydantic import BaseModel
from typing import List, Optional

class forecast(BaseModel):
    dataset:"str"
    tanggal:"str"
    method: Optional[str] = "XGB"
    # outliers: "str"
    # gridsearch:"str"