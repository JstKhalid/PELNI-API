from pydantic import BaseModel
from typing import List, Optional

class forecast(BaseModel):
    dataset:"str"
    tanggal:"str"
    method: Optional[str] = None
    # outliers: "str"
    # gridsearch:"str"