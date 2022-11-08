import pytesseract
import numpy as np
from typing import *

# https://open.spotify.com/track/2KrXkY8yRsncGk2kXPD4Zt?si=319c61fa4876459a

class TesseractOCR:

    name = 'tesseract_ocr'
    def __init__(self, lang: str = 'en', nice: int = 1, *args, **kwargs) -> None:
        self.lang = lang
        self.nice = nice

    def run(self, region: np.ndarray) -> Dict:
        '''

        Gets the OCR text given a region.
        Args: 
            region: Numpy array of the region to be transcribed.
        
        Returns: 
            Dict: {
                "result": // OCR string transciption
            }
        
        '''
        return {
            "result": pytesseract.image_to_string(region, lang = self.lang, nice=self.nice)
        }