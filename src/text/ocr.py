import pytesseract
import numpy as np
from typing import *
from easyocr import Reader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from mmocr.utils.ocr import MMOCR
from PIL import Image
import cv2

# https://open.spotify.com/track/2KrXkY8yRsncGk2kXPD4Zt?si=319c61fa4876459a

class TesseractOCR:

    name = 'tesseract_ocr'
    def __init__(self, lang: str = 'eng', nice: int = 10, *args, **kwargs) -> None:
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
        region = cv2.adaptiveThreshold(region,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        return {
            "result": pytesseract.image_to_string(region, lang = self.lang, nice=self.nice)
        }


class MMOCRWrapper:

    name = 'mm_ocr'
    def __init__(self, lang: str = 'eng', nice: int = 10, *args, **kwargs) -> None:
        self.model = MMOCR()



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
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
        print(region.shape)

        return {
            "result": self.model.readtext(region)
        }

class MsOCR:

    name = 'ms_ocr'
    def __init__(self, lang: str = 'eng', nice: int = 10, *args, **kwargs) -> None:
        self.prep = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed") 
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")



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
        processor, model = self.prep, self.model

        image = Image.fromarray(region).convert('RGB')
        pixel_values = processor(image, return_tensors="pt").pixel_values 
        generated_ids = model.generate(pixel_values)
        return {
            "result": processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
        }

class EasyOCR:

    name = 'easyOCR'
    def __init__(self, lang: str = 'en', *args, **kwargs) -> None:
        self.lang = lang
        self.reader = Reader([self.lang], recog_network='latin_g1',)
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
        region = cv2.adaptiveThreshold(region,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        results = self.reader.readtext(region)
        
        return {
            "result": " ".join([text for (_, text, _) in results if len(text)])
        }

