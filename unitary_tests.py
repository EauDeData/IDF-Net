from src.text.preprocess import StringCleanAndTrim
from src.utils.errors import *

if __name__ == '__main__': 
    try:
        cleaner_obj = StringCleanAndTrim()
        returned = (cleaner_obj(["I'm Diffie, congrats!", "Hello, why?", "Nick likes to play football, however he is not too fond of tennis."]))
        if not isinstance(returned, list): raise WrongTypeReturnedeError
    except Exception as e:
        print(f"Preprocess test not passes, reason: {e}") 
