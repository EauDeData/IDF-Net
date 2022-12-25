
class NDCGEvaluation:

    '''
    
    Takes as true ranking the text embedding ranking [0, 1..., n]
    and compares it with the visual embedding ranking.
    
    '''
    def __init__(self, annoy_dataset) -> None:
        pass

class RawPrecisionEvaluation:

    '''
    
    Given a true rank (text embedding) compares the top-1 retrieved document in both text ad visual. 
    1 if both top-1 documents are the same 0 otherwise
    
    '''
    pass

class IOUEvaluation:
    '''
    
    
    Given a true ranking @K
    Calculates how many interescting documents are in visual and textual rank
    and how much big is the union of both retrieved sets.
    
    '''
    pass

class MAPEvaluation:

    '''
    idk how to manage this one
    '''
    pass