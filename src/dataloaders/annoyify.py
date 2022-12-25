import annoy

class Annoyifier:

    '''
    Creates an annoy instance given a dataset (train data)
    for both visual and textual models.
    This will be used for evaluating ranks (visual) with respect a true rank (textual)
    

    WARNING: Dataset has to be loader with its tokenizer so it returns the text embedding properly.
    '''

    def __init__(self, train_set, visual_model, distance = 'Angular') -> None:

        pass