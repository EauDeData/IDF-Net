from sklearn.metrics import average_precision_score

class NDCGEvaluation:

    '''
    
    Takes as true ranking the text embedding ranking [0, 1..., n]
    and compares it with the visual embedding ranking.
    
    '''
    def __init__(self) -> None:
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
    Use categories for MAP Evaluation maybe.
    It should consider only 1 modal (visual and textual separately) and hopefully it should be the same
    
    '''
    def __init__(self, test_set, train_set, annoyer, k = 5, device = 'cuda') -> None:
        
        self.dataset = test_set
        self.train_set = train_set
        self.annoyer = annoyer
        self.device = device
        self.k = k

    def run(self):

        # TODO: This should be 10 times faster
        buffer_visual = 0
        buffer_textual = 0
        print("Running MAP Evaluation...")
        for idx in range(len(self.dataset)):
            print(idx, '\t', end = '\r')

            data, textual, category = self.dataset.get_with_category(idx) # TODO: Double-check the returned categories belong to the image (offset issues?)
            retrieved, distances = self.annoyer.retrieve_image(data, self.k)
            retrieved_textual, distances_textual = self.annoyer.retrieve_vector(textual, self.k)

            category = set(category)

            labels = [0 for _ in retrieved]
            labels_textual = [0 for _ in retrieved]
            for num, (item, item_textual) in enumerate(zip(retrieved, retrieved_textual)):
                
                cats = self.train_set.get_only_category(item)
                cats_textual = self.train_set.get_only_category(item_textual)

                labels[num] = bool(len(category.intersection(cats)))
                labels_textual[num] = bool(len(category.intersection(cats_textual)))

                if (sum(labels)): buffer_visual += average_precision_score(labels, distances)
                if (sum(labels_textual)): buffer_textual += average_precision_score(labels_textual, distances_textual)
        
        return {
            'visual-map': buffer_visual / (idx + 1),
            'textual-map': buffer_textual / (idx + 1)
        }