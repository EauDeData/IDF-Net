# IDF-Net

## TODO: 

* Topic Models
> TF-IDF


> SVD On GloVe


> Find more topic models

* Loss related
 
> Work around the already done 2-clique- loss


> Try an implementation of our clique loss


> Try NNs Loss (https://openaccess.thecvf.com/content/ICCV2021/papers/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.pdf) but with fixed topic model (frozen network, text embedding, just GT...) you can change Z+k by sim(Zi, Zk) in order to express continous relevance if sim = [0, 1)

> Fundamental matrix approach?? (pFp' has to be 0, if its higher it points out the error, can we compute F since we know the pairs???)

* Evaluation

> Compute evluation metrics
