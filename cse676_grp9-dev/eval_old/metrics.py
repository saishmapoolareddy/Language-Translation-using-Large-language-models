import re
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import ROUGEScore

class Metrics:
    def __init__(self) -> None:
        pass

    def clean(self,gt,mt):
        gts = []
        mts = []
        for i in gt:
            gt_i = i
            gt_i = re.sub(r'[\W]',' ',gt_i)[:-1]
            # gt_i = ' '.join(gt_i.split(' ')[:-3])

            gts.append(gt_i)

        for i in mt:
            mt_i = i
            mt_i = re.sub(r'[\W]',' ',mt_i)
            mt_i = mt_i.split(' ')
            mt_i = mt_i[:-1]
            if len(mt_i)==0:
                mt_i = [""]
            mts.append(' '.join(mt_i))

        # print('\n\n')
        # for i,(x,y,z,w) in enumerate(zip(gt,gts,mt,mts),0):
        #     if i<5:
        #         print(f"({x}|{y}) | ({z}|{w})")
        #         print('-'*20)

        gts = np.array(gts)
        mts = np.array(mts)

        return gts,mts
    
    
class CosineSimilarity(Metrics):
    def __init__(self,batch_size) -> None:
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.batch_size = batch_size

    def evaluate(self,gt_clean,mt_clean):
        # gt_clean,mt_clean = self.clean(ground_truth,neural_translation)

        
        embeddings = self.model.encode(gt_clean)

        mt_embeddings = self.model.encode(mt_clean)


        cos_sim = cosine_similarity(embeddings,mt_embeddings)

        # mean of diagonals
        diags = np.clip(np.diagonal(cos_sim),-1,1)
        mean = np.mean(diags)
        max_diag = np.max(diags)
        min_diag = np.min(diags)

        return (mean,max_diag,min_diag)
    
class BLEUScore(Metrics):
    def __init__(self,batch_size) -> None:
        self.batch_size = batch_size

    def evaluate(self,gt_clean,mt_clean):
        # gt_clean,mt_clean = self.clean(ground_truth,neural_translation)
        bleu_scores = []
        for g,m in zip(gt_clean,mt_clean):
            score = sentence_bleu(m,[g]).score
            score = np.clip(score/100,0,1)
            bleu_scores.append(score)
            

        bleu_scores = np.array(bleu_scores)
        mean = np.mean(bleu_scores)
        max_diag = np.max(bleu_scores)
        min_diag = np.min(bleu_scores)
        

        return (mean,max_diag,min_diag)

class METEORScore(Metrics):
    def __init__(self,batch_size) -> None:
        self.batch_size = batch_size

    def evaluate(self,gt_clean,mt_clean):
        # gt_clean,mt_clean = self.clean(gt,mt)
        meteor_scores = []
        for g,m in zip(gt_clean,mt_clean):
            g = g.split(' ')
            m = m.split(' ')
            meteor_scores.append(meteor_score(references=[g],hypothesis=m))

        meteor_scores = np.array(meteor_scores)
        mean = np.mean(meteor_scores)
        max_diag = np.max(meteor_scores)
        min_diag = np.min(meteor_scores)
        

        return (mean,max_diag,min_diag)

class ROUGEScore_custom(Metrics):
    def __init__(self,batch_size) -> None:
        self.batch_size = batch_size
        self.scorer = ROUGEScore()

    def evaluate(self,gt_clean,mt_clean):
        # gt_clean,mt_clean = self.clean(gt,mt)
        rouge_scores = []
        for g,m in zip(gt_clean,mt_clean):
            # g = g.split(' ')
            # m = m.split(' ')
            # score = self.scorer(m,g)
            self.scorer.update(m,g)
            score = self.scorer.compute()
            self.scorer.reset()

            f1 = score['rougeLsum_fmeasure']
            rouge_scores.append(f1)

        rouge_scores = np.array(rouge_scores)
        mean = np.mean(rouge_scores)
        max_diag = np.max(rouge_scores)
        min_diag = np.min(rouge_scores)
        

        return (mean,max_diag,min_diag)