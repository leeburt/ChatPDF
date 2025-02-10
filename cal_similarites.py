from typing import List, Union, Dict
from loguru import logger
import json
import os 
from similarities import (
    EnsembleSimilarity,
    BertSimilarity,
    BM25Similarity,
)
from similarities.similarity import SimilarityABC

##重写BertSimilarity
class MyBertSimilarity(BertSimilarity):
    def __init__(
            self,
            corpus: Union[List[str], Dict[int, str]] = None,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            device: str = None,
            save_path: str = None,
            save_name: str = "bert_corpus_emb.jsonl",
    ):
        self.corpus_doc_name={}
        self.save_path=save_path
        self.save_name=save_name
        super().__init__(corpus,model_name_or_path,device)

    def add_corpus(
                self,
                corpus: Union[List[str], Dict[int, str]],
                corpus_doc_name: Union[List[str], Dict[int, str]] = None,
                batch_size: int = 32,
                normalize_embeddings: bool = True
        ):
            """
            Extend the corpus with new documents.
            :param corpus: corpus of documents to use for similarity queries.
            :param batch_size: batch size for computing embeddings
            :param normalize_embeddings: normalize embeddings before computing similarity
            :return: corpus, corpus embeddings
            """
            if os.path.exists(os.path.join(self.save_path,self.save_name)):
                self.load_corpus_embeddings(os.path.join(self.save_path,self.save_name))
                logger.debug(f"load doc cache for BertSimilarity, total chunks: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")
            else:
                corpus_new = {}
                corpus_doc_name_new={}
                start_id = len(self.corpus) if self.corpus else 0
                for id, doc in enumerate(corpus):
                    if isinstance(corpus, list):
                        if doc not in self.corpus.values():
                            corpus_new[start_id + id] = doc
                            corpus_doc_name_new[start_id + id]=corpus_doc_name[id]
                    else:
                        if doc not in self.corpus.values():
                            corpus_new[id] = doc
                            corpus_doc_name_new[id]=corpus_doc_name[id]

                if not corpus_new:
                    return
                self.corpus.update(corpus_new)
                self.corpus_doc_name.update(corpus_doc_name_new)
                corpus_embeddings = self.get_embeddings(
                    list(corpus_new.values()),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                ).tolist()
                if self.corpus_embeddings:
                    self.corpus_embeddings = self.corpus_embeddings + corpus_embeddings
                else:
                    self.corpus_embeddings = corpus_embeddings
                self.save_corpus_embeddings(os.path.join(self.save_path,self.save_name))
                logger.debug(f"Add {len(corpus_new)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10) -> List[List[Dict]]:
        old_return = super().most_similar(queries,topn)
        new_return=[]
        for c in old_return:
            q_info=[]
            for info in c:
                corpus_id=info['corpus_id']
                info['doc_name']=self.corpus_doc_name.get(corpus_id,"")
                q_info.append(info)
            new_return.append(q_info)
        return new_return
    
    def save_corpus_embeddings(self, emb_path: str = "bert_corpus_emb.jsonl"):
        """
        Save corpus embeddings to jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        with open(emb_path, "w", encoding="utf-8") as f:
            for id, emb in zip(self.corpus.keys(), self.corpus_embeddings):
                json_obj = {"id": id, "doc": self.corpus[id],"doc_name":self.corpus_doc_name[id], "doc_emb": list(emb)}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        logger.info(f"Save corpus embeddings to file: {emb_path}.")

    def load_corpus_embeddings(self, emb_path: str = "bert_corpus_emb.jsonl"):
        """
        Load corpus embeddings from jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                corpus_embeddings = []
                for line in f:
                    json_obj = json.loads(line)
                    self.corpus[int(json_obj["id"])] = json_obj["doc"]
                    self.corpus_doc_name[int(json_obj["id"])] = json_obj["doc_name"]
                    corpus_embeddings.append(json_obj["doc_emb"])
                self.corpus_embeddings = corpus_embeddings
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")
        except Exception as e:
            logger.error(f"Error: {e}")

##重写BM25Similarity方法
class MyBM25Similarity(BM25Similarity):
    """
    Compute BM25OKapi similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None,save_path: str = None,save_name: str = "bm25_corpus_emb.jsonl"):
        self.corpus_doc_name={}
        self.save_path=save_path
        self.save_name=save_name
        super().__init__(corpus)
        
    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]
                   ,corpus_doc_name: Union[List[str], Dict[int, str]] = None,):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : dict
        """
        if os.path.exists(os.path.join(self.save_path,self.save_name)):
            self.load_corpus_embeddings(os.path.join(self.save_path,self.save_name))
            logger.debug(f"load doc cache for BM25Similarity, total chunks: {len(self.corpus)}")
        else:
            corpus_new = {}
            corpus_doc_name_new={}
            start_id = len(self.corpus) if self.corpus else 0
            if isinstance(corpus, list):
                for id, doc in enumerate(corpus):
                    if doc not in list(self.corpus.values()):
                        corpus_new[start_id + id] = doc
                        corpus_doc_name_new[start_id + id]=corpus_doc_name[id]
            else:
                for id, doc in corpus.items():
                    if doc not in list(self.corpus.values()):
                        corpus_new[id] = doc
                        corpus_doc_name_new[id]=corpus_doc_name[id]
            
            if not corpus_new:
                return
            self.corpus.update(corpus_new)
            self.corpus_doc_name.update(corpus_doc_name_new)
            self.save_corpus_embeddings(os.path.join(self.save_path,self.save_name))
            
            logger.debug(f"Add corpus done, new docs: {len(corpus_new)}, all corpus size: {len(self.corpus)}")
        self.build_bm25()

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10) -> List[List[Dict]]:
        old_return = super().most_similar(queries,topn)
        new_return=[]
        for c in old_return:
            q_info=[]
            for info in c:
                corpus_id=info['corpus_id']
                info['doc_name']=self.corpus_doc_name.get(corpus_id,"")
                q_info.append(info)
            new_return.append(q_info)
        return new_return
    
    def save_corpus_embeddings(self, emb_path: str = "BM25_corpus_emb.jsonl"):
        """
        Save corpus embeddings to jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        with open(emb_path, "w", encoding="utf-8") as f:
            for id in self.corpus.keys():
                json_obj = {"id": id, "doc": self.corpus[id],"doc_name":self.corpus_doc_name[id]}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        logger.info(f"Save corpus embeddings to file: {emb_path}.")

    def load_corpus_embeddings(self, emb_path: str = "BM25_corpus_emb.jsonl"):
        """
        Load corpus embeddings from jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                for line in f:
                    json_obj = json.loads(line)
                    self.corpus[int(json_obj["id"])] = json_obj["doc"]
                    self.corpus_doc_name[int(json_obj["id"])] = json_obj["doc_name"]
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")
        except Exception as e:
            logger.error(f"Error: {e}")

class MyEnsembleSimilarity(EnsembleSimilarity):
    """
    Compute similarity score between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[int, str]] = None,
            similarities: List[SimilarityABC] = None,
            weights: List[float] = None,
            c: int = 60,
    ):
        self.corpus_doc_name={}
        super().__init__(corpus,similarities,weights,c)

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]
                   ,corpus_doc_name: Union[List[str], Dict[int, str]] = None):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str or dict of str
        """
        for i in self.similarities:
            i.add_corpus(corpus,corpus_doc_name)
        self.corpus = self.similarities[0].corpus
        self.corpus_doc_name = self.similarities[0].corpus_doc_name

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10) -> List[List[Dict]]:
        old_return = super().most_similar(queries,topn)
        new_return=[]
        for c in old_return:
            q_info=[]
            for info in c:
                corpus_id=info['corpus_id']
                info['doc_name']=self.corpus_doc_name.get(corpus_id,"")
                q_info.append(info)
            new_return.append(q_info)
        return new_return



