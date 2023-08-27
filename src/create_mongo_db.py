from typing import Dict, List, TypedDict

from bson import ObjectId
import more_itertools
from tqdm import tqdm
from cs_net_parser import CSNetCodeLanguage, CSNetParser, CSNetPartition, CSNetQuery, CSNetQueryLanguage, CSNetSample
from mongo_db_client import MongoDbClient, MongoDbPairDoc, MongoDbQueryDoc, MongoId


class PairFilter(TypedDict):
  partition: CSNetPartition
  language: CSNetCodeLanguage

class CreateMongoDb:
  def __init__(self, pair_filters: List[PairFilter]) -> None:
    self.pair_filters = pair_filters
    self.cs_net_parser = CSNetParser()


  def run(self):
    db_client = MongoDbClient()
    pairs_batch_size = 100
    queries_batch_size = 100

    for filter in self.pair_filters:
      language, partition = filter['language'], filter['partition']

      with tqdm(desc=f"Importing {language} pairs from {partition} partition to MongoDB", total=self.cs_net_parser.get_dataset_len(partition=partition, language=language)) as progress_bar:
        for samples_batch in more_itertools.chunked(self.cs_net_parser.get_samples(partition, language), pairs_batch_size):
          pairs_documents: List[MongoDbPairDoc] = [self.__to_pair(sample) for sample in samples_batch]
          db_client.get_pairs_collection().insert_many(pairs_documents)
          progress_bar.update(len(samples_batch))


    with tqdm(desc=f"Importing CodeSearchNet queries that match imported pairs") as progress_bar:
      for queries_batch in more_itertools.chunked(self.cs_net_parser.get_queries(), queries_batch_size):
        query_urls = [query['GitHubUrl'] for query in queries_batch]
        pairs_ids = { pair['github_url']: pair['_id'] for pair in db_client.get_pairs_collection().find({ "github_url": { "$in": query_urls } }) }
        if len(pairs_ids) == 0:
          continue

        filtered_queries = [query for query in queries_batch if query['GitHubUrl'] in pairs_ids]
        queries_documents: List[MongoDbQueryDoc] = [self.__to_query(query, pair_id=pairs_ids[query['GitHubUrl']]) for query in filtered_queries]

        db_client.get_queries_collection().insert_many(queries_documents)
        progress_bar.update(len(queries_documents))

  def __to_pair(self, sample: CSNetSample) -> MongoDbPairDoc:
    pair_id = ObjectId()
    return {
      "_id": pair_id,
      "id": str(pair_id),
      'code_tokens': sample['code_tokens'],
      'comment_tokens': sample['docstring_tokens'],
      'github_url': sample['url'],
      'language': sample['language'],
      'partition': sample['partition'],
    }
  
  def __to_query(self, sample: CSNetQuery, pair_id: MongoId) -> MongoDbQueryDoc:
    __language_dict: Dict[CSNetQueryLanguage, CSNetCodeLanguage] = {
      'Ruby': 'ruby',
      'Go': 'go',
      'Java': 'java',
      'JavaScript': 'javascript',
      'PHP': 'php',
      'Python': 'python',
    }
    
    return {
      "pair_id": pair_id,
      'github_url': sample['GitHubUrl'],
      'language': __language_dict[sample['Language']],
      'notes': sample['Notes'],
      'query': sample['Query'],
      'relevance': sample['Relevance']
    }
