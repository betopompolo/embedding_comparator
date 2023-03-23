from tqdm import tqdm
from cs_net_code_comment_pair_repository import CSNetCodeCommentPairRepository
from embedding_comparator import EmbeddingComparator
from embedding_generator import EmbeddingGenerator
from generate_embeddings_dataset import GenerateEmbeddingsDataset
from json_parser import OrJsonParser

from transformers import AutoTokenizer, TFAutoModel

from models import DatasetType, SupportedCodeLanguages


if __name__ == '__main__':
    batch_size = 64
    dataset_len = {
        "test": {
            "java": 26909,
            "python": 22176
        },
        "train": {
            "java": 454451,
            "python": 412178
        },
        "valid": {
            "java": 15328,
            "python": 23107
        }
    }

    repo = CSNetCodeCommentPairRepository(json_parser=OrJsonParser())
    embedding_generator = EmbeddingGenerator(
        text_embedding_model=TFAutoModel.from_pretrained(
            "bert-base-uncased"),
        text_tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        code_embedding_model=TFAutoModel.from_pretrained(
            "microsoft/codebert-base", ),
        code_tokenizer=AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"),
    )

    dataset = GenerateEmbeddingsDataset(
        batch_size=batch_size,
        dataset_type=DatasetType.train,
        programming_language=SupportedCodeLanguages.java,
        embedding_generator=embedding_generator,
        pair_repository=repo,
    )

    # with tqdm(total=320, desc='writing embeddings') as pbar:
    #     dataset.write(on_write=lambda x: pbar.update(x))

    # model = EmbeddingComparator()
    for input, target in dataset.read():
        pass

    # model.fit(dataset.read(), batch_size=batch_size)

    # with tqdm(total=320, desc='reading embeddings') as pbar:
    #     for input, target in dataset.read():
    #         pbar.update(batch_size)
