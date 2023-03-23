from dataclasses import dataclass

from code_comment_pair_repository import (CodeCommentPairRepository,
                                          SupportedCodeLanguages)
from embedding_comparator import EmbeddingComparator
from embedding_generator import EmbeddingGenerator
from models import DatasetType, Experiment


@dataclass
class EmbeddingComparatorDefinitionExperiment(Experiment):
    code_comment_repo: CodeCommentPairRepository
    embedding_generator: EmbeddingGenerator
    embedding_comparator: EmbeddingComparator

    async def run(self):
        python_pairs = self.code_comment_repo.list(
            SupportedCodeLanguages.python,
            DatasetType.train,
        )

        pairs_count = 200
        batch_size = 64
        dataset = self.embedding_generator.generate(
            python_pairs, 
            pairs_count=pairs_count, 
            batch_size=batch_size,
        )
        self.embedding_comparator.fit(dataset)
    