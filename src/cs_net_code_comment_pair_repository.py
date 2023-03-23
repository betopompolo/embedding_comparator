import gzip
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

import more_itertools

from json_parser import JsonParser
from models import (CodeCommentPair, CodeCommentPairRepository, DatasetType,
                    SupportedCodeLanguages)


@dataclass
class CSNetCodeCommentPairRepository(CodeCommentPairRepository):
    json_parser: JsonParser
    negative_samples_count: int = 3

    def batch(self, code_language: SupportedCodeLanguages, dataset_type: DatasetType, batch_size: int) -> Generator[List[CodeCommentPair], None, None]:
        paths = self.list_paths(
            f'resources/datasets/{code_language}/final/jsonl/{dataset_type}',
            '.jsonl.gz')

        for path in paths:
            with gzip.open(path, 'r') as file:
                for batch_lines in more_itertools.chunked(file.readlines(), round(batch_size / (self.negative_samples_count + 1))):
                    positive_samples = [self.map_pair_from_jsonl(
                        line, 1) for line in batch_lines]
                    negative_samples = self._generate_negative_samples(
                        positive_samples)

                    yield positive_samples + negative_samples

    def _generate_negative_samples(self, pairs):
        pairs_len = len(pairs)
        negative_samples: List[CodeCommentPair] = []

        for i, _ in enumerate(pairs):
            negative_samples_indexes = [self._get_random(
                0, pairs_len - 1, i) for _ in range(self.negative_samples_count)]
            for negative_index in negative_samples_indexes:
                pair = pairs[negative_index]
                negative_samples.append(
                    CodeCommentPair(pair.code, pair.comment, 0))

        return negative_samples

    def _get_random(self, start, end, exclude, attempts=100):
        for _ in range(attempts):
            random_value = random.randint(start, end)
            if random_value != exclude:
                return random_value
        raise ValueError("too many attempts")

    def list_paths(self, directory: str, file_extension: str) -> list[Path]:
        file_extension = file_extension if file_extension.startswith(
            ".") else f".{file_extension}"
        return sorted(Path(directory).glob(f'**/*{file_extension}'))

    def map_pair_from_jsonl(self, line: bytes, target: int) -> CodeCommentPair:
        json_data = self.json_parser.from_json(line)
        return CodeCommentPair(code=json_data['code'], comment=json_data['docstring'], target=target)
