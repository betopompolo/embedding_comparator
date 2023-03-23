from itertools import islice
import random
import more_itertools
from cs_net_code_comment_pair_repository import CSNetCodeCommentPairRepository
from models import SupportedCodeLanguages, DatasetType
from json_parser import OrJsonParser

repo = CSNetCodeCommentPairRepository(json_parser=OrJsonParser())


def get_random(start, end, exclude, attempts=100):
    for _ in range(attempts):
        random_value = random.randint(start, end)
        if random_value != exclude:
            return random_value
    raise ValueError("ops..")


negative_samples_count = 3
dataset_count = 500
batch_size = 64
gen = islice(repo.list(SupportedCodeLanguages.python,
             DatasetType.train), dataset_count)


def get_samples(batched_pairs):
    for i, pair in enumerate(batched_pairs):
        negative_samples_indexes = [get_random(
            0, len(batched_pairs) - 1, i) for _ in range(negative_samples_count)]
        positive_sample = f'positive{pair.comment[0:2]}'
        negative_samples = [
            f'negative{batched_pairs[n].comment[0:2]}' for n in negative_samples_indexes]
        if positive_sample in negative_samples:
            raise ValueError("Waaat")


for batch in more_itertools.chunked(gen, batch_size):
    get_samples(batch)
print("done!")
