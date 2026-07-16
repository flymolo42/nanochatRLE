import types
import unittest
from unittest import mock
from tempfile import TemporaryDirectory

from scripts.train_phrase_vectors import iter_records
from scripts.phrase_vectors import (
    PhraseOccurrence,
    build_outputs,
    build_sparse_records,
    build_vocab,
    extract_phrase_occurrences,
    iter_tinystories_stories,
    load_spacy_model,
    normalize_story_text,
    normalize_token_text,
    iter_phrase_rows_jsonl,
    write_phrase_rows_jsonl,
    _consume_batch,
)


class FakeToken:
    def __init__(self, text, i, dep="", pos="", head=None, is_space=False):
        self.text = text
        self.i = i
        self.dep_ = dep
        self.pos_ = pos
        self.head = head if head is not None else self
        self.is_space = is_space


class FakeSpan:
    def __init__(self, tokens, label="span"):
        self._tokens = tokens
        self.start = tokens[0].i
        self.end = tokens[-1].i + 1
        self.label_ = label
        self.text = " ".join(token.text for token in tokens)

    def __iter__(self):
        return iter(self._tokens)


class FakeDoc:
    def __init__(self):
        lily = FakeToken("Lily", 0, dep="nsubj", pos="PROPN")
        found = FakeToken("found", 1, dep="ROOT", pos="VERB")
        needle = FakeToken("needle", 3, dep="dobj", pos="NOUN")
        comma = FakeToken(",", 4, dep="punct", pos="PUNCT")
        smiled = FakeToken("smiled", 6, dep="ROOT", pos="VERB")
        period = FakeToken(".", 7, dep="punct", pos="PUNCT")
        a = FakeToken("a", 2, dep="det", pos="DET", head=needle)
        she = FakeToken("she", 5, dep="nsubj", pos="PRON", head=smiled)
        lily.head = found
        found.head = found
        needle.head = found
        comma.head = found
        smiled.head = smiled
        period.head = smiled
        self._tokens = [lily, found, a, needle, comma, she, smiled, period]
        self.sents = [FakeSpan(self._tokens, label="sent")]
        self.noun_chunks = [FakeSpan([lily], label="noun_chunk"), FakeSpan([a, needle], label="noun_chunk")]

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeSpan(self._tokens[item], label="slice")
        return self._tokens[item]


class FakeNlp:
    def __init__(self):
        self.pipe_calls = []

    def pipe(self, texts, batch_size=None, n_process=None):
        self.pipe_calls.append({
            "texts": list(texts),
            "batch_size": batch_size,
            "n_process": n_process,
        })
        return [FakeDoc() for _ in self.pipe_calls[-1]["texts"]]


class ExplodingNlp:
    def pipe(self, texts, batch_size=None, n_process=None):
        raise AssertionError("spaCy should not be called when resume can use cached phrase rows")


class PhraseVectorTests(unittest.TestCase):
    def test_normalize_token_text_keeps_punctuation(self):
        self.assertEqual(normalize_token_text(types.SimpleNamespace(text="Lily"), lowercase=True), "lily")
        self.assertEqual(normalize_token_text(types.SimpleNamespace(text=","), lowercase=True), ",")

    def test_normalize_story_text_repairs_mojibake_and_ascii_punctuation(self):
        text = "â€œNo, donâ€™t run quick!â€. Annie saw a red â€” blue toy."

        self.assertEqual(
            normalize_story_text(text),
            "\"No, don't run quick!\". Annie saw a red - blue toy.",
        )

    def test_extract_phrase_occurrences_includes_punctuation_and_dependency_roles(self):
        phrases = extract_phrase_occurrences(FakeDoc())
        labels = {phrase.label for phrase in phrases}
        token_lists = [phrase.tokens for phrase in phrases]

        self.assertIn("punctuation", labels)
        self.assertIn("noun_chunk", labels)
        self.assertIn("subject", labels)
        self.assertIn("object", labels)
        self.assertIn("verb", labels)
        self.assertIn(["lily", "found", "a", "needle", ","], token_lists)
        self.assertIn(["a", "needle"], token_lists)
        self.assertIn([","], token_lists)

    def test_build_vocab_sorts_by_average_phrase_position(self):
        phrases = extract_phrase_occurrences(FakeDoc())
        vocab = build_vocab(phrases, min_count=1)
        words = [entry["token"] for entry in vocab]

        self.assertLess(words.index("lily"), words.index("found"))
        self.assertLess(words.index("found"), words.index("needle"))
        self.assertTrue(all("avg_position" in entry for entry in vocab))

    def test_build_sparse_records_emits_singletons_and_increasing_pack(self):
        phrases = [
            PhraseOccurrence(label="punctuation", tokens=["a", "cat", "."], start=0, end=3),
        ]
        vocab = [
            {"token": "a", "index": 0},
            {"token": "cat", "index": 1},
            {"token": ".", "index": 2},
        ]
        records = build_sparse_records("train", 7, phrases, vocab)

        self.assertEqual([record["record_type"] for record in records], ["single", "single", "single", "packed"])
        self.assertEqual([record["indices"] for record in records], [[0], [1], [2], [0, 1, 2]])
        self.assertEqual(records[-1]["split"], "train")
        self.assertEqual(records[-1]["story_id"], 7)
        self.assertEqual(records[-1]["values"], "implicit_ones")

    def test_build_sparse_records_accepts_reused_token_index(self):
        phrases = [
            PhraseOccurrence(label="punctuation", tokens=["a", "cat"], start=0, end=2),
        ]
        token_to_index = {"a": 0, "cat": 1}

        records = build_sparse_records("train", 7, phrases, vocab=(), token_to_index=token_to_index)

        self.assertEqual([record["indices"] for record in records], [[0], [1], [0, 1]])

    def test_build_sparse_records_keeps_singletons_when_pack_is_not_increasing(self):
        phrases = [
            PhraseOccurrence(label="punctuation", tokens=["cat", "a"], start=0, end=2),
            PhraseOccurrence(label="punctuation", tokens=["a", "a"], start=2, end=4),
            PhraseOccurrence(label="punctuation", tokens=["a", "cat"], start=4, end=6),
        ]
        vocab = [
            {"token": "a", "index": 0},
            {"token": "cat", "index": 1},
        ]

        records = build_sparse_records("train", 7, phrases, vocab)

        packed_records = [record for record in records if record["record_type"] == "packed"]
        single_records = [record for record in records if record["record_type"] == "single"]
        self.assertEqual([record["tokens"] for record in packed_records], [["a", "cat"]])
        self.assertEqual([record["indices"] for record in packed_records], [[0, 1]])
        self.assertEqual([record["indices"] for record in single_records], [[1], [0], [0], [0], [0], [1]])

    def test_iter_tinystories_stories_reads_txt_split_on_endoftext(self):
        with TemporaryDirectory() as temp_dir:
            train_path = f"{temp_dir}/TinyStories-train.txt"
            with open(train_path, "w", encoding="utf-8") as file:
                file.write("First story line.\nStill first.\n<|endoftext|>\n\nSecond story.\n<|endoftext|>\n")

            stories = list(iter_tinystories_stories(temp_dir, "train", source_format="txt"))

        self.assertEqual(stories, [(0, "First story line.\nStill first."), (1, "Second story.")])

    def test_iter_tinystories_stories_supports_v2_txt_names(self):
        with TemporaryDirectory() as temp_dir:
            valid_path = f"{temp_dir}/TinyStoriesV2-GPT4-valid.txt"
            with open(valid_path, "w", encoding="utf-8") as file:
                file.write("GPT4 validation story.\n<|endoftext|>\n")

            stories = list(iter_tinystories_stories(temp_dir, "validation", source_format="v2-txt"))

        self.assertEqual(stories, [(0, "GPT4 validation story.")])

    def test_load_spacy_model_disables_unused_components(self):
        fake_spacy = types.SimpleNamespace(load=mock.Mock(return_value="nlp"))

        with mock.patch.dict("sys.modules", {"spacy": fake_spacy}):
            nlp = load_spacy_model("en_core_web_sm", disable_components=["ner"])

        self.assertEqual(nlp, "nlp")
        fake_spacy.load.assert_called_once_with("en_core_web_sm", disable=["ner"])

    def test_consume_batch_forwards_batch_size_and_n_process_to_spacy_pipe(self):
        nlp = FakeNlp()
        rows = []
        phrases = []

        _consume_batch(
            nlp,
            "train",
            [0, 1],
            ["story one", "story two"],
            rows,
            phrases,
            lowercase=True,
            batch_size=7,
            n_process=3,
        )

        self.assertEqual(nlp.pipe_calls[0]["batch_size"], 7)
        self.assertEqual(nlp.pipe_calls[0]["n_process"], 3)
        self.assertEqual([row[1] for row in rows], [0, 1])

    def test_phrase_rows_jsonl_round_trip(self):
        with TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/phrase_rows.jsonl"
            rows = [
                ("train", 3, [PhraseOccurrence("punctuation", ["a", "."], 0, 2)]),
                ("validation", 4, [PhraseOccurrence("noun_chunk", ["red", "ball"], 5, 7)]),
            ]

            write_phrase_rows_jsonl(path, rows)
            restored = list(iter_phrase_rows_jsonl(path))

        self.assertEqual(restored, rows)

    def test_phrase_rows_jsonl_gzip_round_trip(self):
        with TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/phrase_rows.jsonl.gz"
            rows = [
                ("train", 3, [PhraseOccurrence("punctuation", ["a", "."], 0, 2)]),
            ]

            write_phrase_rows_jsonl(path, rows)
            restored = list(iter_phrase_rows_jsonl(path))

        self.assertEqual(restored, rows)

    def test_build_outputs_streaming_matches_in_memory_counts(self):
        with TemporaryDirectory() as data_dir, TemporaryDirectory() as memory_out, TemporaryDirectory() as streaming_out:
            train_path = f"{data_dir}/TinyStories-train.txt"
            valid_path = f"{data_dir}/TinyStories-valid.txt"
            story = "Lily found a needle, and she smiled.\n<|endoftext|>\n"
            with open(train_path, "w", encoding="utf-8") as file:
                file.write(story)
            with open(valid_path, "w", encoding="utf-8") as file:
                file.write(story)

            memory_meta = build_outputs(
                nlp=FakeNlp(),
                archive_path="../archive.zip",
                out_dir=memory_out,
                splits=["train", "validation"],
                limit=1,
                batch_size=2,
                tinystories_dir=data_dir,
                source_format="txt",
                streaming=False,
            )
            streaming_meta = build_outputs(
                nlp=FakeNlp(),
                archive_path="../archive.zip",
                out_dir=streaming_out,
                splits=["train", "validation"],
                limit=1,
                batch_size=2,
                tinystories_dir=data_dir,
                source_format="txt",
                streaming=True,
                n_process=1,
            )

        self.assertEqual(streaming_meta["vocab_size"], memory_meta["vocab_size"])
        self.assertEqual(streaming_meta["phrase_records"], memory_meta["phrase_records"])
        self.assertTrue(streaming_meta["streaming"])

    def test_build_outputs_streaming_writes_compressed_jsonl_and_downstream_reader_accepts_it(self):
        with TemporaryDirectory() as data_dir, TemporaryDirectory() as out_dir:
            with open(f"{data_dir}/TinyStories-train.txt", "w", encoding="utf-8") as file:
                file.write("Lily found a needle.\n<|endoftext|>\n")

            metadata = build_outputs(
                nlp=FakeNlp(),
                archive_path="../archive.zip",
                out_dir=out_dir,
                splits=["train"],
                limit=1,
                batch_size=2,
                tinystories_dir=data_dir,
                source_format="txt",
                streaming=True,
                compress=True,
            )
            records = list(iter_records(f"{out_dir}/phrase_index.jsonl.gz"))

        self.assertTrue(metadata["compressed"])
        self.assertTrue(metadata["phrase_rows_cache"].endswith(".jsonl.gz"))
        self.assertTrue(records)

    def test_build_outputs_resume_rebuilds_from_cached_phrase_rows_without_spacy(self):
        with TemporaryDirectory() as data_dir, TemporaryDirectory() as out_dir:
            with open(f"{data_dir}/TinyStories-train.txt", "w", encoding="utf-8") as file:
                file.write("Lily found a needle.\n<|endoftext|>\n")
            first = build_outputs(
                nlp=FakeNlp(),
                archive_path="../archive.zip",
                out_dir=out_dir,
                splits=["train"],
                limit=1,
                batch_size=2,
                tinystories_dir=data_dir,
                source_format="txt",
                streaming=True,
                compress=True,
            )
            second = build_outputs(
                nlp=ExplodingNlp(),
                archive_path="../archive.zip",
                out_dir=out_dir,
                splits=["train"],
                limit=1,
                batch_size=2,
                tinystories_dir=data_dir,
                source_format="txt",
                streaming=True,
                compress=True,
                resume=True,
            )

        self.assertEqual(second["phrase_records"], first["phrase_records"])
        self.assertTrue(second["resumed"])


if __name__ == "__main__":
    unittest.main()
