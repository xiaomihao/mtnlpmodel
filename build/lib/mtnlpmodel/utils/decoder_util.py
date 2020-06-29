from typing import List
from tokenizer_tools.tagset.NER.base_tagset import BaseTagSet
from tokenizer_tools.tagset.exceptions import TagSetDecodeError
from tokenizer_tools.tagset.offset.document import Document
from tokenizer_tools.tagset.offset.span import Span
from tokenizer_tools.tagset.offset.span_set import SpanSet


def tags_to_span_set(tags: List[str]) -> SpanSet:
    decoder = BIOSequenceEncoderDecoder()
    span_info = decoder.decode_to_offset(tags)

    return SpanSet([Span(i[0], i[1], i[2]) for i in span_info])


class BIOEncoderDecoder(BaseTagSet):
    """
    Encoder and Decoder for BILUO scheme
    """

    # O is very easy confused with zero, using oscar instead in the code
    oscar = "O"

    def generate_tag(self, prefix):
        if self.tag_name == self.oscar:
            # O tag is very special, it always return O
            return self.oscar

        if self.tag_name is not None:
            tag = "{}-{}".format(prefix, self.tag_name)
        else:
            # if tag_name is None, no more tag_name in tag
            tag = prefix

        return tag

    def encode(self, sequence):
        len_of_sequence = len(sequence)

        if len_of_sequence == 1:
            return [self.generate_tag("B")]

        elif len_of_sequence == 2:
            return [self.generate_tag("B"), self.generate_tag("I")]

        else:
            return (
                [self.generate_tag("B")]
                + [self.generate_tag("I")] * (len_of_sequence - 2)
                + [self.generate_tag("I")]
            )

    def decode(self, sequence):
        pass

    def all_tag_set(self):
        tag_set = {self.generate_tag(i) for i in "BI"}
        tag_set_oscar = {self.oscar}
        tag_set.update(tag_set_oscar)
        return tag_set


class BIOSequenceEncoderDecoder(object):
    # O is very easy confused with zero, using oscar instead in the code
    oscar = "O"

    legal_set = {
        ("B", "I"),
        ("I", "I"),
        (oscar, "B"),
        ("I", oscar),
        ("B", oscar),
        ("I", "B"),
        ("B", "B"),
    }

    prefix_set = set("BI")

    def __init__(self, *args, **kwargs):
        self.ignore_error = kwargs.get("ignore_error", True)

    def parse_tag(self, tag):
        # TODO: already replaced by inline code, remove me in later version
        if tag == self.oscar:
            return self.oscar, None

        # set maxsplit to 1, so raw_tag_name can contains '-' char legally
        raw_prefix, raw_tag_name = tag.split("-", maxsplit=1)

        prefix = raw_prefix.strip()
        tag_name = raw_tag_name.strip()

        if prefix and tag_name and prefix in self.prefix_set:
            return prefix, tag_name

        raise ValueError("tag: {} is not a avoid tag".format(tag))

    def is_prefix_legal(self, previous, current):
        node = (previous, current)

        return node in self.legal_set

    def decode_to_offset(self, sequence):
        offset_list = []

        last_tag_prefix = None
        next_tag_prefix = None
        tag_length = 0
        tag_name_cache = None

        for index, item in enumerate(sequence):
            # inline function <<< self.parse_tag(item)
            tag = item
            if tag == self.oscar:
                prefix = self.oscar
                tag_name = None
            else:
                # set maxsplit to 1, so raw_tag_name can contains '-' char legally
                prefix, tag_name = tag.split("-", maxsplit=1)

                if prefix and tag_name and prefix in self.prefix_set:
                    pass
                else:
                    raise ValueError("tag: {} is not a avoid tag".format(tag))
            # inline function >>> self.parse_tag(item)

            if last_tag_prefix is None:
                if prefix == self.oscar:
                    # ignore it
                    continue
                elif prefix == "B":
                    if index<=len(sequence)-1:
                        if len(sequence[index+1])>1:
                            next_tag_prefix,_ = sequence[index+1].split("-", maxsplit=1)
                        else:
                            next_tag_prefix = sequence[index+1]
                    else:
                        next_tag_prefix = None
                    if (next_tag_prefix is None
                        or next_tag_prefix=='O'
                        or next_tag_prefix=='B'):
                        offset_list.append((index, index + 1, tag_name))
                    else:
                        tag_name_cache = tag_name
                        last_tag_prefix = prefix
                        tag_length += 1
                else:
                    if not self.ignore_error:
                        raise TagSetDecodeError(
                            "sequence: {} is not a valid tag sequence".format(
                                sequence[: index + 1]
                            )
                        )
                    else:
                        continue
            else:
                if not self.is_prefix_legal(last_tag_prefix, prefix):
                    raise TagSetDecodeError(
                        "sequence: {} is not a valid tag sequence".format(
                            sequence[: index + 1]
                        )
                    )

                if prefix == "I":
                    if index<=len(sequence)-1:
                        if len(sequence[index+1])>1:
                            next_tag_prefix,_ = sequence[index+1].split("-", maxsplit=1)
                        else:
                            next_tag_prefix = sequence[index+1]
                    else:
                        next_tag_prefix = None

                    if (next_tag_prefix is None
                        or next_tag_prefix=='O'
                        or next_tag_prefix=='B'):
                        if tag_name_cache == tag_name:
                            offset_list.append(
                                (index - tag_length, index + 1, tag_name_cache)
                            )

                            # clean up
                            last_tag_prefix = None
                            tag_length = 0
                            tag_name_cache = None
                            next_tag_prefix = None
                        else:
                            raise TagSetDecodeError(
                                "sequence: {} is not a valid tag sequence".format(
                                    sequence[: index + 1]
                                )
                            )
                    elif prefix == "I":
                        if tag_name_cache == tag_name:
                            last_tag_prefix = prefix
                            tag_length += 1
                        else:
                            raise TagSetDecodeError(
                                "sequence: {} is not a valid tag sequence".format(
                                    sequence[: index + 1]
                                )
                            )

        return offset_list

    def to_offset(self, sequence, text, **kwargs):
        span_set = SpanSet(
            [
                Span(offset[0], offset[1], offset[2])
                for offset in self.decode_to_offset(sequence)
            ]
        )

        label = kwargs.pop("label", None)
        id_ = kwargs.pop("id", None)
        extra_attr = kwargs

        seq = Document(text, span_set, id_, label, extra_attr)
        seq.span_set.bind(seq)

        return seq


from collections import namedtuple
from pathlib import Path
import copy

from deliverable_model.processor_base import ProcessorBase
from deliverable_model.request import Request
from deliverable_model.response import Response


PredictResult = namedtuple("PredictResult", ["sequence", "is_failed", "exec_msg"])


class BIOEncodeProcessor(ProcessorBase):
    def __init__(self, decoder=None, **kwargs):
        super().__init__(**kwargs)

        self.decoder = decoder
        self.request_query = None

    @classmethod
    def load(cls, parameter: dict, asset_dir) -> "ProcessorBase":

        decoder = BIOSequenceEncoderDecoder()

        self = cls(decoder, **parameter)

        return self

    def preprocess(self, request: Request) -> Request:
        # record request for postprocess usage
        self.request_query = copy.deepcopy(request[self.pre_input_key])

        # don't change the request
        return request

    def postprocess(self, response: Response) -> Response:
        from tokenizer_tools.tagset.exceptions import TagSetDecodeError
        from tokenizer_tools.tagset.offset.sequence import Sequence

        tags_list = response[self.post_input_key]
        raw_text_list = self.request_query

        infer_result = []

        for raw_text, tags in zip(raw_text_list, tags_list):
            # decode Unicode
            tags_seq = [i.decode() if isinstance(i, bytes) else i for i in tags]

            # BILUO to offset
            is_failed = False
            exec_msg = None
            try:
                seq = self.decoder.to_offset(tags_seq, raw_text)
            except TagSetDecodeError as e:
                exec_msg = str(e)

                # invalid tag sequence will raise exception
                # so return a empty result to avoid batch fail
                seq = Sequence(raw_text)
                is_failed = True

            infer_result.append(PredictResult(seq, is_failed, exec_msg))

        response[self.post_output_key] = infer_result

        return response

    def get_dependency(self) -> list:
        return ["tokenizer_tools"]


if __name__ == "__main__":
    decoder = BIOSequenceEncoderDecoder()
    result = decoder.decode_to_offset(["U-XX"])
    print(result)
    assert result == [(0, 1, "XX")]

    result = decoder.decode_to_offset(["U-XX", "U-YY"])
    print(result)
    assert result == [(0, 1, "XX"), (1, 2, "YY")]

    result = decoder.decode_to_offset(["B-XX", "I-XX", "L-XX"])
    print(result)
    assert result == [(0, 3, "XX")]
