from typing import List

from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        return False


'''

 转换为类似BI标签，如果with_label为True, 标签类似于B_background, I_background
 否则仅为B,I标签

'''


def convert_to_bio_schema(labels, with_label=True):
    bio_labels: List[str] = []
    for index, label in enumerate(labels):
        bio_label = "B"
        if index != 0 and label == labels[index - 1]:
            bio_label = "I"
        if with_label:
            bio_label = bio_label + "_" + label
        bio_labels.append(bio_label)
    assert len(bio_labels) == len(labels)
    return bio_labels


# 将结果转换为Span, 输入为标签列表
# 输出为[('background',(1,2)), ('other', (3,4))]
def convert_to_span(labels: List[str]):
    bio_labels = convert_to_bio_schema(labels, with_label=True)
    spans = bio_tags_to_spans(bio_labels)
    return bio_labels, spans

def pico_tags_to_span(tag_sequence: List[str], classes_to_ignore: List[str] = None)-> List[TypedStringSpan]:
    bio_tags = convert_to_bio_schema(tag_sequence, with_label=True)
    return bio_tags_to_spans(bio_tags, classes_to_ignore)

if __name__ == "__main__":
    labels = ['B', 'S', 'OT', 'P', 'OT', 'O', 'O', 'O', 'O', 'O']
    b_label,spans=convert_to_span(labels)
    print(b_label)
    print(spans)
    # b_labels = convert_to_bio_schema(labels)
    # print(labels)
    # print(b_labels)
    # clabels = count_continous_element(labels)
    # print(clabels)
