"""轻量英文关键词召回：按词频、词语稀有程度和正文长度评分，不调用模型。"""

import math
import re
from collections import Counter


STOP_WORDS = frozenset("a an the is are was were be been being do does did how what why when where which who to of in on at for from with by and or it its they their them this that these those should can could would need needs".split())


def tokenize_english(text: str) -> list[str]:
    """统一英文大小写并拆成词语、去掉常见虚词；text为输入文本；返回词语列表。

    不做中文分词、不把中文自动翻成英文、不合并复数和同义词；否定词not/no保留。
    """
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in STOP_WORDS]


class EnglishBM25:
    """只在内存中构建正文词频表，适合当前几千条英文资料的离线试验。"""

    def __init__(self, ids: list[str], texts: list[str], *, k1: float = 1.5, b: float = 0.75):
        """构建关键词索引。

        参数：ids与texts为等长的唯一记录编号和原始正文；k1控制重复词加分饱和，
            b控制正文长度的影响，默认参数固定，不根据测试答案调整。
        返回值：无；只保存内存统计，不写入向量库，不把元数据或预期答案拼入正文。
        """
        if not ids or len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError("关键词索引的编号、正文必须非空、等长且编号唯一。")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25参数无效。")
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize_english(text)) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(ids)
        if self.average_length == 0:
            raise ValueError("全部正文都没有可用英文词语。")
        self.document_frequency = Counter(word for counts in self.term_counts for word in counts)

    def search(self, question: str, *, top_k: int = 5) -> list[dict]:
        """按英文问题的词语匹配召回记录。

        参数：question为英文检索问题，top_k为正整数候选数量。
        返回值：含chunk_id和score的列表，分数越高越靠前；无词命中返回空，
            不用零分记录凑数。分数不是相关概率，不与向量距离直接相加。
        """
        if type(top_k) is not int or top_k < 1:
            raise ValueError("候选数量必须为正整数。")
        terms = sorted(set(tokenize_english(question)))
        ranked = []
        for index, counts in enumerate(self.term_counts):
            score = 0.0
            for word in terms:
                frequency = counts.get(word, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[word]
                rarity = math.log1p((len(self.ids) - document_frequency + 0.5) / (document_frequency + 0.5))
                length_adjustment = 1 - self.b + self.b * self.lengths[index] / self.average_length
                score += rarity * frequency * (self.k1 + 1) / (frequency + self.k1 * length_adjustment)
            if score > 0:
                ranked.append({"chunk_id": self.ids[index], "score": score})
        return sorted(ranked, key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]
