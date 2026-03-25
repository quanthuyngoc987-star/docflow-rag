"""
向量化模型 —— 将文本映射到高维向量空间

学习要点：
- Embedding 将文本转换为固定维度的向量，使语义相似的文本在向量空间中距离更近
- all-MiniLM-L6-v2 是英文优化模型（384维），中文可换用 text2vec-base-chinese
- 首次运行时模型会自动下载（约 80MB），需要网络连接
"""

import logging
import numpy as np
import hashlib
from functools import lru_cache

# 模型选择说明：
# - all-MiniLM-L6-v2: 英文优化，384维，轻量快速（默认）
# - shibing624/text2vec-base-chinese: 中文优化
# - BAAI/bge-small-zh-v1.5: 中文优化，性能更好
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


class LocalHashEmbedding:
    """
    离线回退向量器：
    当无法下载/加载 SentenceTransformer 模型时，使用稳定哈希生成固定维度向量，
    保证文档上传、索引、检索流程可用（效果弱于语义模型）。
    """

    def __init__(self, dim=384):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def _tokenize(self, text):
        text = (text or "").strip()
        if not text:
            return []

        # 中文文本常无空格：若空格切分退化为 1 个 token，则使用字符级切分。
        tokens = [t for t in text.split() if t]
        if len(tokens) <= 1 and len(text) > 1:
            tokens = list(text)
        return tokens

    def _encode_one(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            weight = 1.0 + (digest[5] / 255.0) * 0.1
            vec[idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self._encode_one(text) for text in texts], axis=0).astype('float32')


@lru_cache(maxsize=1)
def get_embed_model():
    """
    获取向量化模型（单例 + 缓存）

    首次调用时加载模型，后续调用直接返回缓存的实例。
    """
    from sentence_transformers import SentenceTransformer
    logging.info(f"加载向量化模型: {EMBED_MODEL_NAME}")

    # 先尝试本地缓存（离线优先），若不可用再尝试联网加载。
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        logging.info(f"向量化模型加载完成（本地缓存），输出维度: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as local_err:
        logging.warning(f"本地缓存模型不可用，将尝试联网加载: {local_err}")

    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
        logging.info(f"向量化模型加载完成（联网），输出维度: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as remote_err:
        logging.warning(f"联网加载向量模型失败，回退到本地哈希向量器: {remote_err}")
        fallback = LocalHashEmbedding(dim=384)
        logging.info(f"回退向量器已启用，输出维度: {fallback.get_sentence_embedding_dimension()}")
        return fallback


def encode_texts(texts, show_progress=False):
    """
    将文本列表编码为向量

    Args:
        texts: 文本列表
        show_progress: 是否显示进度条

    Returns:
        numpy 数组，形状为 (n_texts, embedding_dim)
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return np.array(embeddings).astype('float32')


def encode_query(query):
    """
    将单个查询文本编码为向量

    Returns:
        numpy 数组，形状为 (1, embedding_dim)
    """
    model = get_embed_model()
    embedding = model.encode([query])
    return np.array(embedding).astype('float32')
