import pathlib, re

f = pathlib.Path(r"E:\taiji\taiji\tools\mini_search.py")
content = f.read_text("utf-8")

# Replace _init_db to use :memory: + JSON persistence
old_init_db = content[content.index("    def _init_db"):content.index("    # ==", content.index("    def _init_db"))]
new_init_db = '''    def _init_db(self):
        """内存数据库 + JSON 持久化"""
        self._memory_pages = []  # page dict list
        import atexit
        self._load_from_disk()
        atexit.register(self._save_to_disk)

    def _json_path(self):
        return os.path.join(self.data_dir, "index_snapshot.json")

    def _save_to_disk(self):
        """持久化到 JSON"""
        try:
            import json
            with open(self._json_path(), "w", encoding="utf-8") as fh:
                json.dump(self._memory_pages, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_from_disk(self):
        """从 JSON 恢复"""
        try:
            import json
            jp = self._json_path()
            if os.path.exists(jp):
                with open(jp, "r", encoding="utf-8") as fh:
                    self._memory_pages = json.load(fh)
        except Exception:
            self._memory_pages = []

    def _save_page(self, url, title, text, links):
        """存储页面到内存列表"""
        with self._lock:
            entry = {
                "url": url, "title": title[:200], "text": text[:10000],
                "links": links, "crawled_at": time.time(),
            }
            self._memory_pages = [p for p in self._memory_pages if p["url"] != url]
            self._memory_pages.append(entry)

    def _rebuild_fts(self):
        """内存索引重建（搜索时实时计算 BM25）"""
        self._save_to_disk()
'''

content = content.replace(old_init_db, new_init_db)

# Replace search method to use in-memory BM25
old_search = content[content.index("    def search(self, query"):content.index("    def search_to_text")]
new_search = '''    def search(self, query, top_k=10):
        """BM25 搜索内存索引"""
        query_terms = self._tokenize(query)
        if not query_terms or not self._memory_pages:
            return []

        # 计算 BM25
        N = len(self._memory_pages)
        avgdl = sum(len(p["text"]) for p in self._memory_pages) / max(N, 1)
        k1, b = 1.5, 0.75
        scored = []

        for p in self._memory_pages:
            doc_text = p["text"]
            doc_len = len(doc_text)
            score = 0.0
            words = doc_text.lower().split()
            for term in query_terms:
                tf = words.count(term)
                df = sum(1 for pp in self._memory_pages if term in pp["text"].lower())
                idf = max(0, __import__("math").log((N - df + 0.5) / (df + 0.5) + 1))
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1))
                score += idf * numerator / max(denominator, 0.1)
            if score > 0:
                snippet = self._make_snippet(doc_text, query_terms)
                scored.append((score, p["url"], p["title"], snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, url, title, snip in scored[:top_k]:
            results.append(SearchHit(url=url, title=title, snippet=snip[:200], score=round(score, 2)))
        return results

    def _tokenize(self, text):
        """简单分词"""
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
        return [t for t in text.split() if len(t) >= 2]

    def _make_snippet(self, text, terms, window=80):
        """生成结果摘要"""
        low = text.lower()
        for t in terms:
            idx = low.find(t)
            if idx >= 0:
                start = max(0, idx - window // 2)
                end = min(len(text), idx + window // 2)
                return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        return text[:window * 2]

    def search_to_text(self, query, top_k=10):
'''

idx = content.index('    def search_to_text(self, query')
old_end = content.index('        return "\n".join(lines)', idx)
content = content.replace(old_search, new_search)

# Update stats to use _memory_pages
old_stats = content[content.index("    def stats(self)"):content.index("    def clear_index")]
new_stats = '''    def stats(self):
        """索引统计"""
        return {"indexed_pages": len(self._memory_pages), "data_dir": self.data_dir}

    def clear_index(self):
        """清空索引"""
        with self._lock:
            self._memory_pages = []
            self._save_to_disk()
'''
content = content.replace(old_stats, new_stats)

# Remove old clear_index that follows
old_clear = content[content.index("    def clear_index(self):\n", content.index("    def clear_index(self):\n")+1):]
if "def clear_index" in old_clear:
    end = content.rindex("    def clear_index")
    content = content[:end]

# Fix _save_page references - remove old SQL-based one
# Also remove old _rebuild_fts that uses SQL
old_rebuild = content[content.index("    def _rebuild_fts", content.index("# 2+3+4")):content.index("# ===", content.index("    def _rebuild_fts", content.index("# 2+3+4")))]
if "INSERT INTO fts_idx" in old_rebuild:
    content = content.replace(old_rebuild, "")

f.write_text(content, "utf-8")
print("OK")
