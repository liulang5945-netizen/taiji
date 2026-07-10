"""Replace Bing search function in discovery.py"""
import re

path = r"E:\taiji\taiji\tools\search\discovery.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_bing = re.search(r"def _search_bing\(.*?return \[\]\n", content, re.DOTALL).group()

new_bing = '''def _search_bing(query: str, max_results: int = 8) -> List[SearchResult]:
    """Bing 搜索结果页爬取（2024+ 新 HTML 结构兼容）"""
    try:
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        html = http_get(url, timeout=12)
        results = []
        # Bing 新结构：结果在 <li class="b_algo"> 或 #b_results > li
        blocks = re.findall(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>', html, re.DOTALL)
        if not blocks:
            # 新版 Bing 可能用不同结构，直接找 h2 > a
            pattern = r'<h2[^>]*>\\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?(?:<p[^>]*>(.*?)</p>)?'
            for href, title, snippet in re.findall(pattern, html, re.DOTALL)[:max_results]:
                results.append(SearchResult(
                    title=_strip_tags(title), url=href,
                    snippet=_strip_tags(snippet) if snippet else "",
                    source="Bing",
                ))
            return results
        for block in blocks[:max_results]:
            title_m = re.search(r'<h2[^>]*>\\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', block, re.DOTALL)
            desc_m = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
            if title_m:
                results.append(SearchResult(
                    title=_strip_tags(title_m.group(2)),
                    url=title_m.group(1),
                    snippet=_strip_tags(desc_m.group(1)) if desc_m else "",
                    source="Bing",
                ))
        return results
    except Exception as e:
        logger.debug(f"Bing 搜索失败: {e}")
        return []
'''

content = content.replace(old_bing, new_bing)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Bing function replaced")
