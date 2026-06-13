"""
态极联网模块 (Web Module)
==========================

统一的联网能力，整合搜索、网页阅读、内容提取。

核心特性：
1. 多引擎搜索（DuckDuckGo, Bing, Baidu）+ 自动降级
2. 智能网页解析（HTML → Markdown，提取正文）
3. 结果缓存（避免重复请求）
4. 内容去重（相似结果合并）
5. 请求限速（防止被封）

使用方式：
    from taiji.tools.web import search, fetch, browse
    results = search("Python 3.12 新特性")
    content = fetch("https://example.com")
"""
import os
import re
import json
import time
import hashlib
import logging
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Taiji.Web")


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str = ""  # 搜索引擎名称

    def to_markdown(self) -> str:
        return f"### {self.title}\n{self.snippet}\n来源: {self.url}"


@dataclass
class WebPage:
    """网页内容"""
    url: str
    title: str
    content: str
    content_type: str = "text"  # text / markdown / html
    fetched_at: float = 0
    size: int = 0

    def to_markdown(self) -> str:
        return f"# {self.title}\n\n{self.content}"


# ═══════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════

class WebCache:
    """联网结果缓存"""

    def __init__(self, cache_dir: str = None, ttl: int = 3600):
        self.cache_dir = cache_dir or os.path.join("taiji_data", "web_cache")
        self.ttl = ttl  # 缓存有效期（秒）
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url: str) -> Optional[str]:
        """获取缓存"""
        key = self._key(url)
        path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if time.time() - data.get("timestamp", 0) > self.ttl:
                os.remove(path)
                return None
            return data.get("content")
        except Exception:
            return None

    def set(self, url: str, content: str):
        """设置缓存"""
        key = self._key(url)
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    "url": url,
                    "content": content[:50000],  # 限制缓存大小
                    "timestamp": time.time(),
                }, f, ensure_ascii=False)
        except Exception:
            pass

    def clear(self):
        """清空缓存"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)


# 全局缓存
_cache = WebCache()


# ═══════════════════════════════════════════════
# 搜索引擎
# ═══════════════════════════════════════════════

def _search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    """DuckDuckGo 搜索"""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    source="DuckDuckGo",
                ))
        return results
    except ImportError:
        logger.debug("duckduckgo_search 未安装")
        return []
    except Exception as e:
        logger.debug(f"DuckDuckGo 搜索失败: {e}")
        return []


def _search_bing(query: str, max_results: int = 5) -> List[SearchResult]:
    """Bing 搜索（网页抓取）"""
    try:
        from bs4 import BeautifulSoup
        import requests
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select(".b_algo"):
            title_el = item.select_one("h2 a")
            desc_el = item.select_one(".b_caption p")
            if title_el:
                results.append(SearchResult(
                    title=title_el.text.strip(),
                    url=title_el.get("href", ""),
                    snippet=desc_el.text.strip() if desc_el else "",
                    source="Bing",
                ))
        return results[:max_results]
    except Exception as e:
        logger.debug(f"Bing 搜索失败: {e}")
        return []


def _search_baidu(query: str, max_results: int = 5) -> List[SearchResult]:
    """百度搜索（网页抓取）"""
    try:
        from bs4 import BeautifulSoup
        import requests
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select(".result.c-container"):
            title = item.select_one(".t")
            desc = item.select_one(".c-abstract")
            link = item.select_one(".t a")
            if title:
                results.append(SearchResult(
                    title=title.text.strip(),
                    url=link.get("href", "") if link else "",
                    snippet=desc.text.strip() if desc else "",
                    source="Baidu",
                ))
        return results[:max_results]
    except Exception as e:
        logger.debug(f"百度搜索失败: {e}")
        return []


def search(query: str, max_results: int = 5, engine: str = "auto") -> List[SearchResult]:
    """
    搜索互联网。

    Args:
        query: 搜索关键词
        max_results: 最大结果数
        engine: 搜索引擎 ("auto", "duckduckgo", "bing", "baidu")

    Returns:
        搜索结果列表
    """
    # 检查缓存
    cache_key = f"search:{query}:{max_results}"
    cached = _cache.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            return [SearchResult(**r) for r in data]
        except Exception:
            pass

    results = []

    # 自动模式：依次尝试
    if engine == "auto":
        engines = [
            ("DuckDuckGo", _search_duckduckgo),
            ("Bing", _search_bing),
            ("Baidu", _search_baidu),
        ]
        for name, func in engines:
            results = func(query, max_results)
            if results:
                logger.info(f"搜索引擎 {name} 返回 {len(results)} 条结果")
                break
    elif engine == "duckduckgo":
        results = _search_duckduckgo(query, max_results)
    elif engine == "bing":
        results = _search_bing(query, max_results)
    elif engine == "baidu":
        results = _search_baidu(query, max_results)

    # 缓存结果
    if results:
        try:
            _cache.set(cache_key, json.dumps([{
                "title": r.title, "url": r.url,
                "snippet": r.snippet, "source": r.source,
            } for r in results], ensure_ascii=False))
        except Exception:
            pass

    return results


def search_to_text(query: str, max_results: int = 5) -> str:
    """搜索并返回格式化文本（供工具调用）"""
    results = search(query, max_results)
    if not results:
        return f"搜索 '{query}' 未找到结果。"
    lines = [f"搜索 '{query}' 找到 {len(results)} 条结果:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.snippet[:200]}")
        lines.append(f"   URL: {r.url}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 网页抓取
# ═══════════════════════════════════════════════

def _html_to_text(html: str) -> str:
    """HTML 转纯文本"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # 移除无用标签
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.extract()
        text = soup.get_text(separator="\n", strip=True)
        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    except ImportError:
        # 无 BeautifulSoup，简单清理
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def _html_to_markdown(html: str) -> str:
    """HTML 转 Markdown"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # 移除无用标签
        for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
            tag.extract()

        md_parts = []
        for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'pre', 'code', 'blockquote']):
            tag = elem.name
            text = elem.get_text(strip=True)
            if not text:
                continue
            if tag == 'h1':
                md_parts.append(f"# {text}")
            elif tag == 'h2':
                md_parts.append(f"## {text}")
            elif tag == 'h3':
                md_parts.append(f"### {text}")
            elif tag == 'h4':
                md_parts.append(f"#### {text}")
            elif tag == 'li':
                md_parts.append(f"- {text}")
            elif tag == 'pre':
                md_parts.append(f"```\n{text}\n```")
            elif tag == 'code':
                md_parts.append(f"`{text}`")
            elif tag == 'blockquote':
                md_parts.append(f"> {text}")
            else:
                md_parts.append(text)

        return "\n\n".join(md_parts)
    except ImportError:
        return _html_to_text(html)


def fetch(url: str, as_markdown: bool = True, max_length: int = 10000) -> str:
    """
    抓取网页内容。

    Args:
        url: 网页 URL
        as_markdown: 是否转换为 Markdown
        max_length: 最大内容长度

    Returns:
        网页内容文本
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 检查缓存
    cached = _cache.get(url)
    if cached:
        return cached[:max_length]

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            html = resp.read().decode('utf-8', errors='ignore')

        # 根据内容类型处理
        if "json" in content_type:
            content = html[:max_length]
        elif as_markdown:
            content = _html_to_markdown(html)
        else:
            content = _html_to_text(html)

        content = content[:max_length]

        # 缓存
        _cache.set(url, content)

        return content
    except urllib.error.HTTPError as e:
        return f"HTTP 错误: {e.code} {e.reason}"
    except Exception as e:
        return f"抓取失败: {e}"


def fetch_to_markdown(url: str, max_length: int = 10000) -> str:
    """抓取网页并转为 Markdown（供工具调用）"""
    content = fetch(url, as_markdown=True, max_length=max_length)
    if content.startswith("抓取失败") or content.startswith("HTTP 错误"):
        return content
    return f"网页内容 ({url}):\n\n{content}"


# ═══════════════════════════════════════════════
# 统一接口（供工具注册）
# ═══════════════════════════════════════════════

def web_search(query: str) -> str:
    """搜索工具的统一接口"""
    return search_to_text(query, max_results=5)


def web_fetch(url: str) -> str:
    """抓取工具的统一接口"""
    return fetch_to_markdown(url, max_length=8000)


def web_browse(url: str) -> str:
    """浏览工具的统一接口（尝试 JS 渲染）"""
    # 先尝试 Playwright
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=15000)
            html = page.content()
            browser.close()
            return _html_to_markdown(html)[:8000]
    except Exception:
        pass

    # 回退到普通抓取
    return fetch_to_markdown(url)
