"""Register smart_crawl tool in tool_registry.py"""

path = r"E:\taiji\taiji\agent_ext\tool_registry.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''            func=tool_browse_web,
            source="local", category="\u7f51\u7edc",
        ))

    except Exception:
        pass'''

new = '''            func=tool_browse_web,
            source="local", category="\u7f51\u7edc",
        ))

        local_tools.append(ToolDef(
            name="smart_crawl",
            description="\u667a\u80fd\u722c\u53d6\uff1a\u5148\u641c\u7d22\u5173\u952e\u8bcd\uff0c\u518d\u5bf9\u641c\u7d22\u7ed3\u679c\u505a\u4e3b\u9898\u805a\u7126\u722c\u53d6\u3002\u81ea\u52a8\u8bc4\u5206\u94fe\u63a5\u3001\u8bc4\u4f30\u5185\u5bb9\u8d28\u91cf\u3001\u81ea\u9002\u5e94\u6df1\u5ea6\uff0c\u53ea\u722c\u6709\u4ef7\u503c\u7684\u9875\u9762\u3002\u9002\u5408\u6df1\u5165\u7814\u7a76\u67d0\u4e2a\u4e3b\u9898\u3002",
            parameters={"type": "object", "properties": {
                "input": {"type": "string", "description": "\u4e3b\u9898\u5173\u952e\u8bcd\u6216\u641c\u7d22\u67e5\u8be2"}
            }, "required": ["input"]},
            func=tool_smart_crawl,
            source="local", category="\u7f51\u7edc",
        ))

    except Exception:
        pass'''

if old in content:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("smart_crawl registered")
else:
    print("ERROR: old block not found")
