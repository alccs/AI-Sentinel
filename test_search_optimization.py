"""
测试脚本: 验证中文搜索优化效果
"""
import jieba
import sys

print("=" * 60)
print("🧪 中文搜索优化测试")
print("=" * 60)

# 1. 测试分词效果
test_queries = [
    "黑白猫",
    "黑白相间的猫",
    "红色汽车",
    "穿蓝衣服的人",
    "白色狗在院子里"
]

print("\n📝 分词测试:")
for q in test_queries:
    keywords = list(jieba.cut(q))
    filtered = [k for k in keywords if len(k.strip()) > 1]
    print(f"  '{q}' -> {filtered}")

# 2. 测试搜索模块是否正常加载
print("\n🔍 搜索模块测试:")
try:
    from src.analysis.search import SearchEngine, JIEBA_AVAILABLE
    print(f"  ✅ SearchEngine 加载成功")
    print(f"  ✅ JIEBA 可用: {JIEBA_AVAILABLE}")
except Exception as e:
    print(f"  ❌ 加载失败: {e}")
    sys.exit(1)

# 3. 测试向量数据库
print("\n📊 数据库连接测试:")
try:
    from src.analysis.vector_db import VectorStore
    vs = VectorStore()
    count = vs.count()
    print(f"  ✅ VectorStore 连接成功")
    print(f"  📊 数据库记录数: {count}")
    
    if count > 0:
        # 4. 测试搜索 "猫"
        print("\n🔎 搜索测试 '猫':")
        results = vs.search("猫", n_results=5)
        print(f"  找到 {len(results)} 条结果:")
        for r in results[:3]:
            desc = r.get('description', '')[:50]
            sim = r.get('similarity', 0)
            print(f"    - 相似度:{sim:.2f} | {desc}...")
        
        # 5. 测试搜索 "黑白"
        print("\n🔎 搜索测试 '黑白':")
        results = vs.search("黑白", n_results=5)
        print(f"  找到 {len(results)} 条结果:")
        for r in results[:3]:
            desc = r.get('description', '')[:50]
            sim = r.get('similarity', 0)
            print(f"    - 相似度:{sim:.2f} | {desc}...")
            
        # 6. 测试搜索 "黑白猫"
        print("\n🔎 搜索测试 '黑白猫':")
        results = vs.search("黑白猫", n_results=5)
        print(f"  找到 {len(results)} 条结果:")
        for r in results[:3]:
            desc = r.get('description', '')[:60]
            sim = r.get('similarity', 0)
            hits = r.get('_keyword_hits', 0)
            print(f"    - 相似度:{sim:.2f} 关键词命中:{hits} | {desc}...")
    else:
        print("  ⚠️ 数据库为空，跳过搜索测试")
        
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 测试完成")
print("=" * 60)
