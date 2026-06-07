"""
评估脚本：测试语义搜索和文本搜索对 "黑白猫" 的搜索效果
"""
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🔍 搜索评估: '黑白猫'")
print("=" * 70)

# 1. 分词效果
print("\n📝 Step 1: 分词效果")
try:
    import jieba
    query = "黑白猫"
    keywords = list(jieba.cut(query))
    IMPORTANT_SINGLE_CHARS = {'猫', '狗', '人', '车', '鸟', '树', '门', '窗'}
    filtered = [k for k in keywords if len(k) > 1 or k in IMPORTANT_SINGLE_CHARS]
    print(f"  查询: '{query}'")
    print(f"  分词结果: {filtered}")
except Exception as e:
    print(f"  ❌ 分词失败: {e}")

# 2. 文本搜索（VectorStore）
print("\n📊 Step 2: 文本搜索 (VectorStore)")
print("-" * 70)
try:
    from src.analysis.vector_db import VectorStore
    vs = VectorStore()
    
    text_results = vs.search("黑白猫", n_results=10, sort_by="similarity")
    print(f"  找到 {len(text_results)} 条结果:\n")
    
    for i, r in enumerate(text_results[:10]):
        desc = r.get('description', '')
        sim = r.get('similarity', 0)
        kw_hits = r.get('_keyword_hits', 0)
        
        # 高亮匹配的关键词
        highlight_desc = desc
        for kw in ['黑白', '猫', '黑白色', '黑白相间']:
            if kw in highlight_desc:
                highlight_desc = highlight_desc.replace(kw, f"【{kw}】")
        
        # 截断显示
        if len(highlight_desc) > 80:
            highlight_desc = highlight_desc[:80] + "..."
        
        print(f"  #{i+1} 相似度: {sim:.3f} | 关键词命中: {kw_hits}")
        print(f"      {highlight_desc}\n")
        
except Exception as e:
    print(f"  ❌ 文本搜索失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 语义搜索（VisualVectorStore）
print("\n📊 Step 3: 语义搜索 (VisualVectorStore)")
print("-" * 70)
try:
    from src.analysis.vector_db import VisualVectorStore
    from src.analysis.feature_extractor import get_embedder
    from src.common.config import get_config
    
    config = get_config()
    
    # 初始化语义存储和嵌入器
    vvs = VisualVectorStore()
    embedder = get_embedder(
        provider="api",
        api_key="ollama",
        api_url="http://127.0.0.1:11434/v1",
        model_name="embeddinggemma:latest"
    )
    
    print(f"  Embedder: {embedder.__class__.__name__}")
    print(f"  数据库记录数: {vvs.count()}")
    
    if vvs.count() > 0:
        semantic_results = vvs.search_by_text(
            embedder=embedder,
            query_text="黑白猫",
            n_results=10
        )
        
        print(f"  找到 {len(semantic_results)} 条结果:\n")
        
        for i, r in enumerate(semantic_results[:10]):
            desc = r.get('description', '') or r.get('metadata', {}).get('description', '')
            sim = r.get('similarity', 0)
            
            # 高亮匹配的关键词
            highlight_desc = desc
            for kw in ['黑白', '猫', '黑白色', '黑白相间']:
                if kw in highlight_desc:
                    highlight_desc = highlight_desc.replace(kw, f"【{kw}】")
            
            if len(highlight_desc) > 80:
                highlight_desc = highlight_desc[:80] + "..."
            
            print(f"  #{i+1} 相似度: {sim:.3f}")
            print(f"      {highlight_desc}\n")
    else:
        print("  ⚠️ 语义数据库为空")
        
except Exception as e:
    print(f"  ❌ 语义搜索失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 使用 SearchEngine 混合搜索
print("\n📊 Step 4: 混合搜索 (SearchEngine)")
print("-" * 70)
try:
    from src.analysis.search import SearchEngine
    from src.analysis.vector_db import VectorStore, VisualVectorStore
    from src.analysis.feature_extractor import get_embedder
    from src.common.config import get_config
    
    config = get_config()
    vs = VectorStore()
    vvs = VisualVectorStore()
    embedder = get_embedder(
        provider="api",
        api_key="ollama",
        api_url="http://127.0.0.1:11434/v1",
        model_name="embeddinggemma:latest"
    )
    
    engine = SearchEngine(
        vector_store=vs,
        visual_store=vvs,
        visual_embedder=embedder
    )
    
    # 混合搜索 - 相关性优先
    print("  【相关性优先】")
    hybrid_results = engine.search("黑白猫", n_results=5, sort_mode="relevance", search_mode="hybrid")
    
    for i, r in enumerate(hybrid_results[:5]):
        desc = r.get('description', '')[:60]
        sim = r.get('similarity', 0)
        source = r.get('_source', 'unknown')
        score = r.get('_hybrid_score', 0)
        print(f"  #{i+1} sim={sim:.3f} score={score:.3f} [{source}] | {desc}...")
    
    print("\n  【最新优先】")
    time_results = engine.search("黑白猫", n_results=5, sort_mode="time", search_mode="hybrid")
    
    from datetime import datetime
    for i, r in enumerate(time_results[:5]):
        desc = r.get('description', '')[:50]
        sim = r.get('similarity', 0)
        ts = r.get('metadata', {}).get('timestamp', 0)
        time_str = datetime.fromtimestamp(ts).strftime('%m-%d %H:%M') if ts > 86400 else "N/A"
        print(f"  #{i+1} sim={sim:.3f} time={time_str} | {desc}...")
        
except Exception as e:
    print(f"  ❌ 混合搜索失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 评估总结
print("\n" + "=" * 70)
print("📋 评估总结")
print("=" * 70)
print("""
检查要点:
1. 分词是否正确? (应该是 ['黑白', '猫'])
2. 文本搜索是否找到包含"黑白"或"猫"的记录?
3. 语义搜索相似度是否合理? (>0.5 为好)
4. 混合搜索是否综合了两者的优点?
5. 关键词命中数是否被正确计算?
""")
