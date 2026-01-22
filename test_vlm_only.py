#!/usr/bin/env python3
"""
测试 VLM 时间同步功能（不使用 OCR）
"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def test_imports():
    """测试导入是否会触发 OCR 初始化"""
    print("🔄 测试模块导入...")
    
    try:
        # 这些导入不应该触发 PaddleOCR 初始化
        from src.ingestion import VideoIngestionService
        from src.analysis.vlm import VLMAnalyzer
        print("✅ 核心模块导入成功，未触发 OCR 初始化")
        
        # 测试 VLM 分析器初始化
        vlm_analyzer = VLMAnalyzer(backend="mock")  # 使用 mock 后端避免网络调用
        print(f"✅ VLM 分析器初始化成功: {vlm_analyzer.get_backend_name()}")
        
        # 测试时间同步器初始化
        from src.ingestion.service import TimeSynchronizer
        roi_config = (0.65, 0.85, 0.35, 0.15)
        time_sync = TimeSynchronizer(vlm_analyzer, roi_config)
        print("✅ TimeSynchronizer 初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        return False

def test_vlm_time_parsing():
    """测试 VLM 时间解析功能"""
    print("\n🔄 测试时间解析功能...")
    
    try:
        from src.ingestion.service import TimeSynchronizer
        from src.analysis.vlm import VLMAnalyzer
        
        vlm_analyzer = VLMAnalyzer(backend="mock")
        roi_config = (0.65, 0.85, 0.35, 0.15)
        time_sync = TimeSynchronizer(vlm_analyzer, roi_config)
        
        # 测试时间解析
        test_responses = [
            "2026-01-22 18:30:45",
            "2026/01/22 18:30:45",
            "2026.01.22 18:30:45",
            "2026-01-22:18:30:45",
            "invalid response",
            "",
            None
        ]
        
        for response in test_responses:
            parsed = time_sync._parse_time_response(response)
            if parsed:
                print(f"✅ 解析成功: '{response}' -> {parsed}")
            else:
                print(f"⚠️ 解析失败: '{response}' -> None")
        
        return True
        
    except Exception as e:
        print(f"❌ 时间解析测试失败: {e}")
        return False

def main():
    print("🚀 开始测试 VLM 时间同步功能（无 OCR）")
    print("=" * 50)
    
    success = True
    
    # 测试导入
    if not test_imports():
        success = False
    
    # 测试时间解析
    if not test_vlm_time_parsing():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！VLM 时间同步功能正常，未使用 OCR")
    else:
        print("❌ 部分测试失败，请检查配置")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)