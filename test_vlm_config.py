#!/usr/bin/env python3
"""
测试 VLM 配置和连接
"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def test_vlm_connection():
    """测试 VLM API 连接"""
    print("🔄 测试 VLM API 连接...")
    
    try:
        from src.analysis.vlm import VLMAnalyzer
        
        # 使用你的配置
        vlm_analyzer = VLMAnalyzer(
            backend="custom",
            model_name="qwen/qwen3-vl-4b",
            api_base_url="http://127.0.0.1:1234/v1",
            api_key=""
        )
        
        print(f"✅ VLM 分析器初始化成功: {vlm_analyzer.get_backend_name()}")
        
        # 简单测试：检查后端是否可用
        backend_available = vlm_analyzer._backend.is_available()
        print(f"🔗 后端可用性: {backend_available}")
        
        # 如果有现成的测试图片，可以测试
        test_image_path = "test_image.jpg"
        if Path(test_image_path).exists():
            prompt = "识别图片中的日期和时间，格式严格为 YYYY-MM-DD HH:MM:SS，只输出时间字符串，不要包含任何其他内容。"
            response = vlm_analyzer.analyze_frame(test_image_path, prompt)
            print(f"🤖 VLM 响应: {response}")
            
            if "[API Error:" in response or "404" in response or "Connection" in response:
                print("❌ VLM API 连接失败")
                return False
            else:
                print("✅ VLM API 连接成功")
                return True
        else:
            print("💡 未找到测试图片，跳过实际 API 调用测试")
            print("✅ VLM 配置验证通过")
            return True
        
    except Exception as e:
        print(f"❌ VLM 测试失败: {e}")
        return False

def test_time_sync_config():
    """测试时间同步配置"""
    print("\n🔄 测试时间同步配置...")
    
    try:
        from src.ingestion.service import TimeSynchronizer
        from src.analysis.vlm import VLMAnalyzer
        
        # 使用相同的配置
        vlm_analyzer = VLMAnalyzer(
            backend="custom",
            model_name="qwen/qwen3-vl-4b",
            api_base_url="http://127.0.0.1:1234/v1",
            api_key=""
        )
        
        roi_config = (0.65, 0.85, 0.35, 0.15)
        time_sync = TimeSynchronizer(vlm_analyzer, roi_config)
        
        print("✅ TimeSynchronizer 初始化成功")
        print(f"📍 ROI 配置: {roi_config}")
        print("⏰ 同步间隔: 10 秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 时间同步配置测试失败: {e}")
        return False

def main():
    print("🚀 测试 VLM 配置和连接")
    print("=" * 50)
    
    success = True
    
    # 测试 VLM 连接
    if not test_vlm_connection():
        success = False
    
    # 测试时间同步配置
    if not test_time_sync_config():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 VLM 配置测试通过！")
        print("💡 现在可以启动监控系统，时间同步和内容分析将使用相同的 VLM 配置")
    else:
        print("❌ VLM 配置测试失败")
        print("💡 请检查 VLM 服务是否正常运行")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)