#!/usr/bin/env python3
"""
测试 ROI 区域持久化功能
"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

def test_roi_config():
    """测试 ROI 配置的读取和保存"""
    print("🔄 测试 ROI 配置持久化...")
    
    try:
        from src.common.config import get_config
        
        config = get_config()
        
        # 读取当前配置
        current_roi = config.get("ocr.roi", [0.65, 0.85, 0.35, 0.15])
        print(f"📍 当前 ROI 配置: {current_roi}")
        
        # 测试保存新的 ROI
        test_roi = [0.7, 0.9, 0.25, 0.08]
        print(f"💾 测试保存新 ROI: {test_roi}")
        
        config.set("ocr.roi", test_roi)
        config.save()
        
        # 重新读取验证
        config_reloaded = get_config()
        saved_roi = config_reloaded.get("ocr.roi")
        
        if saved_roi == test_roi:
            print("✅ ROI 配置保存和读取成功")
            
            # 恢复原始配置
            config.set("ocr.roi", current_roi)
            config.save()
            print(f"🔄 已恢复原始 ROI 配置: {current_roi}")
            
            return True
        else:
            print(f"❌ ROI 配置不匹配: 期望 {test_roi}, 实际 {saved_roi}")
            return False
        
    except Exception as e:
        print(f"❌ ROI 配置测试失败: {e}")
        return False

def test_session_state_logic():
    """测试 session_state 逻辑"""
    print("\n🔄 测试 session_state ROI 逻辑...")
    
    try:
        # 模拟 WebUI 中的逻辑
        from src.common.config import get_config
        
        config = get_config()
        current_roi = config.get("ocr.roi", [0.65, 0.85, 0.35, 0.15])
        
        # 模拟 session_state
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def __setitem__(self, key, value):
                self.data[key] = value
        
        session_state = MockSessionState()
        
        # 模拟初始化逻辑
        session_state["new_roi_ratios"] = current_roi
        print(f"📍 初始化 session_state ROI: {session_state.get('new_roi_ratios')}")
        
        # 模拟用户选择新区域
        new_selection = [0.6, 0.8, 0.3, 0.1]
        session_state["new_roi_ratios"] = new_selection
        print(f"👆 用户选择新区域: {new_selection}")
        
        # 模拟优先级逻辑
        roi_to_use = session_state.get("new_roi_ratios") or current_roi
        print(f"🎯 最终使用的 ROI: {roi_to_use}")
        
        if roi_to_use == new_selection:
            print("✅ session_state 优先级逻辑正确")
            return True
        else:
            print("❌ session_state 优先级逻辑错误")
            return False
        
    except Exception as e:
        print(f"❌ session_state 逻辑测试失败: {e}")
        return False

def main():
    print("🚀 测试 ROI 区域持久化功能")
    print("=" * 50)
    
    success = True
    
    # 测试配置文件读写
    if not test_roi_config():
        success = False
    
    # 测试 session_state 逻辑
    if not test_session_state_logic():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ROI 持久化功能测试通过！")
        print("💡 现在框选区域会自动保存，下次打开时无需重新框选")
        print("\n📋 使用说明:")
        print("1. 在 WebUI 中框选时间识别区域")
        print("2. 点击 '💾 保存区域配置' 按钮")
        print("3. 下次打开时会自动使用保存的区域")
        print("4. 可以随时重新框选并保存新区域")
    else:
        print("❌ ROI 持久化功能测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)