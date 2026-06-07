"""
Notification Manager for AI-Sentinel
Supports PushPlus (WeChat) and Bark (iOS) notifications.
"""
import requests
import json
import logging
import threading
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class NotificationProvider(ABC):
    @abstractmethod
    def send(self, title: str, content: str, **kwargs) -> bool:
        pass

class WeChatOfficialProvider(NotificationProvider):
    """
    WeChat Official Account (Sandbox/Production)
    API: https://api.weixin.qq.com/cgi-bin/message/template/send
    """
    def __init__(self, app_id: str, app_secret: str, open_ids: str, template_id: str):
        self.app_id = app_id
        self.app_secret = app_secret
        # Support multiple OpenIDs separated by comma
        self.open_ids = [oid.strip() for oid in open_ids.split(",") if oid.strip()]
        self.template_id = template_id
        
        self.access_token = None
        self.token_expires_at = 0

    def _get_access_token(self) -> Optional[str]:
        """Get or refresh access token"""
        now = time.time()
        if self.access_token and now < self.token_expires_at:
            return self.access_token
            
        try:
            url = "https://api.weixin.qq.com/cgi-bin/token"
            params = {
                "grant_type": "client_credential",
                "appid": self.app_id,
                "secret": self.app_secret
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if "access_token" in data:
                self.access_token = data["access_token"]
                # Expires in 7200s, refresh a bit earlier (e.g. 6000s)
                self.token_expires_at = now + 6000 
                return self.access_token
            else:
                logger.error(f"WeChat AccessToken Error: {data}")
                return None
        except Exception as e:
            logger.error(f"Failed to get WeChat AccessToken: {e}")
            return None

    def _upload_image(self, image_path: str, access_token: str) -> Optional[str]:
        """Upload image to WeChat temporary media and return media_id"""
        try:
            url = f"https://api.weixin.qq.com/cgi-bin/media/upload?access_token={access_token}&type=image"
            with open(image_path, 'rb') as f:
                files = {'media': f}
                resp = requests.post(url, files=files, timeout=20)
                data = resp.json()
                if "media_id" in data:
                    return data["media_id"]
                else:
                    logger.error(f"WeChat Image Upload Error: {data}")
                    return None
        except Exception as e:
            logger.error(f"Failed to upload image to WeChat: {e}")
            return None

    def _send_custom_image(self, open_id: str, media_id: str, access_token: str):
        """Send image via Customer Service Message (48h interaction limit)"""
        try:
            url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={access_token}"
            payload = {
                "touser": open_id,
                "msgtype": "image",
                "image": {
                    "media_id": media_id
                }
            }
            resp = requests.post(url, json=payload, timeout=5)
            data = resp.json()
            if data.get("errcode") != 0:
                logger.warning(f"Failed to send custom image to {open_id} (likely 48h limit): {data}")
            else:
                logger.info(f"Custom image sent to {open_id}")
        except Exception as e:
            logger.error(f"Failed to send custom image: {e}")

    def send(self, title: str, content: str, **kwargs) -> bool:
        token = self._get_access_token()
        if not token:
            return False
            
        url = f"https://api.weixin.qq.com/cgi-bin/message/template/send?access_token={token}"
        
        # Clean content (Remove HTML tags if any, as WeChat Template is text-only)
        # Simple regex to remove <br>, <p>, etc.
        # Also replace newlines with spaces to prevent chunking issues (Text starting with \n might be hidden)
        # Also remove "监控时间：xxxx" pattern from description since we send it separately in {{time}}
        import re
        clean_content = re.sub(r'<[^>]+>', '', content).strip()
        clean_content = re.sub(r'监控时间[：:]\s*\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}', '', clean_content).strip()
        clean_content = clean_content.replace('\n', ' ')
        # Remove trailing punctuation (commas) if any, as user requested cleaner output
        clean_content = clean_content.rstrip("，,")
        
        # Prepare data based on standard template conventions
        # Strategy: Send a "superset" of common keys to handle various user template definitions
        
        # Extract metadata
        risk_type = kwargs.get("risk_type", "AI检测报警")
        severity = kwargs.get("severity", "Info")
        alert_time = kwargs.get("alert_time", time.strftime("%H:%M:%S"))
        
        # Format Keyword1: [Severity] Type
        # Map severity to Chinese for better readability
        sev_map = {"Critical": "紧急", "High": "高危", "Medium": "中风险", "Low": "低风险", "Info": "提示"}
        sev_cn = sev_map.get(severity, severity)
        kw1_value = f"[{sev_cn}] {risk_type}"
        kw1_color = "#FF0000" if severity in ["Critical", "High"] else "#173177"
        
        # Common default keys (Fallback for {{keyword1}} {{remark}})
        # Split description into chunks if too long (bypass field limit)
        # Note: clean_content might still be long
        
        # Simple chunking for remark fields (Reduced to 13 chars to be safe with Chinese width)
        remarks = [clean_content[i:i+13] for i in range(0, len(clean_content), 13)]
        
        template_data = {
            # Our recommended keys (User configured {{title}} {{content}})
            "title": {"value": f"🚨 {kw1_value}", "color": kw1_color},
            # Remove redundant time prefix to save space for description
            "content": {"value": clean_content, "color": "#173177"},
            "time": {"value": time.strftime("%Y-%m-%d %H:%M:%S"), "color": "#333333"},
            
            # Common default keys (Fallback for {{keyword1}} {{remark}})
            "first": {"value": f"检测到异常事件：{risk_type}", "color": kw1_color},
            "keyword1": {"value": kw1_value, "color": kw1_color},
            "keyword2": {"value": alert_time, "color": "#173177"},
            "keyword3": {"value": severity, "color": "#FF0000" if severity == "Critical" else "#333333"},
            
            # Multi-line remarks
            "remark": {"value": remarks[0] if len(remarks) > 0 else "", "color": "#333333"},
            "remark2": {"value": remarks[1] if len(remarks) > 1 else "", "color": "#333333"},
            "remark3": {"value": remarks[2] if len(remarks) > 2 else "", "color": "#333333"},
            "remark4": {"value": remarks[3] if len(remarks) > 3 else "", "color": "#333333"},
            "remark5": {"value": remarks[4] if len(remarks) > 4 else "..." if len(remarks) > 5 else "", "color": "#333333"}
        }
        
        success_count = 0
        for open_id in self.open_ids:
            try:
                payload = {
                    "touser": open_id,
                    "template_id": self.template_id,
                    "data": template_data
                }
                
                resp = requests.post(url, json=payload, timeout=5)
                data = resp.json()
                
                if data.get("errcode") == 0:
                    success_count += 1
                else:
                     logger.error(f"WeChat Send Error ({open_id}): {data}")
            except Exception as e:
                logger.error(f"WeChat Request Failed: {e}")
        
        # Best Effort: Send Image if provided
        image_path = kwargs.get("image_path")
        if image_path:
            # Upload once
            media_id = self._upload_image(image_path, token)
            if media_id:
                for open_id in self.open_ids:
                    self._send_custom_image(open_id, media_id, token)
                
        return success_count > 0

class PushPlusProvider(NotificationProvider):
    """
    PushPlus (微信推送)
    API: http://www.pushplus.plus/send
    """
    def __init__(self, token: str):
        self.token = token
        self.url = "http://www.pushplus.plus/send"

    def send(self, title: str, content: str, **kwargs) -> bool:
        if not self.token:
            return False
            
        try:
            payload = {
                "token": self.token,
                "title": title,
                "content": content,
                "template": "html",
                "channel": "wechat"
            }
            if "topic" in kwargs:
                payload["topic"] = kwargs["topic"]
                
            resp = requests.post(self.url, json=payload, timeout=5)
            data = resp.json()
            
            if data.get("code") == 200:
                return True
            else:
                logger.error(f"PushPlus error: {data.get('msg')}")
                return False
        except Exception as e:
            logger.error(f"PushPlus request failed: {e}")
            return False

class BarkProvider(NotificationProvider):
    """
    Bark (iOS Push)
    API: https://api.day.app/{key}/{title}/{content}
    """
    def __init__(self, key: str, server_url: str = "https://api.day.app"):
        self.key = key
        self.server_url = server_url.rstrip("/")

    def send(self, title: str, content: str, **kwargs) -> bool:
        if not self.key:
            return False
            
        try:
            # Construct URL
            # Standard: /{key}/{title}/{content}
            url = f"{self.server_url}/{self.key}/{title}/{content}"
            
            params = {
                "group": "AI-Sentinel",
                "icon": kwargs.get("icon", ""),
                "level": kwargs.get("level", "active"), # active, timeSensitive, passive
                "sound": kwargs.get("sound", "minuet")
            }
            
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            if data.get("code") == 200:
                return True
            else:
                logger.error(f"Bark error: {data.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Bark request failed: {e}")
            return False

class NotificationManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(NotificationManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.providers: Dict[str, NotificationProvider] = {}
        self.active_provider_name: Optional[str] = None
        self.last_sent_times: Dict[str, float] = {} # Deduplication key -> timestamp
        self.global_last_sent_time = 0.0 # Track last alert time globally
        self.cooldown_seconds = 60 # Default debounce
        
        self.config = {}
        self._initialized = True
        logger.info("NotificationManager initialized")

    def configure(self, config: Dict[str, Any]):
        """
        Reload configuration.
        config format:
        {
            "enabled": bool,
            "provider": "pushplus" | "bark",
            "pushplus_token": "...",
            "bark_key": "...",
            "cooldown": 60
        }
        """
        self.config = config
        self.providers.clear()
        
        if not config.get("enabled", False):
            self.active_provider_name = None
            return

        # Initialize providers
        # WeChat Official
        wx_appid = config.get("wechat_appid")
        wx_secret = config.get("wechat_secret")
        # Support both 'wechat_touser' and 'wechat_openid' keys
        wx_openid = config.get("wechat_touser", config.get("wechat_openid", ""))
        wx_template = config.get("wechat_template_id")
        
        if wx_appid and wx_secret and wx_openid and wx_template:
            self.providers["wechat"] = WeChatOfficialProvider(wx_appid, wx_secret, wx_openid, wx_template)
            
        # Keep PushPlus for backward compatibility if needed, or remove
        pp_token = config.get("pushplus_token")
        if pp_token:
            self.providers["pushplus"] = PushPlusProvider(pp_token)
            
        bark_key = config.get("bark_key")
        if bark_key:
            self.providers["bark"] = BarkProvider(bark_key)
            
        self.active_provider_name = config.get("provider")
        self.cooldown_seconds = config.get("cooldown", 60)
        
        logger.info(f"NotificationManager configured. Active: {self.active_provider_name}")

    def send_alert(self, title: str, content: str, dedup_key: Optional[str] = None, **kwargs):
        """
        Send alert asynchronously.
        dedup_key: If provided, prevents sending same alert within cooldown period.
        """
        if not self.active_provider_name or self.active_provider_name not in self.providers:
            # logger.debug("Notification skipped: No active provider")
            return

        # Deduplication check
        now = time.time()
        
        # 1. Global Anti-Spam (Hard limit: at least 15s between ANY alerts to prevent mixed-type flooding)
        # unless user configured cooldown is lower than 15
        global_limit = min(15.0, self.cooldown_seconds)
        if now - self.global_last_sent_time < global_limit:
             logger.debug(f"Notification suppressed (global cooldown {global_limit}s)")
             return

        # 2. Per-Type Deduplication
        if dedup_key:
            last_time = self.last_sent_times.get(dedup_key, 0)
            if now - last_time < self.cooldown_seconds:
                logger.debug(f"Notification suppressed (type cooldown {self.cooldown_seconds}s): {dedup_key}")
                return
            self.last_sent_times[dedup_key] = now
            
        # Update global time
        self.global_last_sent_time = now

        # Run in thread to not block main loop
        threading.Thread(
            target=self._send_sync,
            args=(self.active_provider_name, title, content),
            kwargs=kwargs,
            daemon=True
        ).start()

    def _send_sync(self, provider_name: str, title: str, content: str, **kwargs):
        provider = self.providers.get(provider_name)
        if provider:
            success = provider.send(title, content, **kwargs)
            if success:
                logger.info(f"Notification sent via {provider_name}: {title}")
            else:
                logger.warning(f"Notification failed via {provider_name}")
                
    def test_connection(self, provider_name: str) -> bool:
        """Synchronous test for UI"""
        if provider_name not in self.providers:
            return False
            
        return self.providers[provider_name].send(
            title="AI-Sentinel 测试",
            content="恭喜！您的通知服务配置成功。\nThis is a test message from AI-Sentinel.",
            level="active"
        )
