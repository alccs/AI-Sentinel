"""
HTML5 视频播放器组件 - 支持RTSP/HLS直连
避免Python转码开销，提供流畅的视频体验
"""
import streamlit as st
import streamlit.components.v1 as components


def rtsp_to_hls_proxy(rtsp_url: str) -> str:
    """
    将RTSP流转换为HLS/HTTP流（通过FFmpeg或其他代理）

    注意: 浏览器不直接支持RTSP协议,需要转码为HLS/WebRTC/MSE

    选项:
    1. FFmpeg转HLS: rtsp → HLS (.m3u8)
    2. WebRTC: 使用go2rtc/mediamtx等服务
    3. HTTP-FLV: 使用flv.js播放

    Args:
        rtsp_url: RTSP流地址

    Returns:
        可在浏览器播放的流地址
    """
    # TODO: 集成FFmpeg或go2rtc代理服务
    # 这里返回示例,实际需要配置流媒体服务器
    return rtsp_url.replace("rtsp://", "http://localhost:8554/")


def render_video_player(
    stream_url: str,
    stream_type: str = "rtsp",
    width: str = "100%",
    height: str = "600px",
    autoplay: bool = True,
    controls: bool = True
) -> None:
    """
    渲染视频播放器组件

    Args:
        stream_url: 视频流地址
        stream_type: 流类型 (rtsp/hls/file)
        width: 播放器宽度
        height: 播放器高度
        autoplay: 是否自动播放
        controls: 是否显示控件
    """

    # RTSP需要转换为浏览器支持的格式
    if stream_type == "rtsp":
        # 方案1: 提示用户使用流媒体服务器
        st.warning("""
        ⚠️ **RTSP流需要中转服务**

        浏览器不直接支持RTSP协议,推荐使用以下方案:

        **方案1 - go2rtc (推荐)**
        ```bash
        # 安装go2rtc
        docker run -d --name go2rtc -p 8554:8554 alexxit/go2rtc

        # 配置流: go2rtc.yaml
        streams:
          cam_main: {stream_url}
        ```

        **方案2 - FFmpeg手动转码**
        ```bash
        ffmpeg -i {stream_url} -c copy -f hls -hls_time 2 -hls_list_size 3 output.m3u8
        ```

        **方案3 - MediaMTX**
        ```bash
        docker run -d -p 8554:8554 bluenviron/mediamtx
        ```

        配置完成后,使用HLS地址: `http://localhost:8554/cam_main/index.m3u8`
        """.format(stream_url=stream_url))

        # 如果已配置代理,自动转换
        hls_url = rtsp_to_hls_proxy(stream_url)
        player_html = _generate_hls_player(hls_url, width, height, autoplay, controls)

    elif stream_type == "hls":
        # HLS格式直接播放
        player_html = _generate_hls_player(stream_url, width, height, autoplay, controls)

    else:
        # 本地文件或HTTP流
        player_html = _generate_html5_player(stream_url, width, height, autoplay, controls)

    components.html(player_html, height=int(height.replace("px", "")))


def _generate_hls_player(url: str, width: str, height: str, autoplay: bool, controls: bool) -> str:
    """生成HLS播放器HTML (使用hls.js)"""
    autoplay_attr = "autoplay" if autoplay else ""
    controls_attr = "controls" if controls else ""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <style>
            body {{ margin: 0; padding: 0; background: #000; }}
            #video {{ width: {width}; height: {height}; }}
        </style>
    </head>
    <body>
        <video id="video" {autoplay_attr} {controls_attr} muted></video>
        <script>
            var video = document.getElementById('video');
            var videoSrc = '{url}';

            if (Hls.isSupported()) {{
                var hls = new Hls({{
                    enableWorker: true,
                    lowLatencyMode: true,
                    backBufferLength: 90
                }});
                hls.loadSource(videoSrc);
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                    video.play();
                }});
                hls.on(Hls.Events.ERROR, function(event, data) {{
                    console.error('HLS Error:', data);
                }});
            }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                // Safari native HLS support
                video.src = videoSrc;
                video.addEventListener('loadedmetadata', function() {{
                    video.play();
                }});
            }} else {{
                alert('您的浏览器不支持HLS播放');
            }}
        </script>
    </body>
    </html>
    """


def _generate_html5_player(url: str, width: str, height: str, autoplay: bool, controls: bool) -> str:
    """生成标准HTML5播放器"""
    autoplay_attr = "autoplay" if autoplay else ""
    controls_attr = "controls" if controls else ""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background: #000; }}
            video {{ width: {width}; height: {height}; object-fit: contain; }}
        </style>
    </head>
    <body>
        <video {autoplay_attr} {controls_attr} muted>
            <source src="{url}" type="video/mp4">
            您的浏览器不支持视频播放
        </video>
    </body>
    </html>
    """


def render_webrtc_player(stream_url: str, ice_servers: list = None) -> None:
    """
    渲染WebRTC播放器 (超低延迟)

    需要配合mediamtx或其他WebRTC服务器使用

    Args:
        stream_url: WebRTC信令服务器地址
        ice_servers: STUN/TURN服务器配置
    """
    st.info("""
    🚀 **WebRTC超低延迟方案**

    1. 安装MediaMTX:
    ```bash
    docker run -d -p 8554:8554 -p 8889:8889 bluenviron/mediamtx
    ```

    2. 配置RTSP源:
    ```yaml
    # mediamtx.yml
    paths:
      cam_main:
        source: rtsp://your-camera-url
    ```

    3. WebRTC播放地址:
    ```
    http://localhost:8889/cam_main
    ```
    """)

    # WebRTC播放器需要额外的信令逻辑,这里提供基础框架
    # 生产环境建议使用成熟的WebRTC库如aiortc
    st.warning("WebRTC播放器需要额外配置,请参考文档")
