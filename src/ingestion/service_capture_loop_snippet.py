from .motion import MotionDetector

    def _capture_loop(self):
        """Main capture loop running in background thread with Motion Gating."""
        fps = self.source.get_fps()
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # Motion Gating Configuration
        motion_detector = MotionDetector()
        motion_threshold = getattr(self, "motion_threshold", 5.0) # Default 5% change
        min_interval = self.analysis_interval # Minimum time between analyses (rate limit)
        heartbeat_interval = 30.0 # Force analysis every 30s even if static
        
        last_analysis_time = 0.0
        
        logger.info(f"Capture loop started: FPS={fps:.1f}, Motion Threshold={motion_threshold}")
        
        import sys
        
        while self._running:
            loop_start = time.time()
            
            # Read frame
            success, frame = self.source.read_frame()
            
            if not success or frame is None:
                # Handle end of file or stream error
                if isinstance(self.source, FileVideoSource):
                    if self.loop_video:
                        logger.info("Video ended, looping...")
                        self.source.reset()
                        self._frame_number = 0
                        continue
                    else:
                        logger.info("Video ended")
                        break
                else:
                    # RTSP stream error
                    if not self._running:
                        break
                    logger.warning("Stream read error, attempting reconnect...")
                    time.sleep(1) 
                    if hasattr(self.source, 'reconnect'):
                        try:
                            self.source.reconnect()
                            if not self._running:
                                self.source.release()
                                break
                            print("✅ 视频流重连成功！")
                        except Exception as e:
                            logger.error(f"Reconnect failed: {e}")
                            time.sleep(2)
                    continue
            
            self._frame_number += 1
            self._stats["total_frames"] += 1
            
            # Update current frame for UI streaming
            with self._lock:
                self._current_frame = frame
            
            # --- Motion Gating Logic ---
            current_time = time.time()
            time_since_last = current_time - last_analysis_time
            
            # 1. Check Rate Limit (Must satisfy min interval)
            if time_since_last >= min_interval:
                
                # 2. Calculate Motion Score
                motion_score = motion_detector.detect(frame)
                
                # Debug logging: Print score every few frames or if significant > 0.1
                if motion_score > 0.1:
                     # Use ASCII characters to avoid encoding issues
                     msg = f"[CAM] [OpenCV] Delta: {motion_score:.2f}% (Threshold: {motion_threshold}%)"
                     if motion_score > motion_threshold:
                         print(f"\033[92m{msg} -> TRIGGER AI ANALYSIS\033[0m", flush=True) # Green
                     else:
                         print(f"\033[90m{msg} -> IGNORE\033[0m", flush=True)              # Gray
                     logger.debug(msg)
                
                # 3. Decision: Trigger or Skip?
                should_analyze = False
                
                # Condition A: Significant Motion
                if motion_score > motion_threshold:
                    should_analyze = True
                    # logger.info(f"Motion trigger: {motion_score:.2f}% > {motion_threshold}%")
                
                # Condition B: Heartbeat (Force check if stale)
                elif time_since_last > heartbeat_interval:
                    should_analyze = True
                    logger.info(f"Heartbeat trigger: static for {time_since_last:.1f}s")
                
                if should_analyze:
                    # Send copy for analysis
                    self._send_for_analysis(frame.copy(), current_time)
                    last_analysis_time = current_time
                else:
                    # Log skipped static frames periodically to avoid spam
                    if self._frame_number % 100 == 0:
                         pass

            # Maintain frame rate
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._running = False
        logger.info("Capture loop ended")
