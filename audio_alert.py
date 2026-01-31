import os
import threading

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available, audio alerts disabled")

import config

class AudioAlert:
    def __init__(self):
        self.alert_playing = False
        self.alarm_sound = None
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self._initialize_alarm_sound()
            except Exception as e:
                print(f"Warning: Could not initialize pygame mixer: {e}")
                print("Audio alerts will be disabled")
    
    def _initialize_alarm_sound(self):
        alarm_path = os.path.join(config.BASE_DIR, "alarm.wav")
        
        if not os.path.exists(alarm_path):
            self._generate_beep_sound(alarm_path)
        
        try:
            self.alarm_sound = pygame.mixer.Sound(alarm_path)
            self.alarm_sound.set_volume(0.7)
        except Exception as e:
            print(f"Warning: Could not load alarm sound: {e}")
    
    def _generate_beep_sound(self, output_path):
        import wave
        import math
        import struct
        
        duration = 0.5
        frequency = 1000
        sample_rate = 44100
        num_samples = int(duration * sample_rate)
        
        try:
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                
                for i in range(num_samples):
                    value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                    data = struct.pack('<h', value)
                    wav_file.writeframesraw(data)
        except Exception as e:
            print(f"Could not generate alarm sound: {e}")
    
    def play_theft_alarm(self, alert_type='theft'):
        if not PYGAME_AVAILABLE or not self.alarm_sound:
            return
        
        repeat_count = {
            'theft': 3,
            'potential_theft': 2,
            'suspicious_retrieval': 2,
            'abandonment': 1,
            'lost_object': 1
        }
        
        repeats = repeat_count.get(alert_type, 1)
        thread = threading.Thread(target=self._play_sound_threaded, args=(repeats,))
        thread.daemon = True
        thread.start()
    
    def _play_sound_threaded(self, repeat_count):
        self.alert_playing = True
        
        try:
            for i in range(repeat_count):
                if self.alarm_sound:
                    self.alarm_sound.play()
                    pygame.time.wait(int(self.alarm_sound.get_length() * 1000) + 200)
        except Exception as e:
            print(f"Error playing sound: {e}")
        
        self.alert_playing = False
