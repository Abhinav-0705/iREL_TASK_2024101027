"""
Step 1: Data Collection - Extract transcripts from YouTube videos
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import whisper


class TranscriptExtractor:
    """Extract transcripts from YouTube videos using API or Whisper"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model = None
        
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        else:
            raise ValueError(f"Invalid YouTube URL: {url}")
    
    def get_youtube_transcript(self, video_url: str, languages: List[str] = ['hi', 'en']) -> Optional[Dict]:
        """
        Get transcript using YouTube Transcript API
        
        Args:
            video_url: YouTube video URL
            languages: List of language codes to try
            
        Returns:
            Dictionary with transcript data or None if unavailable
        """
        try:
            video_id = self.extract_video_id(video_url)
            
            # For youtube-transcript-api v1.2.4, need to instantiate the class
            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(video_id, languages=languages)
            
            # Convert FetchedTranscriptSnippet objects to dictionaries
            transcript_data = []
            for entry in transcript_obj:
                transcript_data.append({
                    'text': entry.text,
                    'start': entry.start,
                    'duration': entry.duration
                })
            
            # Format transcript
            full_text = " ".join([entry['text'] for entry in transcript_data])
            
            return {
                "video_id": video_id,
                "source": "youtube_api",
                "transcript": transcript_data,
                "full_text": full_text
            }
            
        except Exception as e:
            print(f"Could not get YouTube transcript: {e}")
            return None
    
    def download_audio(self, video_url: str, video_id: str) -> Optional[str]:
        """
        Download audio from YouTube video using yt-dlp
        
        Args:
            video_url: YouTube video URL
            video_id: Video identifier
            
        Returns:
            Path to downloaded audio file or None
        """
        try:
            output_template = str(self.output_dir / f"{video_id}_audio.%(ext)s")
            final_file = str(self.output_dir / f"{video_id}_audio.m4a")
            
            ydl_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': output_template,
                # Try to avoid throttling/bot-protection issues
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
            if os.path.exists(final_file):
                return final_file
            return None
            
        except Exception as e:
            print(f"Could not download audio with yt-dlp: {e}")
            return None
    
    def transcribe_with_whisper(self, audio_file: str, model_size: str = "medium") -> Dict:
        """
        Transcribe audio using OpenAI Whisper
        
        Args:
            audio_file: Path to audio file
            model_size: Whisper model size (tiny, base, small, medium, large)
            
        Returns:
            Dictionary with transcript data
        """
        # Load model if not already loaded
        if self.whisper_model is None or self.whisper_model.model_size != model_size:
            print(f"Loading Whisper model ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            self.whisper_model.model_size = model_size
        
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(
            audio_file,
            language="hi",  # Hindi (will auto-detect English)
            task="transcribe"
        )
        
        return {
            "source": "whisper",
            "full_text": result["text"],
            "segments": result["segments"],
            "language": result["language"]
        }
    
    def extract_transcript(
        self, 
        video_url: str, 
        video_id: str,
        prefer_whisper: bool = False,
        whisper_model_size: str = "medium"
    ) -> Dict:
        """
        Extract transcript using YouTube API or Whisper
        
        Args:
            video_url: YouTube video URL
            video_id: Video identifier
            prefer_whisper: If True, use Whisper even if API transcript available
            whisper_model_size: Whisper model size to use
            
        Returns:
            Dictionary with transcript data
        """
        # Try YouTube API first (if not preferring Whisper)
        if not prefer_whisper:
            transcript = self.get_youtube_transcript(video_url)
            if transcript:
                print(f"✓ Got transcript from YouTube API for {video_id}")
                return transcript
        
        # Fallback to Whisper
        print(f"Using Whisper for transcription of {video_id}...")
        audio_file = self.download_audio(video_url, video_id)
        
        if audio_file:
            transcript = self.transcribe_with_whisper(audio_file, whisper_model_size)
            transcript["video_id"] = video_id
            print(f"✓ Transcribed using Whisper for {video_id}")
            return transcript
        else:
            raise Exception(f"Could not extract transcript for {video_id}")
    
    def save_transcript(self, transcript: Dict, video_id: str):
        """Save transcript to JSON file"""
        output_file = self.output_dir / f"{video_id}_transcript.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved transcript to {output_file}")
    
    def process_videos(self, videos: List[Dict], prefer_whisper: bool = False):
        """
        Process multiple videos
        
        Args:
            videos: List of video dictionaries with 'id' and 'url' keys
            prefer_whisper: If True, use Whisper for all videos
        """
        for video in videos:
            video_id = video['id']
            video_url = video['url']
            
            if not video_url:
                print(f"⚠ No URL provided for {video_id}, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {video['title']} ({video_id})")
            print(f"{'='*60}")
            
            try:
                transcript = self.extract_transcript(
                    video_url, 
                    video_id,
                    prefer_whisper=prefer_whisper
                )
                self.save_transcript(transcript, video_id)
                
            except Exception as e:
                print(f"✗ Error processing {video_id}: {e}")
                continue


if __name__ == "__main__":
    # Example usage
    from src.utils.config_loader import ConfigLoader
    
    config = ConfigLoader()
    videos_config = config.load_videos()
    
    extractor = TranscriptExtractor()
    extractor.process_videos(videos_config['videos'])
