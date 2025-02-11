import whisper
import torch
import logging
import time

from transformers import pipeline
from datetime import datetime
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VNSTTTranscriptionService:
    def __init__(self):
        self.model_id = "vinai/PhoWhisper-medium"

    def transcribe(self, audio_data: bytes) -> str:
        start_time = time.perf_counter()
        start_datetime = datetime.now().isoformat()
        # logger.info(f"Starting transcription at: {start_datetime}")
        
        try:
            # Time the pipeline execution specifically
            pipeline_start = time.perf_counter()
            transcriber = pipeline("automatic-speech-recognition", model=self.model_id)
            
            with NamedTemporaryFile(suffix=".wav") as temp:
                with open(temp.name, "wb") as temp_file:
                    temp_file.write(audio_data)
                
                result = transcriber(temp.name)
                output = result.get("text", "")
            
            pipeline_duration = time.perf_counter() - pipeline_start
            transcription_text = output
            
            # Calculate total duration
            total_duration = time.perf_counter() - start_time
            
            # # Log timing information and transcribed text
            # logger.info(
            #     f"Transcription completed - "
            #     f"Start time: {start_datetime}, "
            #     f"Pipeline execution: {pipeline_duration:.3f}s, "
            #     f"Total time: {total_duration:.3f}s, "
            #     f"Result length: {len(transcription_text)} chars, "
            #     f"Transcribed text: {transcription_text}"
            # )
            
            return transcription_text
        except Exception as e:
            error_duration = time.perf_counter() - start_time
            logger.error(
                f"Error in transcription after {error_duration:.3f}s: {e}"
            )
            raise