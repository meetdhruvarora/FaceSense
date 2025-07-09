import moviepy.editor as mp
import speech_recognition as sr

def extract_audio_from_video(video_file, audio_output_file="extracted_audio.wav"):
    # Load the video file
    video = mp.VideoFileClip(video_file)
    
    # Extract the audio
    video.audio.write_audiofile(audio_output_file)
    
    return audio_output_file

def recognize_speech_from_video(video_file):

    audio_file = extract_audio_from_video(video_file)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Interview Transcription: {text}")
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with request: {e}"

# video_file = r"C:\Users\dhruv\OneDrive\Pictures\Camera Roll\WIN_20241003_20_08_58_Pro.mp4" 
# response = recognize_speech_from_video(video_file)
# print(f"Interview Transcription: {response}")
