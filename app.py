# import emotion_recognition
# import sentiment_analysis
# import transcript
# import eye_detection
# video_file = r"C:\Users\dhruv\OneDrive\Pictures\Camera Roll\WIN_20241006_18_11_47_Pro.mp4"
# eye_detection.detect_eyes_and_pose(video_file)
# text=transcript.recognize_speech_from_video(video_file)
# sentiment_analysis.analyze_sentiment_and_tone(text)
# emotion_recognition.emotion_recog(video_file)
from flask import Flask, render_template, request, redirect, url_for
import os
import emotion_recognition
import sentiment_analysis
import transcript
import eye_detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']

    if file.filename == '':
        return redirect(request.url)

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        # Call the analysis functions on the uploaded video
        ans=eye_detection.detect_eyes_and_pose_and_emotion(video_path)
        text = transcript.recognize_speech_from_video(video_path)
        sentiment = sentiment_analysis.analyze_sentiment_and_tone(text)

        return f"Eye detection and pose analysis done.{ans}. Sentiment: {sentiment}. Emotion recognition completed.Video Transcript:{text}"
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
