import speech_recognition as sr
from gtts import gTTS
import os
from OpenAIHelper import call_openai_api
import openai
recognizer = sr.Recognizer()
a=True
while(a):
    with sr.Microphone() as source:
        print("Please say something:")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            response = call_openai_api(text)
            if response and "choices" in response:
                answer = response["choices"][0]["message"]["content"]
                answer = answer.replace('#', '')
                tts = gTTS(text=answer, lang='en')
                tts.save("output.mp3")
                os.system("start output.mp3")
                print(f"{answer}")
                a=False
            else:
                print("No appropriate response found :(")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        print("Have more questions?")
        ans=input("Enter (y/n)")
        if(ans=="y"):
            a=True
        else:
            continue
