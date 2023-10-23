import requests
import audioread
from PIL import Image
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import json
import urllib
import os
import pysrt
import whisper
from gtts import gTTS
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from icrawler.builtin import GoogleImageCrawler
from moviepy.config import change_settings
from moviepy.video.fx.all import crop
import re
import os
import pysrt
from moviepy.editor import VideoFileClip
import whisper
import datetime
import torch
import re
import logging
import threading
import configparser

# Load configuration from a config file
config = configparser.ConfigParser()
config.read('config.ini')

# Getting configurations from config
max_filename_length = int(config['General']['max_filename_length'])
logs_dir = config['General']['logs_dir']
general_log = config['General']['general_log']
google_api_key = config['API']['google_custom_search_api_key']
search_engine_id = config['API']['search_engine_id']


# Get the search query from the user
query = input("Enter search query: ")

# Truncate the query to the maximum filename length
filename = query[:min(len(query), max_filename_length)]

# Replacing all non-alphanumeric characters with a hyphen using regular expression
filename = re.sub('[^0-9a-zA-Z.-]+', '-', filename)


#settingfilepaths
output_dir = "output"
image_dir = os.path.join(output_dir,filename)
audio_dir = os.path.join(output_dir,'audio')
video_dir = os.path.join(output_dir,'video')
subtitle_dir = os.path.join(output_dir,'subtitle')
# keeping the logs file seperate
llm_log = os.path.join(logs_dir,'results.txt')


#create directories function
def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok= True)
    print(f'created {dir_path} directory')


create_dir(logs_dir)

# Initialize logging
logging.basicConfig(filename = os.path.join(logs_dir,general_log), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# logging the filename
logging.info(f'Filename: {filename}')

def change_settings(settings):
    try:
        # Your existing settings change code...
        print("Hardware acceleration is set to: ", settings["FFMPEG_HWACCEL"])

    except Exception as e:
        print("An error occurred when trying to use hardware acceleration: ", e)
        print("Falling back to running FFmpeg without hardware acceleration.")
        
        # Modify settings to not use hardware acceleration
        settings["FFMPEG_HWACCEL"] = None
        settings["FFMPEG_VIDEO_CODEC"] = "h264"
        
        # Your existing settings change code...
        print("Hardware acceleration is set to: ", settings["FFMPEG_HWACCEL"])

# Call the function with your settings
change_settings({ 
    "FFMPEG_HWACCEL": "auto",
    "FFMPEG_VIDEOPRESET": "fast",
    "FFMPEG_VIDEO_CODEC": "h264"
})


# Step 1: Search for interesting topics
def search_topic(query, api_key, search_engine_id):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}"
        res = requests.get(url)
        data = json.loads(res.text)
        return data.get('items', [])
    except Exception as e:
        logging.error(f'Error in search_topic: {str(e)}')
        return []

width, height = (1920, 1080)

# Step 2: Gather media
def gather_media(query):
    try:
        create_dir(image_dir)
        google_Crawler = GoogleImageCrawler(storage ={'root_dir': image_dir})
        print(filename)
        google_Crawler.crawl(keyword=query, min_size=(width, height), max_size=None, max_num=200)
        images = os.listdir(image_dir)
        return [os.path.join(image_dir, image) for image in images]
    except Exception as e:
        logging.error(f'Error in gather_media: {str(e)}')
        return []

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device_id = device.index
# Define a function to generate text using the model
def generate_text(description):

    prefix ="A well-crafted and beautifully written script for a video generation program, with a focus on balance and harmony."

    # set_seed(seed)
    seed = 1

    model_name = "gpt2"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    try:
        generator = pipeline('text-generation',
                             max_new_tokens=1000,
                             model=model,
                             tokenizer=tokenizer,
                             prefix = prefix,
                             device=device_id,
                             temperature=1,
                             top_k=50,
                             top_p=1,
                             repetition_penalty=1.2,
                             length_penalty=0.5,
                             do_sample=True,
                             num_beams=4,
                             no_repeat_ngram_size=3,
                             num_return_sequences=1,
                             )

        # Generate text
        additional_sentences_ = (generator(description)[0]['generated_text'])
        additional_sentences = additional_sentences_

        # Delete the model to free GPU memory
        del generator
        del model
        torch.cuda.empty_cache()

        return additional_sentences
    except Exception as e:
        print(e)
        logging.error(f'error in generator pipeline :{str(e)}')
    
    prompt = description
    # Open the file in append mode and write the log
    with open(llm_log, "a") as f:
        # Write the timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Timestamp: {timestamp}\n")

        # Write the seed value
        f.write(f"Seed value: {seed}\n")

        # Write the model parameters
        f.write(f"Model parameters: {model.config}\n")

        # Write the input prompt
        f.write(f"Input prompt: {prompt}\n")

        # Generate the text and write it to the log
        f.write(f"Generated text:\n{additional_sentences}\n\n")


# Step 3: Create audio
def create_audio(description):
    
    try:

        create_dir(audio_dir)
        # Use a pre-trained language model to generate additional sentences based on the initial description
        
        additional_sentences = generate_text(description)

        print(f"Generated audio for: {additional_sentences}")
        

        # Concatenate the original description with the additional sentences
        # text = " ".join([description] + [additional_sentences])
        text = additional_sentences
        # Generate audio file using gTTS
        tts = gTTS(text=text, lang='en')
        tts.save(os.path.join(audio_dir, filename + '.mp3'))
        return additional_sentences
    except Exception as e:
        logging.error(f'Error in create_audio: {str(e)}')
        return description

# Step 4: Create video
def create_video(images, audio_file):
    try:
        create_dir(video_dir)
        with audioread.audio_open(audio_file) as f:
            audio_duration = int(f.duration)
        image_duration = 3
        print('Total Duration: {} seconds'.format(audio_duration))

        num_loops = int(audio_duration / image_duration)
        print('number of loops:{}'.format(num_loops))

        width, height = (1920, 1080)

        clips = []
        i = 0
        while True:
            print("in while")
            for image in images:
                try:
                    clip = ImageClip(image).resize(width=width, height=height).crop(x1=0, y1=0, x2=width, y2=height).set_duration(image_duration)
                    clips.append(clip)
                    i = i + 1
                except Exception as e:
                    print(f"Error opening image: {image}. Error message: {str(e)}")
            if i >= num_loops:
                print("in if")
                break
        print("after while")
        concat_clip = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(audio_file)
        video = concat_clip.set_audio(audio)
        video.write_videofile(os.path.join(video_dir, filename + '.mp4'), fps=24)
    except Exception as e:
        logging.error(f'Error in create_video: {str(e)}')

def generate_subtitle(audio_file):
    try:
        create_dir(subtitle_dir)
        # Load the transcription model and transcribe the audio file
        try:
            model = whisper.load_model("base", device="cuda")
            result = model.transcribe(audio_file)
        except Exception as e:
            print(e)
            model = whisper.load_model("base", device="cpu")
            result = model.transcribe(audio_file)

        # Extract the transcribed text and segments from the result
        text = result["text"]
        segments = result["segments"]

        # Generate subtitle files
        subtitles = pysrt.SubRipFile()
        for i, seg in enumerate(segments):
            start_time = int(seg["start"] * 1000)  # Convert start time to milliseconds
            end_time = int(seg["end"] * 1000)  # Convert end time to milliseconds
            subtitle = pysrt.SubRipItem(index=i, start=pysrt.SubRipTime(milliseconds=start_time),
                                        end=pysrt.SubRipTime(milliseconds=end_time), text=seg["text"])
            subtitles.append(subtitle)

        # Save the subtitle file
        subtitles.save(os.path.join(subtitle_dir, filename + '.srt'))
    except Exception as e:
        logging.error(f'Error in generate_subtitle: {str(e)}')

def add_subtitles(video_file):
    try:
        create_dir(video_dir)
        # Load the subtitles from the subtitle file
        subs = pysrt.open(os.path.join(subtitle_dir, filename + ".srt"))

        # Check if there are subtitles available
        if subs:
            # Add the subtitles to the video file
            video = VideoFileClip(video_file)
            generator = lambda text: TextClip(text, font='Arial-Bold',
                                              fontsize=32, 
                                              color='white',
                                              bg_color='aqua')
            sub = SubtitlesClip(os.path.join(subtitle_dir, filename + ".srt"), generator)
            video = CompositeVideoClip([video, sub.set_pos(('center', 'bottom'))])
            video.write_videofile(os.path.join(video_dir, filename + 'with_subs.mp4'))
        else:
            print("No subtitles found")
    except Exception as e:
        logging.error(f'Error in add_subtitles: {str(e)}')

# Define main function
def main():
    try:
        # Step 1: Search for interesting topics
        search_results = search_topic(query, google_api_key, search_engine_id)
        if search_results:
            title = search_results[0]['title']
            description = search_results[0]['snippet']
            print("\nDescription\n" + description + "\n")

            url = search_results[0]['link']

            # Step 2: Gather media
            media_links = gather_media(query)
            print(len(media_links))

            # Step 3: Create audio
            create_audio(description)

            generate_subtitle(os.path.join(audio_dir, filename + '.mp3'))

            # Step 4: Create video
            create_video(media_links, os.path.join(audio_dir, filename + '.mp3'))

            # Step 5: Add subtitles
            add_subtitles(os.path.join(video_dir, filename + '.mp4'))
    except Exception as e:
        logging.error(f'Error in main: {str(e)}')

if __name__ == '__main__':
    main()
