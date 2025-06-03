import tkinter as tk
from tkinter import messagebox
import enum
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import webbrowser
from pytube import YouTube
from numpy.lib.type_check import imag
import torch
from torch.functional import norm
import torchvision.transforms.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector

from model.model import Mini_Xception
from utils import get_label_emotion, normalization, histogram_equalization, standerlization
from face_alignment.face_alignment import FaceAlignment

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daftar tipe video untuk setiap kategori emosi
video_types = {
    'Senang': ['energetic'],
    'Sedih': ['funny', 'motivational', 'adorable'],
    'Jijik': ['adorable', 'funny'],
    'Marah': ['calming', 'adorable'],
    'Takut': ['motivational', 'calming'],
    'Terkejut': ['calming'],
    'Netral': ['energetic', 'calming', 'funny', 'motivational', 'adorable']
}

video_links = {
    'energetic': [
        'https://www.youtube.com/watch?v=BpU61EtpaDY',
        'https://www.youtube.com/watch?v=HduKGO9oPPI',
        'https://www.youtube.com/watch?v=rjOhZZyn30k',
        'https://www.youtube.com/watch?v=9AEoUa0Hlso',
        'https://www.youtube.com/watch?v=ZbZSe6N_BXs',
        'https://www.youtube.com/watch?v=gdZLi9oWNZg',
        'https://www.youtube.com/watch?v=o_v9MY_FMcw'
    ],
    'motivational': [
        'https://www.youtube.com/shorts/_c_IXj0BE1s',
        'https://www.youtube.com/shorts/hI9EW_3-gUI',
        'https://www.youtube.com/shorts/ZY-7xINpmak',
        'https://www.youtube.com/watch?v=wnHW6o8WMas',
        'https://www.youtube.com/watch?v=NSQVsvYH9xo',
        'https://www.youtube.com/watch?v=NSQVsvYH9xo',
        'https://www.youtube.com/shorts/A9IqY0lmNkY'
    ],
    'funny': [
        'https://www.youtube.com/watch?v=-72P_EFphSc',
        'https://www.youtube.com/watch?v=G9nqB8BwGHU',
        'https://www.youtube.com/watch?v=HRiCRmPYAl8',
        'https://www.youtube.com/watch?v=8dwgbtN2GPU',
        'https://www.youtube.com/watch?v=I-JtHoQtUNY',
        'https://www.youtube.com/watch?v=2mSr6GgQV2Y',
        'https://www.youtube.com/watch?v=veZOrXVHf7U'
    ],
    'adorable': [
        'https://www.youtube.com/shorts/dFg8Nu2X5Mo',
        'https://www.youtube.com/shorts/i_XSK-53PYU',
        'https://www.youtube.com/shorts/Po098TRdOn4',
        'https://www.youtube.com/shorts/RMVGyNclAtU',
        'https://www.youtube.com/shorts/XTi0-Y5xhXA',
        'https://www.youtube.com/shorts/nI-an_Gh7OA',
        'https://www.youtube.com/shorts/cspDcNkMUws'
    ],
    'calming': [
        'https://www.youtube.com/watch?v=6WZ67f9M3RE',
        'https://www.youtube.com/watch?v=d-3cEQ1d1E4',
        'https://www.youtube.com/watch?v=uHvk5d1i6UY',
        'https://www.youtube.com/watch?v=1R47EQrxgfw',
        'https://www.youtube.com/watch?v=0dkZcuQux80',
        'https://www.youtube.com/watch?v=EQqj7TMqfYM',
        'https://www.youtube.com/watch?v=JYPIDIQSvb8'
    ]
}

def get_youtube_video_title(video_url):
    try:
        yt = YouTube(video_url)
        video_title = yt.title
        return video_title
    except Exception as e:
        print(f"Error: {e}")
        return video_url

def play_emotion_video(emotion):
    if emotion in video_types:
        # Pilih satu tipe video secara acak untuk emosi yang diberikan
        selected_video_type = np.random.choice(video_types[emotion])
        # Pilih satu link secara acak dari tipe video yang dipilih
        video_url = np.random.choice(video_links[selected_video_type])
        
        # Dapatkan judul video YouTube
        video_title = get_youtube_video_title(video_url)

        # Tampilkan pesan prompt sebelum membuka tautan
        root = tk.Tk()
        root.withdraw()

        # Ubah pesan prompt berdasarkan emosi
        if emotion.lower() == 'sedih':
            prompt_message = f"Anda terlihat sedang sedih. Apakah Anda ingin menonton video berjudul:\n'{video_title}' untuk menghiburmu?"
        elif emotion.lower() == 'marah':
            prompt_message = f"Anda terlihat sedang marah. Apakah Anda ingin menonton video berjudul:\n'{video_title}' untuk menenangkanmu?"
        elif emotion.lower() == 'takut':
            prompt_message = f"Anda terlihat sedang takut. Apakah Anda ingin menonton video berjudul:\n'{video_title}' untuk menenangkanmu?"
        elif emotion.lower() == 'terkejut':
            prompt_message = f"Anda terlihat sedang terkejut. Apakah Anda ingin menonton video berjudul:\n'{video_title}' untuk menenangkanmu?"
        elif emotion.lower() == 'jijik':
            prompt_message = f"Anda terlihat jijik. Apakah Anda ingin menonton video berjudul:\n'{video_title}' untuk menenangkanmu?"
        elif emotion.lower() == 'senang':
            prompt_message = f"Anda terlihat senang! Sepertinya lagu:\n'{video_title}' cocok untuk suasana hatimu saat ini!"
        else:
            # Untuk emosi lainnya, gunakan pesan umum
            prompt_message = f"Suasana hati anda {emotion.lower()}! Apakah Anda ingin menonton video berjudul:\n'{video_title}'?"

        user_response = messagebox.askokcancel("Prompt", prompt_message)

        if user_response:
            webbrowser.open(video_url)
            global last_opened_time
            last_opened_time = time.time()

def main(args):
    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()

    checkpoint = torch.load(args.pretrained, map_location=device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    face_alignment = FaceAlignment()

    root = 'face_detector'
    face_detector = None
    if args.haar:
        face_detector = HaarCascadeDetector(root)
    else:
        face_detector = DnnDetector(root)

    video = None
    isOpened = False
    if not args.image:
        if args.path:
            video = cv2.VideoCapture(args.path) 
        else:
            video = cv2.VideoCapture(0)
        isOpened = video.isOpened()
    
    t1 = 0
    t2 = 0

    last_detected_time = time.time()
    cooldown_time = 15
    
    last_opened_time = None
    
    while args.image or isOpened:
        if args.image:
            frame = cv2.imread(args.path)
        else:
            _, frame = video.read()
            isOpened = video.isOpened()    

        if args.path:
            frame = cv2.resize(frame, (640, 480))

        t2 = time.time()
        fps = round(1/(t2-t1))
        t1 = t2

        faces = face_detector.detect_faces(frame)

        for face in faces:
            (x, y, w, h) = face

            input_face = face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (48, 48))

            input_face = histogram_equalization(input_face)
            cv2.imshow('input face', cv2.resize(input_face, (120, 120)))

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                input_face = input_face.to(device)
                t = time.time()
                emotion_output = mini_xception(input_face)

                torch.set_printoptions(precision=6)
                softmax = torch.nn.Softmax(dim=0)
                emotions_soft = softmax(emotion_output.squeeze()).reshape(-1, 1).cpu().detach().numpy()
                emotions_soft = np.round(emotions_soft, 3)

                detected_emotion = torch.argmax(emotion_output)
                percentage = round(emotions_soft[detected_emotion].item(), 2)
                detected_emotion = detected_emotion.squeeze().cpu().detach().item()
                detected_emotion_label = get_label_emotion(detected_emotion)

                frame[y - 30:y, x:x + w] = (50, 50, 50)
                cv2.putText(frame, detected_emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200))
                cv2.putText(frame, str(percentage), (x + w - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

                current_time = time.time()
                if current_time - last_detected_time >= 5 and (last_opened_time is None or current_time - last_opened_time >= cooldown_time):
                    play_emotion_video(detected_emotion_label)
                    last_detected_time = current_time
    
        cv2.putText(frame, str(fps), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.imshow("Video", frame)   
        if cv2.waitKey(1) & 0xff == 27:
            video.release()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true', help='run the haar cascade face detector')
    parser.add_argument('--pretrained',type=str,default='checkpoint/model_weights/weights_epoch_75.pth.tar' 
                        ,help='load weights')
    parser.add_argument('--head_pose', action='store_true', help='visualization of head pose euler angles')
    parser.add_argument('--path', type=str, default='', help='path to video to test')
    parser.add_argument('--image', action='store_true', help='specify if you test image or not')
    args = parser.parse_args()

    main(args)
