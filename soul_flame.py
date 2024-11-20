#import packages
import sys
import random
import time
from array import array
import cv2
import mediapipe as mp
import numpy as np
import pygame
import moderngl

from detectors import FaceMeshDetector
from shaders import vert_shader, frag_shader
from particles import Flame


# load file from command line or fallback to live camera
if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
else:
    capture = cv2.VideoCapture(0)

print(cv2.getBuildInformation())
detector = FaceMeshDetector(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For calculating the FPS
previous_time = 0
current_time = 0

# pygame setup
pygame.init()
pygame.display.set_caption('Soul Flame CV')
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
running = True

display = pygame.Surface((800, 600))
ctx = moderngl.create_context()

clock = pygame.time.Clock()

quad_buffer = ctx.buffer(data=array('f', [
    # position (x, y), uv coords (x, y)
    -1.0, 1.0, 0.0, 0.0,  # topleft
    1.0, 1.0, 1.0, 0.0,   # topright
    -1.0, -1.0, 0.0, 1.0, # bottomleft
    1.0, -1.0, 1.0, 1.0,  # bottomright
]))

camera_program = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)
camera_feed = ctx.vertex_array(camera_program, [(quad_buffer, '2f 2f', 'vert', 'texcoord')])

def surf_to_texture(surf):
    tex = ctx.texture(surf.get_size(), 4)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.swizzle = 'BGRA'
    tex.write(surf.get_view('1'))
    return tex

t = 0
dt = 0

active_hand = 'left'
last_switch = None
lock_switch = False
left_median = 0
right_median = 0
draw_hand_marks = False
draw_face_marks = True
export_frames = False

file_num = 0
flame = Flame(screen)
flame_particles = [Flame(screen) for i in range(42)]
left_iris_particles = [
    Flame(screen, flame_intensity=1, intensity_ratio=15, max_radius=3) for i in range(4)
]
right_iris_particles = [
    Flame(screen, flame_intensity=1, intensity_ratio=15, max_radius=3) for i in range(4)
]
mouth_particles = [
    Flame(screen, flame_intensity=1, intensity_ratio=15, max_radius=3) for i in range(20)
]

while capture.isOpened() and running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # capture frame by frame
    ret, frame = capture.read()
    now = time.time()

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # mark the image as not writeable to to improve performance
    image.flags.writeable = False
    results = holistic_model.process(image)
    image, face_landmarks = detector.find_mesh_in_face(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
    image = cv2.divide(img_gray, img_blur, scale=256)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_blur = cv2.GaussianBlur(image, (21, 21), 0, 0)
    image = cv2.multiply(image, img_blur, scale=1/256)

    if draw_face_marks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
        )

    if draw_hand_marks:
        mp_drawing.draw_landmarks(
          image,
          results.right_hand_landmarks,
          mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
          image,
          results.left_hand_landmarks,
          mp_holistic.HAND_CONNECTIONS
        )

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Displaying FPS on the image
    cv2.putText(
        image,
        str(int(fps)) + " FPS",
        (10, 70),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv_image = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")

    imagerect = cv_image.get_rect()
    screen.blit(cv_image, imagerect)

    for f in flame_particles:
        f.x = -500
        f.y = -500

    for f in left_iris_particles + right_iris_particles:
        f.x = -500
        f.y = -500

    for l in face_landmarks.get('left_iris', []):
        for f in left_iris_particles:
            f.x = l[0] + random.randint(-15, 15)
            f.y = l[1] + random.randint(-15, 15)

    for l in face_landmarks.get('right_iris', []):
        for f in right_iris_particles:
            f.x = l[0] + random.randint(-15, 15)
            f.y = l[1] + random.randint(-15, 15)

    for f in left_iris_particles + right_iris_particles:
        f.draw_flame()


    for f in mouth_particles:
        f.x = -500
        f.y = -500

    mouth_landalmarks = face_landmarks.get('mouth')
    if mouth_landalmarks:
        y_coords = [l[1] for l in mouth_landalmarks]
        max_y = np.max(y_coords)
        min_y = np.min(y_coords)
        print('max_y', max_y, 'min_y', min_y, 'diff', max_y - min_y)

        if (max_y - min_y) > 36:
            for index, l in enumerate(mouth_landalmarks):
                mouth_particles[index].x = l[0] + random.randint(-15, 15)
                mouth_particles[index].y = l[1] + random.randint(-15, 15)


    for f in mouth_particles:
        f.draw_flame()

    if results.right_hand_landmarks and active_hand == 'right':
        for index, f in enumerate(flame_particles[:21]):
            f.x = results.right_hand_landmarks.landmark[index].x * 800
            f.y = results.right_hand_landmarks.landmark[index].y * 600

    if results.left_hand_landmarks and active_hand == 'left':
        for index, f in enumerate(flame_particles[:21]):
            f.x = results.left_hand_landmarks.landmark[index].x * 800
            f.y = results.left_hand_landmarks.landmark[index].y * 600

    for index, f in enumerate(flame_particles[:21]):
        f.draw_flame()

    if results.left_hand_landmarks:
        left_median = np.median([(l.x, l.y) for l in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        right_median = np.median([(l.x, l.y) for l in results.right_hand_landmarks.landmark])

    if abs(left_median - right_median) <= 0.10:
        if not last_switch:
            last_switch = now

        print('time since last switch:', now - last_switch)
        if last_switch and (now - last_switch) > 0.12 and not lock_switch:
            active_hand = 'left' if active_hand == 'right' else 'right'
            last_switch = None
            lock_switch = True
    else:
        lock_switch = False

    print('hands proximity', abs(left_median - right_median))

    frame_tex = surf_to_texture(screen)
    frame_tex.use(0)
    camera_program['tex'] = 0
    camera_feed.render(mode=moderngl.TRIANGLE_STRIP)

    pygame.display.flip()
    frame_tex.release()


    # limits FPS to 60
    dt = clock.tick(60) / 1000
    t += dt * 0.2

    if export_frames:
        frame_filename = "frames/%04d.png" % file_num
        pygame.image.save(cv_image, frame_filename)
        file_num += 1


capture.release()
