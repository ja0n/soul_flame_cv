import cv2
import mediapipe as mp


class FaceMeshDetector:

    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_con=0.5,
        min_tracking_con=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_con = (
            min_detection_con
        )
        self.min_tracking_con = min_tracking_con

        # Initialize Mediapipe FaceMesh solution
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            self.refine_landmarks,
            self.min_detection_con,
            self.min_tracking_con,
        )

        # Store the landmark indices for specific facial features
        # These are predefined Mediapipe indices for left and right eyes, iris, nose, and mouth

        self.LEFT_EYE_LANDMARKS = [
            463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362
        ]

        self.RIGHT_EYE_LANDMARKS = [
            33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
        ]

        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]

        self.NOSE_LANDMARKS = [
            193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274
        ]

        self.MOUTH_LANDMARKS = [
            0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37
        ]

    def find_mesh_in_face(self, img):
        # Initialize a dictionary to store the landmarks for facial features
        landmarks = {}

        # Convert the input image to RGB as Mediapipe expects RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find face landmarks using the FaceMesh model
        resullt = self.faceMesh.process(imgRGB)

        # Check if any faces were detected
        if resullt.multi_face_landmarks:
            # Iterate over detected faces (here, max_num_faces = 1, so usually one face)
            for result_landmarks in resullt.multi_face_landmarks:
                # Initialize lists in the landmarks dictionary to store each facial feature's coordinates
                landmarks["left_eye"] = []
                landmarks["right_eye"] = []
                landmarks["left_iris"] = []
                landmarks["right_iris"] = []
                landmarks["nose"] = []
                landmarks["mouth"] = []
                landmarks["all"] = []

                # Loop through all face landmarks
                for i, lm in enumerate(result_landmarks.landmark):
                    h, w, ic = img.shape  # Get image height, width, and channel count
                    x, y = (
                        int(lm.x * w),
                        int(lm.y * h),
                    )  # Convert normalized coordinates to pixel values

                    # Store the coordinates of all landmarks
                    landmarks["all"].append((x, y))

                    # Store specific feature landmarks based on the predefined indices
                    if i in self.LEFT_EYE_LANDMARKS:
                        landmarks["left_eye"].append((x, y))
                    if i in self.RIGHT_EYE_LANDMARKS:
                        landmarks["right_eye"].append((x, y))
                    if i in self.LEFT_IRIS_LANDMARKS:
                        landmarks["left_iris"].append((x, y))
                    if i in self.RIGHT_IRIS_LANDMARKS:
                        landmarks["right_iris"].append((x, y))
                    if i in self.NOSE_LANDMARKS:
                        landmarks["nose"].append((x, y))
                    if i in self.MOUTH_LANDMARKS:
                        landmarks["mouth"].append((x, y))

        # Return the processed image and the dictionary of feature landmarks
        return img, landmarks
