import cv2
import mediapipe as mp
import math
import json
import streamlit as st
from streamlit_lottie import st_lottie
import base64
import time
#load animation
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
      return json.load(f)


lottie_coding=load_lottiefile("lott.json")
st_lottie(
    lottie_coding,
    width=800,
    height=400,
    
          )
#set bakgroudn image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>

    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    [data-testid="stHeader"]{
    background-color:rgba(0,0,0,0)
    }

       
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('backgroundyoga.png')

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose):
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            
    
    # Return the output image and the found landmarks.
    return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def classifyWarriorIIPose(landmarks, output_image):
    '''
    This function classifies Warrior II pose depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: An image of the person with the detected pose landmarks drawn.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if both arms are straight and one leg is straight and the other leg is bended at the required angle.
    if (left_elbow_angle > 165 and left_elbow_angle < 195 and
        right_elbow_angle > 165 and right_elbow_angle < 195 and
        left_shoulder_angle > 80 and left_shoulder_angle < 110 and
        right_shoulder_angle > 80 and right_shoulder_angle < 110 and
        (left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195) and
        (left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120)):

        # Specify the label of the pose that is Warrior II pose.
        label = 'Warrior II Pose'
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Return the output image and the classified label.
    return output_image, label

def classifyTPose(landmarks, output_image):
    '''
    This function classifies T pose depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: An image of the person with the detected pose landmarks drawn.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if both arms are straight and both legs are straight.
    if (left_elbow_angle > 165 and left_elbow_angle < 195 and
        right_elbow_angle > 165 and right_elbow_angle < 195 and
        left_shoulder_angle > 80 and left_shoulder_angle < 110 and
        right_shoulder_angle > 80 and right_shoulder_angle < 110 and
        left_knee_angle > 160 and left_knee_angle < 195 and
        right_knee_angle > 160 and right_knee_angle < 195):

        # Specify the label of the pose that is T pose.
        label = 'T Pose'
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Return the output image and the classified label.
    return output_image, label

def classifyTreePose(landmarks, output_image):
    '''
    This function classifies Tree pose depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: An image of the person with the detected pose landmarks drawn.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the Tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight and the other leg is bended at the required angle.
    if ((left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195) and
        (left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45)):

        # Specify the label of the pose that is Tree pose.
        label = 'Tree Pose'
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Return the output image and the classified label.
    return output_image, label

def classifyWarriorIIIPose(landmarks, output_image):
    '''
    This function classifies Warrior III pose depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: An image of the person with the detected pose landmarks drawn.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, hip and ankle points. 
    left_body_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right shoulder, hip and ankle points. 
    right_body_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Get the angle between the left shoulder, hip and knee points. 
    left_leg_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])

    # Get the angle between the right shoulder, hip and knee points. 
    right_leg_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the Warrior III pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the body is almost parallel to the ground and one leg is raised.
    if ((left_body_angle > 175 and left_body_angle < 185) or
        (right_body_angle > 175 and right_body_angle < 185) and
        (left_leg_angle > 85 and left_leg_angle < 95) and
        (right_leg_angle > 85 and right_leg_angle < 95) and
        (left_knee_angle > 160 and left_knee_angle < 170) and
        (right_knee_angle > 160 and right_knee_angle < 170)):

        # Specify the label of the pose that is Warrior III pose.
        label = 'Warrior III Pose'
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Return the output image and the classified label.
    return output_image, label



def classifyCowPose(landmarks, output_image):
    '''
    This function classifies Cow pose depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: An image of the person with the detected pose landmarks drawn.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the Cow pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if both knees are bent and both arms are straight.
    if (left_knee_angle > 160 and left_knee_angle < 180 and
        right_knee_angle > 160 and right_knee_angle < 180 and
        left_elbow_angle > 170 and left_elbow_angle < 190 and
        right_elbow_angle > 170 and right_elbow_angle < 190):

        # Specify the label of the pose that is Cow pose.
        label = 'Cow Pose'
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Return the output image and the classified label.
    return output_image, label


def classifyPose(landmarks, output_image, selected_pose):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: An image of the person with the detected pose landmarks drawn.
        selected_pose: The selected pose by the user.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
    '''
    
    if selected_pose == "T Pose":
        return classifyTPose(landmarks, output_image)
    elif selected_pose == "Tree Pose":
        return classifyTreePose(landmarks, output_image)
    elif selected_pose=="Warrior II Pose":  # For all other poses including "Unknown Pose"
        return classifyWarriorIIPose(landmarks, output_image)
    elif selected_pose=="Cow Pose":
        return classifyCowPose(landmarks,output_image)
    elif selected_pose=="Warrior III Pose":
        return classifyWarriorIIIPose(landmarks,output_image)
    else:
        return classifyWarriorIIPose(landmarks,output_image)

# Streamlit App

def home():
    st.title("YOGA POSE ESTIMATOR-YOUR PERSONAL AI TRAINER")
    st.write("Welcome to the Yoga Pose Estimation application! Whether you're a beginner or an experienced yogi, our platform offers an innovative way to enhance your practice. Dive in by selecting Let's Start to begin real-time pose detection or explore our tutorials for comprehensive guidance.")
    st.write("Please select an option from below:")
    if st.button("Let's Start"):
        st.session_state.page = 'Select Pose'
    if st.button("Tutorials"):
        st.session_state.page = 'Tutorials'

def select_pose():
    st.title("Select Pose Page")
    st.write("Please select a pose:")
    if st.button("T Pose"):
        st.session_state.selected_pose = "T Pose"
        st.session_state.page = 'Pose Estimation'
    if st.button("Tree Pose"):
        st.session_state.selected_pose = "Tree Pose"
        st.session_state.page = 'Pose Estimation'
    if st.button("Warrior II Pose"):
        st.session_state.selected_pose = "Warrior II Pose"
        st.session_state.page = 'Pose Estimation'
    if  st.button("Cow Pose"):
        st.session_state.selected_pose = "Cow Pose"
        st.session_state.page = 'Pose Estimation' 
    if  st.button("Warrior III Pose"):
        st.session_state.selected_pose = "Warrior III Pose"
        st.session_state.page = 'Pose Estimation'
    if  st.button("Downward Dog"):
        st.session_state.selected_pose = "Downward Dog"
        st.session_state.page = 'Pose Estimation'
    if st.button("Back to Home"):
        st.session_state.page = 'Home'

def pose_estimation():
    st.title("Pose Estimation Page")
    st.write("Selected Pose:", st.session_state.selected_pose)
    st.write("Click 'Run Pose Estimation' to start detecting poses.")

    if st.button("Back", key="back_button"):
        st.session_state.page = 'Select Pose'

    run_pose_estimation = st.checkbox("Run Pose Estimation")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()
    message_placeholder = st.empty()  # Placeholder for the message

    first_run = True  # Flag to display the message only once

    # Initialize variables for timer and best time
    start_time = None
    end_time = None
    best_time = 0  # Initialize with 0

    # Placeholder for timer and best time
    timer_placeholder = st.empty()
    best_time_placeholder = st.empty()

    while run_pose_estimation:
        ret, frame = cap.read()
        if ret:
            # Detect pose
            frame, landmarks = detectPose(frame, pose)
            if landmarks:
                # Classify pose
                frame, pose_label = classifyPose(landmarks, frame, st.session_state.selected_pose)
                
                # Start the timer when the correct pose is detected and displayed
                if pose_label != "Unknown Pose":
                    if start_time is None:
                        start_time = time.time()
                else:
                    # Stop the timer when the pose is no longer detected or an incorrect pose is detected
                    if start_time is not None:
                        end_time = time.time()
                        # Calculate elapsed time
                        elapsed_time = end_time - start_time
                        # Update the best time if the current time is greater
                        best_time = max(best_time, elapsed_time)
                        # Reset start time
                        start_time = None
        
            # Display frame
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            # Display timer
            if start_time is not None:
                elapsed_time = time.time() - start_time
                timer_placeholder.text(f"Timer: {elapsed_time:.2f} sec")
            
            # Display best time
            best_time_placeholder.text(f"Best Time: {best_time:.2f} sec")
    
    # Close video capture
    cap.release()


def tutorials():
    st.markdown(
        "<h1 style='text-align: center;'>Tutorials Page</h1>",
        unsafe_allow_html=True
    )
    
    
    st.write("Check out these resources:")
    st.write("TREE POSE:(https://youtu.be/wdln9qWYloU?si=RmD6UUdUYDRadhed)")
    st.write("T POSE:(https://youtu.be/wEx9dkGBV0Q?si=vQeRN84pVpI3A13m)")
    st.write("WARRIOR II POSE:(https://youtu.be/Mn6RSIRCV3w?si=37ha6zoa2WswH6wM)")
    st.write("COW POSE:(https://youtu.be/W5KVx0ZbB_4?si=WcD5GjrS5bcGeYsX)")
    st.write("WARRIOR III POSE:(https://youtu.be/uEc5hrgIYx4?si=AKRTncEWSZ9pdf4K)")
    st.write("DOWNWARD DOG:(https://youtu.be/j97SSGsnCAQ?si=oKxueD_5q99mr9nB)")

    if st.button("Back to Home"):
        st.session_state.page = 'Home'

def main():
    # Initialize session state variables if they don't exist
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    if st.session_state.page == 'Home':
        home()
    elif st.session_state.page == 'Select Pose':
        select_pose()
    elif st.session_state.page == 'Pose Estimation':
        pose_estimation()
    elif st.session_state.page == 'Tutorials':
        tutorials()


if __name__ == "__main__":
    main()