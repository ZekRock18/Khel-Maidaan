import cv2
import streamlit as st
from fer import FER
import time
import mediapipe as mp
import numpy as np
from datetime import datetime
import random
import requests


# Welcome Page
def welcome_page():
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()  # Record the start time when the Home Page is first loaded
    st.title("Khel Maidaan App")
    st.markdown("""<style>
        .container {
            text-align: center;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            max-width: 80%;
            margin: 30px auto;
        }
        .title {
            color: #2c3e50;
            font-size: 32px;
            font-weight: bold;
        }
        .content {
            color: #7f8c8d;
            font-size: 18px;
            line-height: 1.6;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<p class="title">Welcome to Khel Maidaan App!</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">This app helps children develop healthy mobile usage habits with AI. Let’s start a healthier journey!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Registration Page
def registration_page():
    st.title("User Registration")
    with st.form(key="registration_form"):
        time_spent_on_app = round((time.time() - st.session_state.start_time) / 60, 2)  # Time in minutes
        name = st.text_input("Parent Name")
        name = st.text_input("Child Name")
        age = st.number_input("Child Age", min_value=5, max_value=16)
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")
        if submit_button:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                st.session_state.user_data = {"name": name, "age": age, "email": email, "password": password}
                st.success("Registration successful!")

# Login Page
def login_page():
    st.title("User Login")
    if 'user_data' not in st.session_state:
        st.warning("Please register first!")
        return
    with st.form(key="login_form"):
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            user_data = st.session_state.user_data
            if email == user_data["email"] and password == user_data["password"]:
                st.success(f"Welcome back, {user_data['name']}!")
                st.session_state.logged_in = True
            else:
                st.error("Invalid email or password.")

# Emotion Detection Page with Live Video
def emotion_detection_page():
    st.title("Face Emotion Recognition")
    st.write("This feature uses your webcam to detect your facial emotions. Please allow webcam access.")
    
    # Initialize FER detector
    detector = FER()

    # Video placeholder in Streamlit
    video_placeholder = st.empty()
    emotion_text_placeholder = st.empty()

    # Timer to run for 10 seconds
    start_time = time.time()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        # Convert frame to RGB for FER
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion, score = detector.top_emotion(rgb_frame)

        # Update the video and emotion text in Streamlit
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)  # Use container width for images

        # Display the detected emotion with a friendly message
        if emotion:
            if emotion == "happy":
                message = "You seem happy! 😊"
                youtube_url = "https://youtu.be/92xTPH7OtLs?si=rr8cZdffB2Zn6AoA"  # Example YouTube URL for happy yoga
            elif emotion == "sad":
                message = "You seem sad, is everything okay? 😢"
                youtube_url = "https://youtu.be/92xTPH7OtLs?si=rr8cZdffB2Zn6AoA"  # Example YouTube URL for sad yoga
            elif emotion == "neutral":
                message = "You have a neutral expression."
                youtube_url = "https://youtu.be/92xTPH7OtLs?si=rr8cZdffB2Zn6AoA"  # Example YouTube URL for neutral yoga
            else:
                message = "You have a neutral expression."
                youtube_url = None

            emotion_text_placeholder.write(f"Detected Emotion: **{emotion.capitalize()}** - {message}")
            
            if youtube_url:
                st.write("Here is a recommended Yoga video for you!")
                st.video(youtube_url)  # Embed YouTube video

        else:
            emotion_text_placeholder.write("No clear emotion detected. Please try again.")

    # Release the camera after 10 seconds
    cap.release()
    cv2.destroyAllWindows()

# Hand Drawing Game for Kids (Age 5-7)
def hand_game_page():
    st.title("Drawing Game for Kids (5-7 years)")
    st.write("Please draw shapes with your finger. We will try to recognize the shape you draw.")
    
    # Initialize Mediapipe for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Canvas to draw on
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Drawing mode control using Streamlit button
    drawing = False
    drawing_button = st.button("Start Drawing")
    if drawing_button:
        drawing = not drawing
    
    points = []

    # Function to detect shapes based on contours
    def detect_shape(cnt):
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            return "Triangle"
        elif len(approx) == 4:
            # Check if it is a square or rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "Square"
            else:
                return "Rectangle"
        elif len(approx) > 4:
            return "Circle"
        return "Unknown"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        
        # Flip the frame horizontally to create a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand landmarks
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get the fingertip (index finger tip)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Start drawing when the finger is detected
                if drawing:
                    points.append((cx, cy))
                    for i in range(1, len(points)):
                        cv2.line(canvas, points[i-1], points[i], (255, 255, 255), 3)
                        
                # Draw fingertip position
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        
        # Detect shapes based on contours in the canvas
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_canvas, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            shape = detect_shape(cnt)
            if shape != "Unknown":
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)  # Draw contour in red
                cv2.putText(frame, shape, (cnt[0][0][0], cnt[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
        # Add drawing mode toggle
        if drawing:
            st.write("Drawing mode is active! Try drawing shapes like circles, squares, or triangles.")
        else:
            st.write("Drawing mode is inactive. Press the button to start drawing.")
        
        video_placeholder.image(frame, channels="BGR", use_container_width=True)  # Display frame
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Mental Math Game Page
def mental_math_game_page():

    st.title("Interactive Mental Math Game")

    # Fixed set of 5 questions
    questions = [
        {"question": "5 + 3", "answer": 8},
        {"question": "12 - 7", "answer": 5},
        {"question": "8 * 6", "answer": 48},
        {"question": "18 / 3", "answer": 6},
        {"question": "15 + 2", "answer": 17}
    ]
    
    correct_answers = 0

    # Ask the questions one by one
    for i, q in enumerate(questions):
        st.subheader(f"Question {i + 1}: {q['question']}")
        user_answer = st.text_input("Your answer:", key=f"answer_{i}")

        # Check the answer when the user submits
        if user_answer:
            try:
                if float(user_answer) == q["answer"]:
                    st.success("Correct! 🎉")
                    correct_answers += 1
                else:
                    st.error(f"Wrong! The correct answer is {q['answer']}.")
            except ValueError:
                st.error("Please enter a valid number.")

    # Show the score after all questions
    if correct_answers > 0:
        st.markdown(f"**You answered {correct_answers}/{len(questions)} questions correctly!**")
    else:
        st.markdown("**You didn't get any correct answers. Better luck next time!**")

# Simple emotion detection for demo purposes
def detect_emotion(text):
    emotions = {
        "happy": ["happy", "joy", "excited", "good", "great"],
        "sad": ["sad", "depressed", "down", "blue", "low"],
        "angry": ["angry", "frustrated", "mad", "upset", "rage"],
        "neutral": ["okay", "fine", "alright", "neutral", "calm"]
    }
    
    text = text.lower()
    for emotion, keywords in emotions.items():
        if any(keyword in text for keyword in keywords):
            return emotion
    return "neutral"  # Default if no emotion found

# Simple chatbot response generator
def chatbot_response(emotion):
    responses = {
        "happy": "That's great to hear! Keep up the positivity!",
        "sad": "I'm sorry you're feeling this way. I'm here if you need to talk.",
        "angry": "I understand you're upset. Would you like to discuss what's bothering you?",
        "neutral": "I see you're calm. Let me know if you'd like to talk."
    }
    return responses.get(emotion, "How can I assist you today?")

# Chatbot Page
def chatbot_page():
    st.title("Chat with the Bot")

    # User input for chatting
    user_message = st.text_input("How are you feeling today?", "")

    if user_message:
        # Detect emotion from user's message
        emotion = detect_emotion(user_message)
        response = chatbot_response(emotion)
        
        # Display user message and bot response
        st.write(f"**You:** {user_message}")
        st.write(f"**Bot:** {response}")
    
    else:
        st.write("Please share your feelings or thoughts, and I'll try to understand.")
     
# Digital Wellbeing Page
def digital_wellbeing_page():
    st.title("Digital Wellbeing")

    # Calculate time spent on the app from the Home Page
    time_spent_on_app = round((time.time() - st.session_state.start_time) / 60, 2)  # Time in minutes
    st.write(f"**Time spent on the app so far:** {time_spent_on_app} minutes")

    # Form to record child's activity
    with st.form(key='digital_wellbeing_form'):
        # Child's Name (Assuming you want to auto-fill this based on the registration)
        child_name = st.text_input("Record Name", placeholder="Enter the record name")

        # Activity Selection
        activity = st.selectbox("What was your child doing? (apart from the app)", ["Playing a Game", "Reading", "Watching TV", "Learning", "Other"])

        # Time Spent Input (in minutes)
        time_spent = st.number_input("Time spent (in minutes)", min_value=0, step=1)

        # Additional Comments (Optional)
        comments = st.text_area("Additional Comments", placeholder="Any other details...")

        # Submit button
        submit_button = st.form_submit_button(label='Submit Activity Record')

        if submit_button:
            if child_name and time_spent > 0:
                # Display entered information
                st.write(f"**Child's Name:** {child_name}")
                st.write(f"**Activity:** {activity}")
                st.write(f"**Time Spent:** {time_spent} minutes")
                st.write(f"**Time spent on the app:** {time_spent_on_app} minutes")
                if comments:
                    st.write(f"**Comments:** {comments}")
                st.success("Activity record submitted successfully!")
            else:
                st.error("Please provide the child's name and time spent.")     


# Main function to render the pages based on user choice
def main():
    st.set_page_config(page_title="Khel Maidaan App", page_icon="🧒", layout="wide")
    
    # Create session state variable to track user login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Create navigation
    menu = ["Home", "Login", "Register", "Emotion Detection", "Hand Drawing Game", "Mental Math Game", "Chatbot", "Digital Wellbeing"]
    choice = st.sidebar.selectbox("Select an option", menu)
    
    if choice == "Home":
        welcome_page()
    elif choice == "Login":
        if st.session_state.logged_in:
            st.write(f"Welcome back, {st.session_state.user_data['name']}!")
        else:
            login_page()
    elif choice == "Register":
        registration_page()
    elif choice == "Emotion Detection":
        if st.session_state.logged_in:
            emotion_detection_page()
        else:
            st.warning("Please log in first to access this feature.")
    elif choice == "Hand Drawing Game":
        if st.session_state.logged_in:
            hand_game_page()
        else:
            st.warning("Please log in first to access this feature.")

    elif choice == "Mental Math Game":
        if st.session_state.logged_in:
            mental_math_game_page()
        else:
            st.warning("Please log in first to access this feature.")  

    elif choice == "Chatbot":
        if st.session_state.logged_in:
            chatbot_page()
        else:
            st.warning("Please log in first to access this feature.")      

    elif choice == "Digital Wellbeing":
        digital_wellbeing_page()                

if __name__ == "__main__":
    main()
