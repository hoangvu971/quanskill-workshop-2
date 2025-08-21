
import streamlit as st
import pyperclip
from crew import ContentCreationCrew
from datetime import datetime

st.title("Crew AI Marketing Content Generator")

topic = st.text_input("Enter the topic for content creation:")

if st.button("Generate Content"):
    if topic:
        inputs = {
            "topic": topic,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        }

        crew = ContentCreationCrew()
        with st.spinner("Generating content..."):
            result = crew.content_crew().kickoff(inputs=inputs)

        st.header("Results")

        for i, task in enumerate(result.tasks):
            st.subheader(task.description)
            st.write(task.output)
            if st.button(f"Copy output from {task.agent.role}", key=f"copy_button_{i}"):
                pyperclip.copy(task.output.json())
                st.success("Copied to clipboard!")

    else:
        st.error("Please enter a topic.")
