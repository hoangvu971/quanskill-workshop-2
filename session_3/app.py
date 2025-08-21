from crewai.utilities.events.base_event_listener import BaseEventListener
import os
import mimetypes
import base64
from crewai.utilities.events import (
    crewai_event_bus,
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
)
import streamlit as st
import json
import threading
import time
from datetime import datetime, date
from typing import Dict, Any, Optional
from queue import Queue, Empty
import uuid

# Import your CrewAI classes
from crew import TheMarketingCrew
from crewai import Agent, Task, Crew

# Configure page
st.set_page_config(
    page_title="CrewAI Marketing Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def to_data_uri(path: str):
    if not path or not os.path.exists(path):
        return None
    mt, _ = mimetypes.guess_type(path)
    mt = mt or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mt};base64,{b64}"


logo_path = "assets/logo.png"
logo = to_data_uri(logo_path)


def set_gemini_api_key(api_key):
    """
    Set the Gemini API key as an environment variable

    Args:
        api_key (str): The API key to set

    Returns:
        bool: True if API key was set successfully, False otherwise
    """
    if api_key and api_key.strip():
        os.environ["GEMINI_API_KEY"] = api_key.strip()
        return True
    return False


def get_gemini_api_key():
    """
    Get the Gemini API key from environment variables

    Returns:
        str: The API key if found, empty string otherwise
    """
    return os.environ.get("GEMINI_API_KEY", "")


# Modern UI theme CSS (same as before)
st.markdown(
    """
<style>
#MainMenu, footer {visibility:hidden;}
/* unify all top bars with light bg */
[data-testid="stHeader"], [data-testid="stDecoration"] { background: #f6f7fb !important; }
:root{
  --bg:#f6f7fb; --surface:#ffffff; --ink:#0f172a; --muted:#5b6375;
  --border:#e6e8ef; --accent:#ff8a1f; --accent-ink:#1a1a1a;
  --success-bg:#eaf7ee; --success-bd:#cde8d6; --success-txt:#0f5132;
  --dark:#0b0f15; --dark-bd:#1f2937; --dark-ink:#e5e7eb;
  --warning-bg:#fff3cd; --warning-bd:#ffeaa7; --warning-txt:#856404;
  --info-bg:#d1ecf1; --info-bd:#bee5eb; --info-txt:#0c5460;
}
.stApp { background: var(--bg); }

/* Sidebar styling */
section[data-testid="stSidebar"]{
  background:#fff;
  border-right:1px solid var(--border);
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stMarkdown{
  color: var(--ink) !important;
}

/* Main content styling */
[data-testid="stAppViewContainer"]{
  color: var(--ink) !important;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stMarkdown{
  color: var(--ink) !important;
}

/* Brand area */
.brand-header{ 
  padding:20px 0; 
  border-bottom:1px solid var(--border); 
  margin-bottom:20px;
}

/* Cards & KPI */
.card{ 
  background:var(--surface); 
  border:1px solid var(--border); 
  border-radius:16px;
  box-shadow:0 6px 24px rgba(15,23,42,.06); 
  padding:20px; 
  margin-bottom:20px;
}
.agent-card{
  background:var(--surface); 
  border:1px solid var(--border); 
  border-radius:16px;
  box-shadow:0 6px 24px rgba(15,23,42,.06); 
  padding:24px; 
  margin-bottom:24px;
}
.agent-header{
  display:flex; 
  align-items:center; 
  gap:12px; 
  margin-bottom:16px;
}
.agent-icon{
  width:40px; 
  height:40px; 
  border-radius:50%; 
  background:var(--accent); 
  display:flex; 
  align-items:center; 
  justify-content:center; 
  color:white; 
  font-weight:bold;
  font-size:18px;
}
.agent-title{
  font-size:1.4rem; 
  font-weight:800; 
  color:var(--ink);
}
.agent-status{
  padding:6px 12px; 
  border-radius:20px; 
  font-size:0.8rem; 
  font-weight:600;
}
.status-pending{ background:var(--warning-bg); border:1px solid var(--warning-bd); color:var(--warning-txt); }
.status-running{ background:var(--info-bg); border:1px solid var(--info-bd); color:var(--info-txt); }
.status-complete{ background:var(--success-bg); border:1px solid var(--success-bd); color:var(--success-txt); }
.status-error{ background:#fee; border:1px solid #fcc; color:#c33; }

/* Tabs styling */
div[data-baseweb="tab-list"] { gap:8px !important; }
button[role="tab"]{
  background:var(--surface) !important; 
  border:1px solid var(--border) !important;
  color:var(--muted) !important; 
  padding:12px 20px !important; 
  border-radius:16px !important;
  box-shadow:0 4px 14px rgba(15,23,42,.04);
  font-weight:600 !important;
}
button[role="tab"][aria-selected="true"]{
  background: linear-gradient(180deg,#fff,#f9fafc) !important;
  color:var(--ink) !important; 
  border-color:var(--accent) !important; 
  font-weight:800 !important;
}

/* Buttons */
.stButton>button{
  background:var(--accent); 
  color:var(--accent-ink); 
  border:0; 
  font-weight:700;
  border-radius:12px; 
  padding:12px 24px;
  transition: all 0.2s ease;
}
.stButton>button:hover{ 
  filter:brightness(1.05); 
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(255,138,31,.3);
}

/* Progress bar */
.progress-container{
  background:var(--border); 
  border-radius:8px; 
  height:8px; 
  overflow:hidden; 
  margin:16px 0;
}
.progress-bar{
  background:var(--accent); 
  height:100%; 
  transition: width 0.3s ease;
}

/* Output sections */
.output-section{
  background:var(--dark); 
  color:var(--dark-ink) !important;
  border:1px solid var(--dark-bd); 
  border-radius:14px; 
  padding:16px 20px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  white-space:pre-wrap;
  margin-top:12px;
  max-height: 400px;
  overflow-y: auto;
}
.output-section *{ color:var(--dark-ink) !important; }

/* Input styling */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[role="combobox"],
.stNumberInput input{
  border:1px solid var(--border) !important;
  border-radius:8px !important;
  background:var(--surface) !important;
  color:var(--ink) !important;
}

/* Live log styling */
.live-log {
  background: var(--dark);
  color: var(--dark-ink);
  border: 1px solid var(--dark-bd);
  border-radius: 8px;
  padding: 12px;
  font-family: monospace;
  font-size: 0.9rem;
  max-height: 300px;
  overflow-y: auto;
  margin: 10px 0;
}

.log-entry {
  margin: 4px 0;
  padding: 2px 0;
}

.log-timestamp {
  color: #888;
  font-size: 0.8rem;
}

.log-agent { color: #4ade80; }
.log-task { color: #60a5fa; }
.log-error { color: #f87171; }
.log-info { color: #a78bfa; }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitCrewEventListener(BaseEventListener):
    """Event listener for CrewAI events that sends updates to Streamlit UI"""

    def __init__(self, event_queue: Queue):
        super().__init__()
        self.event_queue = event_queue

    def setup_listeners(self, crewai_event_bus):
        """Setup event listeners according to CrewAI documentation"""

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            self.event_queue.put(
                {
                    "type": "crew_started",
                    "crew_name": getattr(event, "crew_name", "Marketing Crew"),
                    "message": f"🚀 Crew '{getattr(event, 'crew_name', 'Marketing Crew')}' has started execution!",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            self.event_queue.put(
                {
                    "type": "crew_complete",
                    "crew_name": getattr(event, "crew_name", "Marketing Crew"),
                    "output": getattr(event, "output", "Completed successfully"),
                    "message": f"✅ Crew '{getattr(event, 'crew_name', 'Marketing Crew')}' has completed execution!",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event):
            self.event_queue.put(
                {
                    "type": "crew_error",
                    "crew_name": getattr(event, "crew_name", "Marketing Crew"),
                    "error": getattr(event, "error", "Unknown error"),
                    "message": f"❌ Crew '{getattr(event, 'crew_name', 'Marketing Crew')}' failed to complete execution",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event):
            agent_role = (
                getattr(event.agent, "role", "Unknown Agent")
                if hasattr(event, "agent")
                else "Unknown Agent"
            )
            self.event_queue.put(
                {
                    "type": "agent_start",
                    "agent_name": agent_role,
                    "message": f"🤖 Agent '{agent_role}' started execution",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event):
            agent_role = (
                getattr(event.agent, "role", "Unknown Agent")
                if hasattr(event, "agent")
                else "Unknown Agent"
            )
            output = getattr(event, "output", "")
            self.event_queue.put(
                {
                    "type": "agent_finish",
                    "agent_name": agent_role,
                    "output": str(output)[:200] + "..."
                    if len(str(output)) > 200
                    else str(output),
                    "message": f"✅ Agent '{agent_role}' completed execution",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def on_agent_error(source, event):
            agent_role = (
                getattr(event.agent, "role", "Unknown Agent")
                if hasattr(event, "agent")
                else "Unknown Agent"
            )
            error = getattr(event, "error", "Unknown error")
            self.event_queue.put(
                {
                    "type": "agent_error",
                    "agent_name": agent_role,
                    "error": str(error),
                    "message": f"❌ Agent '{agent_role}' encountered an error",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source, event):
            task_desc = (
                getattr(event, "description", "Unknown task")
                if hasattr(event, "description")
                else "Task started"
            )
            self.event_queue.put(
                {
                    "type": "task_start",
                    "task_description": task_desc[:100] + "..."
                    if len(task_desc) > 100
                    else task_desc,
                    "message": f"📋 Task started: {task_desc[:50]}{'...' if len(task_desc) > 50 else ''}",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event):
            task_desc = (
                getattr(event, "description", "Unknown task")
                if hasattr(event, "description")
                else "Task completed"
            )
            self.event_queue.put(
                {
                    "type": "task_complete",
                    "task_description": task_desc[:100] + "..."
                    if len(task_desc) > 100
                    else task_desc,
                    "message": f"✅ Task completed: {task_desc[:50]}{'...' if len(task_desc) > 50 else ''}",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event):
            task_desc = (
                getattr(event, "description", "Unknown task")
                if hasattr(event, "description")
                else "Task failed"
            )
            error = getattr(event, "error", "Unknown error")
            self.event_queue.put(
                {
                    "type": "task_error",
                    "task_description": task_desc[:100] + "..."
                    if len(task_desc) > 100
                    else task_desc,
                    "error": str(error),
                    "message": f"❌ Task failed: {task_desc[:50]}{'...' if len(task_desc) > 50 else ''}",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event):
            tool_name = (
                getattr(event, "tool_name", "Unknown tool")
                if hasattr(event, "tool_name")
                else "Tool"
            )
            self.event_queue.put(
                {
                    "type": "tool_start",
                    "tool_name": tool_name,
                    "message": f"🔧 Tool '{tool_name}' started",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event):
            tool_name = (
                getattr(event, "tool_name", "Unknown tool")
                if hasattr(event, "tool_name")
                else "Tool"
            )
            self.event_queue.put(
                {
                    "type": "tool_finish",
                    "tool_name": tool_name,
                    "message": f"🔧 Tool '{tool_name}' completed",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event):
            tool_name = (
                getattr(event, "tool_name", "Unknown tool")
                if hasattr(event, "tool_name")
                else "Tool"
            )
            error = getattr(event, "error", "Unknown error")
            self.event_queue.put(
                {
                    "type": "tool_error",
                    "tool_name": tool_name,
                    "error": str(error),
                    "message": f"🔧 Tool '{tool_name}' encountered an error",
                    "timestamp": event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else datetime.now().isoformat(),
                }
            )


def run_crew_in_background(
    inputs: Dict[str, Any], event_queue: Queue, result_queue: Queue
):
    """Run CrewAI in background thread with proper event listening"""
    # Global event listener instance (required for proper registration)
    global streamlit_listener

    try:
        # Create and register event listener
        streamlit_listener = StreamlitCrewEventListener(event_queue)

        # Initialize crew
        crew_instance = TheMarketingCrew()
        crew = crew_instance.marketingcrew()

        # Start execution log
        event_queue.put(
            {
                "type": "info",
                "message": "🚀 Starting CrewAI execution...",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Execute crew (events will be automatically captured)
        result = crew.kickoff(inputs=inputs)

        # Put final result
        result_data = result
        if hasattr(result, "raw"):
            result_data = result.raw
        elif hasattr(result, "dict"):
            result_data = result.dict()
        else:
            result_data = str(result)

        result_queue.put(
            {
                "success": True,
                "result": result_data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        # Put error result
        result_queue.put(
            {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
        )

        event_queue.put(
            {
                "type": "error",
                "error": str(e),
                "message": f"❌ Execution failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }
        )


def create_agent_card(
    agent_name: str, agent_icon: str, status: str = "pending", content: str = None
):
    """Create a styled agent card"""
    status_class = f"status-{status.lower()}"
    status_text = {
        "pending": "Pending",
        "running": "Running...",
        "complete": "Complete",
        "error": "Error",
    }.get(status.lower(), "Unknown")

    card_html = f"""
    <div class="agent-card">
        <div class="agent-header">
            <div class="agent-icon">{agent_icon}</div>
            <div class="agent-title">{agent_name}</div>
            <div class="agent-status {status_class}">{status_text}</div>
        </div>
    """

    if content:
        card_html += f"""
        <div class="output-section">{content}</div>
        """

    card_html += "</div>"
    return card_html


def create_progress_bar(current: int, total: int):
    """Create a progress bar"""
    percentage = (current / total) * 100 if total > 0 else 0
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {percentage}%"></div>
    </div>
    <p style="text-align:center; color:var(--muted); margin-top:8px;">
        Step {current} of {total} ({percentage:.0f}% complete)
    </p>
    """


def format_log_entry(event):
    """Format a log entry for display"""
    timestamp = event.get("timestamp", "")
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%H:%M:%S")
        except:
            formatted_time = timestamp[:8]
    else:
        formatted_time = datetime.now().strftime("%H:%M:%S")

    event_type = event.get("type", "info")

    if event_type == "agent_start":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-agent">🤖 {event.get("agent_name", "Agent")}</span> started execution</div>'
    elif event_type == "agent_finish":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-agent">✅ {event.get("agent_name", "Agent")}</span> completed execution</div>'
    elif event_type == "agent_error":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-error">❌ {event.get("agent_name", "Agent")} error:</span> {event.get("error", "Unknown error")}</div>'
    elif event_type == "task_start":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-task">📋 Task started:</span> {event.get("task_description", "")}</div>'
    elif event_type == "task_complete":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-task">✅ Task completed:</span> {event.get("task_description", "")}</div>'
    elif event_type == "task_error":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-error">❌ Task failed:</span> {event.get("task_description", "")}</div>'
    elif event_type == "tool_start":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-info">🔧 Tool started:</span> {event.get("tool_name", "Unknown tool")}</div>'
    elif event_type == "tool_finish":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-info">🔧 Tool completed:</span> {event.get("tool_name", "Unknown tool")}</div>'
    elif event_type == "tool_error":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-error">🔧 Tool error:</span> {event.get("tool_name", "Unknown tool")}</div>'
    elif event_type == "crew_started":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-info">🚀 Crew started:</span> {event.get("crew_name", "Marketing Crew")}</div>'
    elif event_type == "crew_complete":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-info">🎉 Crew completed:</span> {event.get("crew_name", "Marketing Crew")}</div>'
    elif event_type == "crew_error":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-error">❌ Crew failed:</span> {event.get("crew_name", "Marketing Crew")}</div>'
    elif event_type == "error":
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-error">❌ Error:</span> {event.get("error", "Unknown error")}</div>'
    else:
        message = event.get("message", str(event))
        return f'<div class="log-entry"><span class="log-timestamp">[{formatted_time}]</span> <span class="log-info">ℹ️</span> {message}</div>'


# Global variable to keep event listener in memory (required by CrewAI)
streamlit_listener = None

# Initialize session state
if "crew_results" not in st.session_state:
    st.session_state.crew_results = {}
if "execution_status" not in st.session_state:
    st.session_state.execution_status = {}
if "crew_running" not in st.session_state:
    st.session_state.crew_running = False
if "crew_thread" not in st.session_state:
    st.session_state.crew_thread = None
if "event_queue" not in st.session_state:
    st.session_state.event_queue = Queue()
if "result_queue" not in st.session_state:
    st.session_state.result_queue = Queue()
if "live_logs" not in st.session_state:
    st.session_state.live_logs = []
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None

# Header
st.markdown(
    f"""
<div class="brand-header">
    <h1>🚀 CrewAI Marketing Dashboard</h1>
    <p style="color:var(--muted); font-size:1.1rem;">AI-Powered Marketing Strategy & Content Creation</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar - Input Configuration
with st.sidebar:
    if logo:  # assuming 'logo' is your image path variable
        st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
        st.image(logo, width=200)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="brand-card">
          <div class="brand-logos"
            {f"<img src='{logo}'>" if logo else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")  # Add a separator line
    st.markdown("### API Configuration")

    # Get current API key if it exists
    current_api_key = get_gemini_api_key()

    # API Key input with placeholder
    api_key_input = st.text_input(
        "Gemini API Key",
        value=current_api_key if current_api_key else "",
        type="password",
        placeholder="Enter your Gemini API key here...",
        help="Your API key will be stored securely for this session",
    )

    # Set API key when user inputs it
    if api_key_input:
        if set_gemini_api_key(api_key_input):
            if not current_api_key:  # Only show success message when first setting
                st.success("API key set successfully!")
        else:
            st.error("Please enter a valid API key")

    st.markdown("## Campaign Configuration")
    # Basic Information
    st.markdown("### Product Information")
    product_name = st.text_input("Product Name", value="Premium Anti-Aging Serum")
    product_description = st.text_area(
        "Product Description",
        value="A revolutionary anti-aging serum with natural ingredients that reduces fine lines and wrinkles, giving you youthful, radiant skin.",
        height=100,
    )
    industry = st.selectbox(
        "Industry",
        [
            "Beauty & Cosmetics",
            "Healthcare",
            "Finance",
            "Education",
            "Retail",
            "Manufacturing",
            "Other",
        ],
    )
    if industry == "Other":
        custom_industry = st.text_input("Please specify industry:")

    # Target Audience
    st.markdown("### Target Audience")
    target_audience = st.text_input(
        "Target Audience", value="Women aged 25-45 interested in skincare"
    )

    # Location
    st.markdown("### Location")
    location = st.text_input("Target Location", value="Ho Chi Minh City, Vietnam")

    # Campaign Details
    st.markdown("### Campaign Settings")
    budget = st.text_input("Budget", value="VND 10,000,000")
    campaign_duration = st.selectbox(
        "Campaign Duration",
        ["1 week", "1 month", "3 months", "6 months", "12 months", "Other"],
    )
    if campaign_duration == "Other":
        custom_duration = st.text_input("Please specify duration:")

    primary_goal = st.selectbox(
        "Primary Goal",
        [
            "Lead generation and brand awareness",
            "Brand awareness",
            "Lead generation",
            "Sales conversion",
            "Customer retention",
            "Other",
        ],
    )
    if primary_goal == "Other":
        custom_goal = st.text_input("Please specify primary goal:")

    # Advanced Settings
    with st.expander("Advanced Settings"):
        current_date = st.date_input("Campaign Start Date", value=date.today())
        custom_requirements = st.text_area(
            "Additional Requirements",
            placeholder="Any specific requirements or notes...",
        )
        # Auto-refresh settings
        auto_refresh = st.checkbox("Auto-refresh logs", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 3)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 🎯 Campaign Overview")

    # Display current configuration
    config_card = f"""
    <div class="card">
        <h3>Configuration Summary</h3>
        <p><strong>Product:</strong> {product_name}</p>
        <p><strong>Industry:</strong> {industry}</p>
        <p><strong>Target:</strong> {target_audience}</p>
        <p><strong>Budget:</strong> {budget}</p>
        <p><strong>Duration:</strong> {campaign_duration}</p>
        <p><strong>Goal:</strong> {primary_goal}</p>
    </div>
    """
    st.markdown(config_card, unsafe_allow_html=True)

with col2:
    st.markdown("## 🚀 Execution Controls")

    # Execution button
    if not st.session_state.crew_running:
        if st.button(
            "🚀 Start Marketing Crew", use_container_width=True, type="primary"
        ):
            # Prepare inputs
            inputs = {
                "product_name": product_name,
                "target_audience": target_audience,
                "product_description": product_description,
                "budget": budget,
                "current_date": current_date.strftime("%Y-%m-%d"),
                "industry": industry,
                "campaign_duration": campaign_duration,
                "primary_goal": primary_goal,
                "location": location,
            }

            if custom_requirements:
                inputs["additional_requirements"] = custom_requirements

            # Clear previous results
            st.session_state.crew_results = {}
            st.session_state.execution_status = {}
            st.session_state.live_logs = []
            st.session_state.execution_id = str(uuid.uuid4())

            # Create new queues
            st.session_state.event_queue = Queue()
            st.session_state.result_queue = Queue()

            # Start crew in background thread
            st.session_state.crew_thread = threading.Thread(
                target=run_crew_in_background,
                args=(
                    inputs,
                    st.session_state.event_queue,
                    st.session_state.result_queue,
                ),
                daemon=True,
            )
            st.session_state.crew_thread.start()
            st.session_state.crew_running = True
            st.rerun()
    else:
        st.button("⏳ Crew Running...", disabled=True, use_container_width=True)
        if st.button("🛑 Stop Execution", use_container_width=True):
            st.session_state.crew_running = False
            # Note: In a production app, you'd want to implement proper thread cancellation
            st.rerun()

# Live Logs Section
if st.session_state.crew_running or st.session_state.live_logs:
    st.markdown("## 📝 Live Execution Logs")

    # Process new events from queue
    new_events = []
    try:
        while True:
            event = st.session_state.event_queue.get_nowait()
            new_events.append(event)
            st.session_state.live_logs.append(event)
    except Empty:
        pass

    # Check for final result
    try:
        final_result = st.session_state.result_queue.get_nowait()
        st.session_state.crew_results = final_result
        st.session_state.crew_running = False
        if new_events or final_result:
            st.rerun()
    except Empty:
        pass

    # Display logs
    if st.session_state.live_logs:
        log_html = '<div class="live-log">'
        for event in st.session_state.live_logs[-50:]:  # Show last 50 events
            log_html += format_log_entry(event)
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh and st.session_state.crew_running:
        time.sleep(refresh_interval)
        st.rerun()

# Define agents and their details
agents_info = [
    ("Market Research Agent", "📊", "market_research_task"),
    ("Marketing Strategy Agent", "🎯", "marketing_strategy_task"),
    ("Content Calendar Agent", "📅", "content_calendar_task"),
    ("Content Writer Agent", "✍️", "content_drafting_blogs_task"),
    ("Social Content Agent", "📱", "content_drafting_social_task"),
    ("SEO Specialist Agent", "🔍", "seo_optimization_task"),
    ("Social Script Agent", "🎬", "script_generation_task"),
]

# Results Section
st.markdown("## 📊 Agent Results")

# Display agent cards with updated status from logs
for agent_name, agent_icon, task_key in agents_info:
    # Determine status based on logs and results
    status = "pending"
    content = None

    # Check logs for this agent
    for log in st.session_state.live_logs:
        if (
            log.get("type") == "agent_start"
            and agent_name.lower() in log.get("agent_name", "").lower()
        ):
            status = "running"
        elif (
            log.get("type") == "agent_finish"
            and agent_name.lower() in log.get("agent_name", "").lower()
        ):
            status = "complete"
            content = (
                log.get("result", "")[:500] + "..."
                if len(log.get("result", "")) > 500
                else log.get("result", "")
            )

    # Check if crew is complete
    if st.session_state.crew_results and st.session_state.crew_results.get("success"):
        status = "complete"
        result_data = st.session_state.crew_results.get("result", {})
        if isinstance(result_data, dict) and task_key in str(result_data):
            content = json.dumps(result_data, indent=2)[:500] + "..."
        elif isinstance(result_data, str):
            content = (
                result_data[:500] + "..." if len(result_data) > 500 else result_data
            )

    # Check for errors
    for log in st.session_state.live_logs:
        if log.get("type") == "error":
            status = "error"
            content = log.get("error", "")
            break

    # Create and display card
    # card_html = create_agent_card(agent_name, agent_icon, status, content)
    # st.markdown(card_html, unsafe_allow_html=True)

# Results Summary
if st.session_state.crew_results:
    st.markdown("## 📋 Final Results Summary")

    if st.session_state.crew_results.get("success"):
        tabs = st.tabs(["📊 Raw Output", "📁 Structured Data", "📈 Analysis"])

        with tabs[0]:
            st.markdown("### Complete Crew Output")
            result_data = st.session_state.crew_results.get("result", {})
            st.code(
                json.dumps(result_data, indent=2)
                if isinstance(result_data, dict)
                else str(result_data),
                language="json",
            )

        with tabs[1]:
            st.markdown("### Structured Results")
            result_data = st.session_state.crew_results.get("result", {})
            if isinstance(result_data, dict):
                for key, value in result_data.items():
                    st.markdown(f"**{key.replace('_', ' ').title()}:**")
                    if isinstance(value, (dict, list)):
                        st.json(value)
                    else:
                        st.write(value)
            else:
                st.write(result_data)

        with tabs[2]:
            st.markdown("### Performance Analysis")
            execution_time = "N/A"
            if st.session_state.live_logs:
                start_time = None
                end_time = None
                for log in st.session_state.live_logs:
                    if log.get("type") == "info" and "Starting" in log.get(
                        "message", ""
                    ):
                        start_time = log.get("timestamp")
                    elif log.get("type") == "crew_complete":
                        end_time = log.get("timestamp")

                if start_time and end_time:
                    try:
                        start_dt = datetime.fromisoformat(
                            start_time.replace("Z", "+00:00")
                        )
                        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                        duration = (end_dt - start_dt).total_seconds()
                        execution_time = f"{duration:.2f} seconds"
                    except:
                        pass

            st.metric("Execution Time", execution_time)
            st.metric("Total Agents", len(agents_info))
            st.metric("Log Entries", len(st.session_state.live_logs))
    else:
        pass
# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:var(--muted); padding:20px;">
        <p>🤖 Powered by CrewAI | Built with Streamlit</p>
        <p><small>Execution ID: {}</small></p>
    </div>
    """.format(st.session_state.execution_id or "None"),
    unsafe_allow_html=True,
)
