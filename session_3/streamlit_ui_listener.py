# streamlit_ui_listener.py

import queue
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    CrewKickoffCompletedEvent,
)
from typing import Dict, Any


class StreamlitUIListener(BaseEventListener):
    """
    A listener that captures CrewAI events and puts them into a thread-safe queue.
    This allows the Streamlit UI to update in real-time as the crew executes.
    """

    def __init__(self, ui_queue: queue.Queue):
        """
        Initializes the listener with a queue to send UI updates.

        Args:
            ui_queue: A thread-safe queue for sending event data to the Streamlit app.
        """
        super().__init__()
        self.ui_queue = ui_queue

    def _put_event(self, event_type: str, data: Dict[str, Any]):
        """A helper method to put a structured event into the queue."""
        self.ui_queue.put({"type": event_type, "data": data})

    def setup_listeners(self, crewai_event_bus):
        """Sets up the listeners for the CrewAI event bus."""

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            """Handles the event when an agent starts executing a task."""
            self._put_event(
                "agent_start",
                {
                    "agent": event.agent.role,
                },
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event: AgentExecutionCompletedEvent):
            """Handles the event when an agent completes a task."""
            self._put_event(
                "agent_end",
                {
                    "agent": event.agent.role,
                    "output": event.output.raw_output,
                },
            )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            """Handles the event when the entire crew completes its execution."""
            # The final output can be a complex object, so we convert it to a dict if possible
            final_output = (
                event.output.dict()
                if hasattr(event.output, "dict")
                else str(event.output)
            )
            self._put_event(
                "crew_end",
                {
                    "final_output": final_output,
                },
            )
