from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    DirectoryReadTool,
    FileWriterTool,
    FileReadTool,
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv

_ = load_dotenv()
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)


class Content(BaseModel):
    content_type: str = Field(
        ...,
        description="The type of content to be created (e.g., blog post, social media post, video script)",
    )
    topic: str = Field(..., description="The topic of the content")
    target_audience: str = Field(..., description="The target audience for the content")
    tags: List[str] = Field(..., description="Tags to be used for the content")
    content: str = Field(..., description="The content itself")


class MarketResearch(BaseModel):
    market_trends: List[str] = Field(
        ..., description="Current market trends and opportunities"
    )
    competitors: List[str] = Field(
        ..., description="Top competitors and their strategies"
    )
    audience_insights: List[str] = Field(
        ..., description="Target audience insights and pain points"
    )


class ContentIdeas(BaseModel):
    ideas: List[dict] = Field(
        ...,
        description="List of content ideas with titles, angles, and recommendations",
    )


@CrewBase
class ContentCreationCrew:
    """Content creation crew for comprehensive marketing content workflow"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def market_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["market_research_agent"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
            ],
            reasoning=True,
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            # max_rpm=3,
        )

    @agent
    def content_ideation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["content_ideation_agent"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
                DirectoryReadTool("resources/drafts"),
                FileWriterTool(),
                FileReadTool(),
            ],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=30,
            max_rpm=3,
        )

    @agent
    def blog_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer_agent"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
                DirectoryReadTool("resources/drafts/blogs"),
                FileWriterTool(),
                FileReadTool(),
            ],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=5,
            max_rpm=3,
        )

    @agent
    def social_media_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["social_media_agent"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
                DirectoryReadTool("resources/drafts/blogs"),
                FileWriterTool(),
                FileReadTool(),
            ],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=5,
            max_rpm=3,
        )

    @agent
    def script_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["script_writer_agent"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
                DirectoryReadTool("resources/drafts/blogs"),
                FileWriterTool(),
                FileReadTool(),
            ],
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=5,
            max_rpm=3,
        )

    @task
    def market_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_research_task"],
            agent=self.market_research_agent(),
            output_pydantic=MarketResearch,
        )

    @task
    def content_ideation_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_ideation_task"],
            agent=self.content_ideation_agent(),
            context=[self.market_research_task()],
            output_pydantic=ContentIdeas,
        )

    @task
    def blog_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_writing_task"],
            agent=self.blog_writer_agent(),
            context=[self.market_research_task(), self.content_ideation_task()],
            output_pydantic=Content,
        )

    @task
    def social_media_task(self) -> Task:
        return Task(
            config=self.tasks_config["social_media_task"],
            agent=self.social_media_agent(),
            context=[self.blog_writing_task(), self.content_ideation_task()],
            output_pydantic=Content,
        )

    @task
    def script_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["script_writing_task"],
            agent=self.script_writer_agent(),
            context=[self.blog_writing_task(), self.content_ideation_task()],
            output_pydantic=Content,
        )

    @crew
    def content_crew(self) -> Crew:
        """Creates the Content Creation crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            planning_llm=llm,
        )


if __name__ == "__main__":
    from datetime import datetime

    inputs = {
        "topic": "AI-powered marketing automation for small businesses",
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }

    crew = ContentCreationCrew()
    result = crew.content_crew().kickoff(inputs=inputs)
    print("Content creation crew has been successfully created and run.")
    print("Results:", result)
