from typing import List
import os
from crewai import Agent, Crew, Process, Task, LLM
from langchain_openai import AzureChatOpenAI
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

load_dotenv()

# llm = LLM(
#     model="azure/gpt-4o",  # LiteLLM format for Azure models
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     temperature=0.7,
# )

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)


class Content(BaseModel):
    content_type: str = Field(
        ...,
        description="The type of content to be created (e.g., blog post, social media post, video)",
    )
    topic: str = Field(..., description="The topic of the content")
    target_audience: str = Field(..., description="The target audience for the content")
    tags: List[str] = Field(..., description="Tags to be used for the content")
    content: str = Field(..., description="The content itself")


@CrewBase
class TheMarketingCrew:
    """The marketing crew is responsible for creating and executing marketing strategies, content creation, and managing marketing campaigns."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def market_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["market_research_agent"],
            tools=[
                SerperDevTool(),
                DirectoryReadTool("resources"),
                FileWriterTool(),
                FileReadTool(),
            ],
            reasoning=True,
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_iter=5,
            max_rpm=3,
        )

    @agent
    def marketing_strategy_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["marketing_strategy_agent"],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
                DirectoryReadTool("resources/research"),
                FileWriterTool(),
                FileReadTool(),
            ],
            reasoning=True,
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_rpm=3,
        )

    @agent
    def content_calendar_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["content_calendar_agent"],
            tools=[
                # SerperDevTool(),
                # ScrapeWebsiteTool(),
                DirectoryReadTool("resources/strategy"),
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
    def content_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["content_writer_agent"],
            tools=[
                # SerperDevTool(),
                # ScrapeWebsiteTool(),
                DirectoryReadTool("resources/calendar"),
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
    def seo_specialist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["seo_specialist_agent"],
            tools=[
                # SerperDevTool(),
                # ScrapeWebsiteTool(),
                DirectoryReadTool("resources/content"),
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
    def social_script_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["social_script_agent"],
            tools=[
                # SerperDevTool(),
                # ScrapeWebsiteTool(),
                DirectoryReadTool("resources/content"),
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
        )

    @task
    def marketing_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["marketing_strategy_task"],
            agent=self.marketing_strategy_agent(),
            context=[self.market_research_task()],
        )

    @task
    def content_calendar_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_calendar_task"],
            agent=self.content_calendar_agent(),
            context=[self.market_research_task(), self.marketing_strategy_task()],
        )

    @task
    def content_drafting_blogs_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_drafting_blogs_task"],
            agent=self.content_writer_agent(),
            context=[self.marketing_strategy_task(), self.content_calendar_task()],
        )

    @task
    def content_drafting_social_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_drafting_social_task"],
            agent=self.content_writer_agent(),
            context=[self.marketing_strategy_task(), self.content_calendar_task()],
        )

    @task
    def seo_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config["seo_optimization_task"],
            agent=self.seo_specialist_agent(),
            context=[
                self.content_drafting_blogs_task(),
                self.marketing_strategy_task(),
            ],
        )

    @task
    def script_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["script_generation_task"],
            agent=self.social_script_agent(),
            context=[
                self.content_drafting_blogs_task(),
                self.content_drafting_social_task(),
                self.marketing_strategy_task(),
            ],
        )

    @crew
    def marketingcrew(self) -> Crew:
        """Creates the Marketing crew with sequential workflow"""
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
        "product_name": "AI Powered Excel Automation Tool",
        "target_audience": "Small and Medium Enterprises (SMEs)",
        "product_description": "A tool that automates repetitive tasks in Excel using AI, saving time and reducing errors.",
        "budget": "Rs. 50,000",
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "industry": "Business Software",
        "campaign_duration": "3 months",
        "primary_goal": "Lead generation and brand awareness",
    }

    crew = TheMarketingCrew()
    result = crew.marketingcrew().kickoff(inputs=inputs)
    print("Marketing crew has been successfully created and run.")
    print(f"Final result: {result}")
