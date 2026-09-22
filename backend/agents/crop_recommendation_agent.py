"""
Crop Recommendation Agent — Week 3 Option A
Wraps the Week 2 ML module (crop_service) into a LangChain Tool.
"""
import os
import sys
from pathlib import Path

# Make sure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import Tool
from services.crop_service import predict_crop


def crop_recommendation_tool(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
) -> str:
    """
    Call this when a farmer asks for a crop recommendation.
    Returns a readable sentence with the crop name.
    """
    result = predict_crop(
        nitrogen, phosphorus, potassium,
        temperature, humidity, ph, rainfall
    )

    if "error" in result:
        return f"⚠️ Input error: {result['error']}"

    crop_name = result["recommended_crop"]
    confidence = result.get("confidence")
    return (
        f"✅ Based on your soil and weather conditions "
        f"(N={nitrogen}, P={phosphorus}, K={potassium}, "
        f"Temp={temperature}°C, Humidity={humidity}%, pH={ph}, Rain={rainfall}mm), "
        f"I recommend growing: **{crop_name.upper()}**. "
        f"Confidence: {confidence if confidence else 'N/A'}."
    )


# LangChain Tool object
crop_tool = Tool(
    name="CropRecommendation",
    func=crop_recommendation_tool,
    description=(
        "Recommends the best crop based on 7 farming inputs. "
        "Use when farmer provides: nitrogen (N), phosphorus (P), potassium (K), "
        "temperature, humidity, pH, rainfall. All must be numbers."
    ),
)
# ==============================
# Week 3 - AI Agent
# ==============================

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()


class CropRecommendationAgent:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[crop_recommendation_tool],
            system_prompt=(
                "You are a helpful Sustainable Farming AI assistant. "
                "Your job is to help farmers with crop recommendations. "
                "When the user provides soil and weather parameters, "
                "use the crop recommendation tool to get the prediction. "
                "Do not invent missing soil or weather values. "
                "If required information is missing, ask the user for it. "
                "After getting the tool result, explain the recommendation "
                "clearly and mention the confidence when available."
            )
        )

    def chat(self, user_message: str) -> str:
        response = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
        )

        return response["messages"][-1].content