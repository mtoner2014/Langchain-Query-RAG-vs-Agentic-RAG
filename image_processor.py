"""
Medical Image Analysis Module
Extracts medical context from uploaded images using gpt-4o-mini vision.
"""

import base64
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
MAX_IMAGE_SIZE_MB = 5

IMAGE_ANALYSIS_PROMPT = """You are a medical image analysis assistant. Analyze this medical image and extract information.

Determine the image type and respond in this EXACT format:

IMAGE_TYPE: [one of: lab_report, prescription, symptom_photo, other]
EXTRACTED_TEXT: [All readable text from the image, preserving numbers and values]
MEDICAL_TERMS: [Comma-separated list of medical terms, drug names, test names, conditions visible]
SUMMARY: [2-3 sentence plain-language summary of what this image shows]
SEARCH_QUERY: [A concise search query (3-8 words) to look up relevant health information about the key topics in this image]

IMPORTANT:
- For lab reports: extract test names, values, reference ranges, and flag abnormal results
- For prescriptions: extract drug names, dosages, frequencies, and instructions
- For symptom photos: describe visible symptoms, location, characteristics (color, size, texture)
- If the image is unclear or not medical, say so in SUMMARY and set IMAGE_TYPE to other
- Do NOT diagnose. Only describe and extract."""

IMAGE_WITH_TEXT_PROMPT = """You are a medical image analysis assistant. The user uploaded a medical image and also wrote: "{user_text}"

Analyze the image in the context of their message and respond in this EXACT format:

IMAGE_TYPE: [one of: lab_report, prescription, symptom_photo, other]
EXTRACTED_TEXT: [All readable text from the image, preserving numbers and values]
MEDICAL_TERMS: [Comma-separated list of medical terms, drug names, test names, conditions visible]
SUMMARY: [2-3 sentence plain-language summary of what this image shows]
SEARCH_QUERY: [A concise search query (3-8 words) to look up relevant health information, incorporating the user's question]

IMPORTANT:
- For lab reports: extract test names, values, reference ranges, and flag abnormal results
- For prescriptions: extract drug names, dosages, frequencies, and instructions
- For symptom photos: describe visible symptoms, location, characteristics (color, size, texture)
- If the image is unclear or not medical, say so in SUMMARY and set IMAGE_TYPE to other
- Do NOT diagnose. Only describe and extract."""


class ImageAnalysisResult(BaseModel):
    """Structured output schema for medical image analysis."""
    image_type: str = Field(description="One of: lab_report, prescription, symptom_photo, other")
    extracted_text: str = Field(description="All readable text from the image, preserving numbers and values")
    medical_terms: List[str] = Field(description="List of medical terms, drug names, test names, conditions visible")
    summary: str = Field(description="2-3 sentence plain-language summary of what this image shows")
    search_query: str = Field(description="A concise search query (3-8 words) to look up relevant health information about the key topics in this image")


class MedicalImageAnalyzer:
    """Analyzes medical images using gpt-4o-mini vision capabilities."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(ImageAnalysisResult)

    def analyze_image(
        self, image_bytes: bytes, mime_type: str, user_text: str = ""
    ) -> dict:
        """
        Analyze a medical image and extract structured information.

        Returns dict with keys: image_type, extracted_text, medical_terms,
                                summary, search_query
        """
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        if user_text.strip():
            prompt_text = IMAGE_WITH_TEXT_PROMPT.format(user_text=user_text)
        else:
            prompt_text = IMAGE_ANALYSIS_PROMPT

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                        "detail": "high",
                    },
                },
            ]
        )

        result = self.structured_llm.invoke([message])

        # Fallback: derive search_query from medical_terms if empty
        response = result.model_dump()
        if not response["search_query"] and response["medical_terms"]:
            response["search_query"] = " ".join(response["medical_terms"][:3])

        return response


def validate_upload(file_name: str, file_size: int) -> str | None:
    """Validate an uploaded file. Returns error message or None if valid."""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '.{ext}'. "
            f"Please upload: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        return (
            f"Image too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Maximum is {MAX_IMAGE_SIZE_MB} MB."
        )
    return None
