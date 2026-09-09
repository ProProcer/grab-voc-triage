from src.models.base import ReviewClassificationModel
from src.schemas.prediction import ReviewClassification
from openai import OpenAI
import config

class OpenAIModel(ReviewClassificationModel):
    def __init__(self, system_prompt : str, model : str):
        self.system_prompt = system_prompt
        self.model = model
        self.client = OpenAI()
    def classify(self, text) -> dict:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.0, 
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f'Review: "{text}"'}
                ],
                response_format=ReviewClassification,
            )
            return response.choices[0].message.parsed.model_dump()
        except Exception as e:
            print(f"Inference error on review: {e}")
            return {
                c : "ERROR" for c in config.CATEGORIES
            }