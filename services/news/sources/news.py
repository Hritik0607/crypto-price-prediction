from datetime import datetime

from pydantic import BaseModel


class News(BaseModel):
    """
    This is the data model for the news.
    """

    title: str
    published_at: str  # "2024-12-18T12:29:27Z"

    def to_dict(self) -> dict:
        return {
            **self.model_dump(),
            'timestamp_ms': int(
                datetime.fromisoformat(
                    self.published_at.replace('Z', '+00:00')
                ).timestamp()
                * 1000
            ),
        }
