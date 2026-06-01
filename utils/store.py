import requests
from datetime import datetime, timezone, timedelta

SGT = timezone(timedelta(hours=8))

def save_to_notion(notion_api_key: str, notion_data_source_id: str, notion_version: str, title: str, topic: str, body: str) -> dict: 
    url = "https://api.notion.com/v1/pages"

    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Content-Type": "application/json",
        "Notion-Version": notion_version,
    }

    payload = {
        "parent": {
            "data_source_id": notion_data_source_id
        },
        "properties": {
            "Doc name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
            "Topic": {
                "multi_select": [
                    {"name": topic}
                ]
            }
        },
        "markdown": body,
    }

    response = requests.post(url, headers=headers, json=payload)

    if not response.ok:
        raise RuntimeError(f"{response.status_code}: {response.text}")

    return response.json()
