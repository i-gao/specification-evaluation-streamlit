
from data.database import Database
import pandas as pd
import json
import os
from PIL import Image
import requests

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))

class Catalog(Database):
    def __init__(self, image_url = None):
        # Load the catalog data
        df = pd.read_csv(f"{DATASET_ROOT}/assets/catalog.csv")

        # Load column descriptions
        column_descriptions = json.load(
            open(f"{DATASET_ROOT}/assets/column_descriptions.json")
        )

        # Initialize the Database parent class
        super().__init__(
            {
                "catalog": (
                    "Catalog of products from H&M",
                    df,
                    column_descriptions,
                )
            }
        )

        # Store the dataframe for backward compatibility
        self.df = df
        self.image_url = image_url

    def get_row_by_article_id(self, article_id: int) -> pd.Series:
        article_id = int(article_id)
        matches = self.df[self.df["article_id"] == article_id]
        if len(matches) == 0:
            raise ValueError(f"Catalog ID {article_id} not found in catalog")
        return matches.iloc[0]

    def get_image_by_article_id(self, article_id: int) -> Image.Image:
        article_id = str(article_id)

        if self.image_url is not None:
            return Image.open(requests.get(self.image_url).content)
        
        try:
            img_path = (
                f"{DATASET_ROOT}/assets/images/0{article_id[:2]}/0{article_id}.jpg"
            )
            return Image.open(img_path)
        except FileNotFoundError:
            return None
