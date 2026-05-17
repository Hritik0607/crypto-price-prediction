from typing import Optional

import pandas as pd


class DummyModel:
    """
    A dummy model that predicts the last known close price as the close price.
    N minutes into the future.
    """

    def __init__(
        self,
        from_feature: Optional[str] = 'close',
    ):
        """
        Initialize the dummy model. We store the name of the feature we will use as the prediction.

        Args:
            from_feature: The feature to use as the prediction.
        """
        self.from_feature = from_feature

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the next close price.
        """
        try:
            return data[self.from_feature]
        except Exception as e:
            raise Exception(f'Feature {self.from_feature} not found in data') from e
