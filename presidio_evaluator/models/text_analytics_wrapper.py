from typing import List, Optional, Dict

from presidio_evaluator import InputSample, span_to_tag
from presidio_evaluator.models import BaseModel
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


class TextAnalyticsWrapper(BaseModel):
    def __init__(
        self,
        ta_client: Optional[TextAnalyticsClient] = None,
        ta_key: Optional[str] = "",
        ta_endpoint: Optional[str] = "",
        entities_to_keep: List[str] = None,
        verbose: bool = False,
        labeling_scheme: str = "BIO",
        score_threshold: float = 0.4,
        language: str = "en",
        entity_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Evaluation wrapper for the Presidio Analyzer
        :param ta_client: object of type TextAnalyticsClient (from presidio-analyzer)
        """
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )
        self.score_threshold = score_threshold
        self.language = language
        self.ta_key = ta_key
        self.ta_endpoint = ta_endpoint

        if not ta_client:
            ta_client = self.authenticate_client(ta_key, ta_endpoint)
            #self._update_recognizers_based_on_entities_to_keep(ta_client)
        self.ta_client = ta_client

    
    def authenticate_client(self, key: str, endpoint: str):
        ta_credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=ta_credential
        )
        return text_analytics_client


    def predict(self, sample: InputSample) -> List[str]:
        documents = [sample.full_text]
        response = self.ta_client.recognize_pii_entities(documents, 
                                                        language="en")
        results = [doc for doc in response if not doc.is_error]
        starts = []
        ends = []
        scores = []
        tags = []
        #
        for res in results:
            for entity in res.entities:
                starts.append(entity.offset)
                ends.append(entity.offset + len(entity.text))
                tags.append(entity.category)
                scores.append(entity.confidence_score)

        response_tags = span_to_tag(
            scheme="IO",
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tokens=sample.tokens,
            scores=scores,
            tags=tags,
        )
        return response_tags
    
    # Mapping between dataset entities and text analytics entities. Key: Dataset entity, Value: Presidio entity
    ta_entities_map = {
    }

