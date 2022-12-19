import pandas as pd
from typing import Optional, List

from presidio_evaluator import Span


class SpanOutput:
    def __init__(
        self,
        output_type: str,
        overlap_score: float,
        full_text: str,
        gold_span: Optional[Span] = None,
        pred_span: Optional[Span] = None
    ):
        """
        Holds information about model prediction output for analysis purposes
        :params 
        """
        self.output_type = output_type
        self.pred_span = pred_span
        self.gold_span = gold_span
        self.overlap_score = overlap_score
        self.full_text = full_text

    def __repr__(self):
        return (
            f"Output type: {self.output_type}\n"
            f"Overlap score: {self.overlap_score}\n"
            f"Full text: {self.full_text}\n"
            f"Gold span: {self.gold_span}\n"
            f"Predicted span: {self.pred_span}\n"
        )


    @staticmethod
    def get_spans_output_by_type(output_type:str, outputs=List["SpanOutput"], entity=None):
        """
        Get a list of all spans output by type and entity
        :params output_type: str, e.g. correct, partial, incorrect, suprious, miss
        :params outputs: List["SpanOutput"]: list of spans contains the model output's and gold spans
        :entity: str, e.g. PERSON, TITLE
        """
        filtered_output = []
        if isinstance(entity, str):
            entity = [entity]
        if entity:
            filtered_output = [
                output
                for output in outputs
                if output.output_type == output_type and output.pred_span.entity_type in entity
            ]
        else:
            filtered_output = [
                output
                for output in outputs
                if output.output_type == output_type
            ]
            
        if len(filtered_output) == 0:
            print( "No outputs of type {} and entity {} were found".format(
                    output_type, entity
                ))
        return filtered_output

    @staticmethod
    def get_span_output_df(outputs=List["SpanOutput"]):
        """
        Get SpanOutput as pd.DataFrame format
        """
        output_df = pd.DataFrame.from_records(
            [output.__dict__ for output in outputs]
        )
        return output_df
