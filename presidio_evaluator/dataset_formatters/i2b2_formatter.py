
from presidio_evaluator.data_objects import Span
from typing import List, Optional

import collections
import os
import xmltodict

from presidio_evaluator import InputSample
from presidio_evaluator.dataset_formatters import DatasetFormatter

class I2B22014Formatter(DatasetFormatter):
  def __init__(
        self,
        files_path="../../data/i2b2",        
    ):
        self.files_path = files_path        

  def _create_span(self, item):        
    span = Span(entity_type=item['@TYPE'], 
                entity_value=item['@text'], 
                start_position=int(item['@start']),
                end_position=int(item['@end']))    
    return span

  def to_input_samples(self, folder: Optional[str] = None) -> List[InputSample]:      
      input_samples = []
      if folder:
        self.files_path = folder
      print(f"Parsing files in {self.files_path}")

      for root, dirs, files in os.walk(self.files_path):
        for file in files:
          spans = []
          filename = os.path.join(root,file)          
          xml_content = open(filename,"r").read()
                    
          ordered_dict = xmltodict.parse(xml_content)
          data = dict(ordered_dict['deIdi2b2'])
          text = data['TEXT']
          tags = data['TAGS']
          for item in tags.items():                        
            if type(item[1]) is collections.OrderedDict:              
              spans.append(self._create_span(item[1]))                                                       
            else:
              for sub in item[1]:
                spans.append(self._create_span(sub))                    
          input_samples.append(InputSample(full_text=text, spans=spans, create_tags_from_span=True))                            
      return input_samples
        

if __name__ == "__main__":
    formatter = I2B22014Formatter()    
    train_samples = formatter.to_input_samples()
    print(train_samples)