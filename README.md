# TrOCR for French

## Overview

TrOCR has not yet released for French, so we trained a French model for PoC purpose. Based on this model, it is recommended to collect more data to additionally train the 1st stage or perform fine-tuning as the 2nd stage.

It's a special case of the [English trOCR model](https://huggingface.co/microsoft/trocr-base-printed) introduced in the paper [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) by Li et al. and first released in [this repository](https://github.com/microsoft/unilm/tree/master/trocr)

This was possible thanks to [daekun-ml](https://huggingface.co/daekeun-ml/ko-trocr-base-nsmc-news-chatbot) and [Niels Rogge](https://github.com/NielsRogge/) than enabled us to publish this model with their tutorials and code.

## Collecting data

### Text data
We created training data of ~723k examples by taking random samples of the following datasets:

- [MultiLegalPile](https://huggingface.co/datasets/joelito/Multi_Legal_Pile) - 90k
- [French book Reviews](https://huggingface.co/datasets/Abirate/french_book_reviews) - 20k
- [WikiNeural](https://huggingface.co/datasets/Babelscape/wikineural) - 83k
- [Multilingual cc news](https://huggingface.co/datasets/intfloat/multilingual_cc_news) - 119k
- [Reviews Amazon Multi](https://huggingface.co/datasets/amazon_reviews_multi) - 153k
- [Opus Book](https://huggingface.co/datasets/opus_books) - 70k
- [BerlinText](https://huggingface.co/datasets/biglam/berlin_state_library_ocr) - 38k
  
We collected parts of each of the datasets and then cut randomly the sentences to collect the final training set.

### Image Data

Image data was generated with TextRecognitionDataGenerator (https://github.com/Belval/TextRecognitionDataGenerator) introduced in the TrOCR paper.
Below is a code snippet for generating images.

```shell
python3 ./trdg/run.py -i ocr_dataset_poc.txt -w 5 -t {num_cores} -f 64 -l ko -c {num_samples} -na 2 --output_dir {dataset_dir}
```

## Training

### Base model
The encoder model used `facebook/deit-base-distilled-patch16-384` and the decoder model used `camembert-base`. It is easier than training by starting weights from `microsoft/trocr-base-stage1`.

### Parameters
We used heuristic parameters without separate hyperparameter tuning.
- learning_rate = 4e-5
- epochs = 25
- fp16 = True
- max_length = 32

### Results on dev set

For the dev set we got those results
- size of the test set: 72k examples
- CER: 0.13
- WER: 0.26
- Val Loss: 0.424

## Usage

### inference.py

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests 
from io import BytesIO
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("agomberto/trocr-base-printed-fr")
tokenizer = AutoTokenizer.from_pretrained("agomberto/trocr-base-printed-fr")

url = "https://github.com/agombert/trocr-base-printed-fr/blob/main/sample_imgs/0.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

pixel_values = processor(img, return_tensors="pt").pixel_values 
generated_ids = model.generate(pixel_values, max_length=32)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
print(generated_text)
```