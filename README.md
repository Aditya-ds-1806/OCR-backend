# OCR-backend
Flask API for OCR front end available at https://github.com/Aditya-ds-1806/OCR

Sometimes the request might timeout since I am using Heroku's free hosting services. You can try after some time.

# Usage

**Endpoint**: http://tesseract-ocr-backend.herokuapp.com

It has only one `POST` route which looks for the following parameters in the body:

`img`: image file (binary)

`alignment`: Perform document alignment `(1|0)`

`gaussian`: Apply Gaussian Blur `(1|0)`

`ed`: Erosion and dilation `(1|0)`

`median`: Median filtering `(1|0)`

# Example with cURL

```bash
curl -F "img=@/path/to/image.png" -F "alignment=1" -F "gaussian=1" -F "ed=1" -F "median=0" http://tesseract-ocr-backend.herokuapp.com
```
responds with:

```json
{
  "text": "some text here"
}
```

# Credits
OCR code modified from: https://github.com/v1gneshn/DocScanner
