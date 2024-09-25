The system comprises of a Filter classifier, which boost the image processing task by 
filtering out images without dimensional attributes; an OCR engine to extract texts from the image; a 
Parser to parse measurement values from the words de- tected by OCR and finally a model 
(Bounding-Box classi- fier) to classify the OCR-texts as a dimensional attribute.
```
## Filter Classifier: Filtration of images without dimensional information
Since one product may have multiple images and not all images have dimensional description. 
Hence, application of the extraction algorithm on all images is highly ineffi- cient and unnecessary. 
To reduce the OCR induced cost, we trained a MobileNetV3-L architecture to predict a binary output if 
the input image carries dimensional information in it. It is fed with a high resolution input image 
(600 pixels) so that it does not lose out the tiny text and thin dimensional axes/lines in the image, 
which are very important features to classify dimensional images.
```
## OCR engine: Detect Text from Images
I used the Rust OCR CLI tool for extracting text from images. It is very much efficient and faster
source Link : https://lib.rs/crates/ocrs-cli#:~:text=CLI%20tool%20for%20extracting%20text%20from%20images%20using%20the%20ocrs,the%20engine%20as%20a%20library
```
## Parser: Parse measurement values from the words detected by OCR
The parser is a simple python script that takes the output of the OCR engine and extracts the measurement values from the text.
```
## prediction model: Bounding-Box classifier
The model is a simple binary classifier that takes the output of the parser and classifies the text as a dimensional attribute or not.
```
## output csv => dataset/final_test_out.csv 
```
